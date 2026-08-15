//! Byte fetchers. Production uses HTTPS; tests inject a fake or a blocked client.

use std::io::{Read, Write};

use crate::cancel::Cancel;
use crate::error::{Error, Result};

/// User-Agent Hugging Face and GitHub accept. Also identifies the product.
pub const USER_AGENT: &str = "syllabix/0.1.0 (+https://github.com/syllabix-ai/syllabix)";

/// Source of asset bytes. Tests replace this to avoid the public internet.
pub trait Fetcher: Send + Sync {
    /// Write the body of `url` into `writer`, checking `cancel` between chunks.
    fn fetch(
        &self,
        url: &str,
        writer: &mut dyn Write,
        on_chunk: &mut dyn FnMut(u64),
        cancel: &Cancel,
    ) -> Result<()>;
}

/// HTTPS GET with rustls. No OpenSSL / libssl.
#[derive(Debug, Default, Clone, Copy)]
pub struct HttpFetcher;

impl HttpFetcher {
    /// rustls-backed client.
    pub fn new() -> Self {
        Self
    }
}

impl Fetcher for HttpFetcher {
    fn fetch(
        &self,
        url: &str,
        writer: &mut dyn Write,
        on_chunk: &mut dyn FnMut(u64),
        cancel: &Cancel,
    ) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let response = ureq::get(url)
            .set("User-Agent", USER_AGENT)
            .call()
            .map_err(|err| Error::ModelCache {
                message: format!("download failed: {err}"),
            })?;
        let mut reader = response.into_reader();
        let mut buf = [0u8; 64 * 1024];
        let mut copied = 0u64;
        loop {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            let n = reader.read(&mut buf).map_err(|err| Error::ModelCache {
                message: format!("read failed: {err}"),
            })?;
            if n == 0 {
                break;
            }
            writer.write_all(&buf[..n])?;
            copied += n as u64;
            on_chunk(copied);
        }
        writer.flush()?;
        Ok(())
    }
}

/// Always fails. Used to prove a populated cache does not touch the network.
#[derive(Debug, Default)]
pub struct BlockedFetcher {
    /// How many times `fetch` was called.
    pub hits: std::sync::atomic::AtomicUsize,
}

impl Fetcher for BlockedFetcher {
    fn fetch(
        &self,
        url: &str,
        _writer: &mut dyn Write,
        _on_chunk: &mut dyn FnMut(u64),
        _cancel: &Cancel,
    ) -> Result<()> {
        self.hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Err(Error::ModelCache {
            message: format!("network blocked ({url})"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn serve_body(body: &'static [u8]) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut req = [0u8; 2048];
            let _ = stream.read(&mut req);
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(header.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
        });
        (format!("http://{addr}/asset.bin"), handle)
    }

    #[test]
    fn http_fetcher_copies_local_body() {
        let (url, handle) = serve_body(b"hello-weights");
        let mut out = Vec::new();
        let mut last = 0u64;
        HttpFetcher::new()
            .fetch(&url, &mut out, &mut |n| last = n, &Cancel::new())
            .unwrap();
        handle.join().unwrap();
        assert_eq!(out, b"hello-weights");
        assert_eq!(last, out.len() as u64);
    }

    #[test]
    fn http_fetcher_maps_connect_errors() {
        let err = HttpFetcher::new()
            .fetch(
                "http://127.0.0.1:1/missing",
                &mut Vec::new(),
                &mut |_| {},
                &Cancel::new(),
            )
            .unwrap_err();
        assert!(matches!(err, Error::ModelCache { .. }), "{err:?}");
        assert!(err.to_string().contains("download failed"), "{err}");
    }

    #[test]
    fn http_fetcher_honors_cancel_before_connect() {
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = HttpFetcher::new()
            .fetch(
                "http://127.0.0.1:1/x",
                &mut Vec::new(),
                &mut |_| {},
                &cancel,
            )
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn blocked_fetcher_records_hits() {
        let fetcher = BlockedFetcher::default();
        let err = fetcher
            .fetch(
                "https://example.invalid",
                &mut Vec::new(),
                &mut |_| {},
                &Cancel::new(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("network blocked"));
        assert_eq!(fetcher.hits.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
