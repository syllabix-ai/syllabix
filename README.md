# Syllabix

Local voice agent. One native binary. Apache-2.0.

v0 target: a stranger downloads `syllabix`, runs it, and is in a voice conversation on their laptop in under 3 minutes — no Python, pip, or API key.

This repository is the product. Founder docs live in [`syllabix-ai/syllabix_founder_documents`](https://github.com/syllabix-ai/syllabix_founder_documents). The launch contract is `V0_LAUNCH.md`.

```bash
cargo run -p syllabix -- --help
cargo run -p syllabix -- run    # exits 2 until local audio lands
```

## Intended 3-minute path (not shipping yet)

```bash
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-$(uname -s)-$(uname -m) -o syllabix
chmod +x syllabix
./syllabix run
```

GitHub Releases are not published yet. Do not expect a spoken reply from this checkout.

## CLI

| Command | Now | Launch |
| --- | --- | --- |
| `syllabix run` | Not implemented | Zero-config local mic/speaker conversation |
| `syllabix init [dir]` | Not implemented | Optional `syllabix.yaml` scaffold |

There is no `serve`, `bench`, cloud provider, or API key in v0.

## Develop

Requires Rust 1.83+.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## License

Apache-2.0. See [LICENSE](LICENSE).
