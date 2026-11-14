//! Tiny NumPy `.npy` v1/v2 reader for the little-endian C-order arrays Hima ships.

use syllabix_native::error::{Error, Result};

const MAGIC: &[u8; 6] = b"\x93NUMPY";

fn npy_err(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: "npy",
        message: message.into(),
    }
}

/// Promoted float array (`<f4` as-is, `<f2` widened to f32).
#[derive(Debug, Clone, PartialEq)]
pub struct NpyF32 {
    /// C-order shape.
    pub shape: Vec<usize>,
    /// Length is the product of `shape`.
    pub data: Vec<f32>,
}

/// Little-endian i64 array (`<i8`).
#[derive(Debug, Clone, PartialEq)]
pub struct NpyI64 {
    /// C-order shape.
    pub shape: Vec<usize>,
    /// Length is the product of `shape`.
    pub data: Vec<i64>,
}

/// Parsed `.npy` payload for the dtypes Hima uses.
#[derive(Debug, Clone, PartialEq)]
pub enum NpyArray {
    /// `<f4`.
    F32(NpyF32),
    /// `<f2` promoted to f32.
    F16(NpyF32),
    /// `<i8`.
    I64(NpyI64),
}

/// Read a v1 or v2 `.npy` buffer. Rejects Fortran order, big-endian, and other dtypes.
pub fn read(bytes: &[u8]) -> Result<NpyArray> {
    let (descr, shape, payload) = parse_header(bytes)?;
    let n = element_count(&shape)?;
    match descr {
        "<f4" => {
            let data = read_f32_payload(payload, n)?;
            Ok(NpyArray::F32(NpyF32 { shape, data }))
        }
        "<f2" => {
            let data = read_f16_payload(payload, n)?;
            Ok(NpyArray::F16(NpyF32 { shape, data }))
        }
        "<i8" => {
            let data = read_i64_payload(payload, n)?;
            Ok(NpyArray::I64(NpyI64 { shape, data }))
        }
        other => Err(npy_err(format!(
            "unsupported npy dtype {other} (need little-endian f32, f16, or i64)"
        ))),
    }
}

/// Read `<f4` C-order data.
pub fn read_f32(bytes: &[u8]) -> Result<NpyF32> {
    match read(bytes)? {
        NpyArray::F32(arr) => Ok(arr),
        other => Err(npy_err(format!(
            "expected little-endian f32 npy, got {}",
            other.dtype_name()
        ))),
    }
}

/// Read `<f2` C-order data, promoted to f32.
pub fn read_f16_as_f32(bytes: &[u8]) -> Result<NpyF32> {
    match read(bytes)? {
        NpyArray::F16(arr) => Ok(arr),
        other => Err(npy_err(format!(
            "expected little-endian f16 npy, got {}",
            other.dtype_name()
        ))),
    }
}

/// Read `<i8` C-order data.
pub fn read_i64(bytes: &[u8]) -> Result<NpyI64> {
    match read(bytes)? {
        NpyArray::I64(arr) => Ok(arr),
        other => Err(npy_err(format!(
            "expected little-endian i64 npy, got {}",
            other.dtype_name()
        ))),
    }
}

impl NpyArray {
    fn dtype_name(&self) -> &'static str {
        match self {
            Self::F32(_) => "f32",
            Self::F16(_) => "f16",
            Self::I64(_) => "i64",
        }
    }
}

fn parse_header(bytes: &[u8]) -> Result<(&'static str, Vec<usize>, &[u8])> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(npy_err("not a NumPy .npy file"));
    }
    let major = bytes[6];
    let minor = bytes[7];
    let (header_len, header_start) = match (major, minor) {
        (1, 0) => {
            if bytes.len() < 10 {
                return Err(npy_err("truncated npy v1 header"));
            }
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10usize)
        }
        (2, 0) => {
            if bytes.len() < 12 {
                return Err(npy_err("truncated npy v2 header"));
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12usize)
        }
        (maj, min) => {
            return Err(npy_err(format!(
                "unsupported npy version {maj}.{min} (need 1.0 or 2.0)"
            )));
        }
    };
    let header_end = header_start
        .checked_add(header_len)
        .ok_or_else(|| npy_err("npy header length overflow"))?;
    if bytes.len() < header_end {
        return Err(npy_err("truncated npy header"));
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|_| npy_err("npy header is not UTF-8"))?;
    if parse_fortran_order(header)? {
        return Err(npy_err("npy fortran_order must be False (C-order only)"));
    }
    let descr = parse_descr(header)?;
    let shape = parse_shape(header)?;
    Ok((descr, shape, &bytes[header_end..]))
}

fn parse_descr(header: &str) -> Result<&'static str> {
    let rest = after_key(header, "descr")?;
    let token = quoted_token(rest)?;
    match token {
        "<f4" => Ok("<f4"),
        "<f2" => Ok("<f2"),
        "<i8" => Ok("<i8"),
        other => Err(npy_err(format!(
            "unsupported npy dtype {other} (need little-endian f32, f16, or i64)"
        ))),
    }
}

fn parse_fortran_order(header: &str) -> Result<bool> {
    let rest = after_key(header, "fortran_order")?;
    let token = ident_token(rest)?;
    match token {
        "False" => Ok(false),
        "True" => Ok(true),
        other => Err(npy_err(format!("bad npy fortran_order {other}"))),
    }
}

fn parse_shape(header: &str) -> Result<Vec<usize>> {
    let rest = after_key(header, "shape")?;
    let start = rest
        .find('(')
        .ok_or_else(|| npy_err("npy shape is missing '('"))?;
    let end = rest[start..]
        .find(')')
        .ok_or_else(|| npy_err("npy shape is missing ')'"))?;
    let inner = rest[start + 1..start + end].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| npy_err(format!("bad npy shape dim {part}")))
        })
        .collect()
}

fn after_key<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let single = format!("'{key}':");
    let double = format!("\"{key}\":");
    if let Some(pos) = header.find(&single) {
        Ok(&header[pos + single.len()..])
    } else if let Some(pos) = header.find(&double) {
        Ok(&header[pos + double.len()..])
    } else {
        Err(npy_err(format!("npy header missing {key}")))
    }
}

fn quoted_token(s: &str) -> Result<&str> {
    let s = s.trim_start();
    let quote = s
        .chars()
        .next()
        .filter(|c| *c == '\'' || *c == '"')
        .ok_or_else(|| npy_err("npy descr is not a quoted string"))?;
    let rest = &s[quote.len_utf8()..];
    let end = rest
        .find(quote)
        .ok_or_else(|| npy_err("unterminated npy descr"))?;
    Ok(&rest[..end])
}

fn ident_token(s: &str) -> Result<&str> {
    let s = s.trim_start();
    let end = s
        .find(|c: char| !c.is_ascii_alphabetic())
        .unwrap_or(s.len());
    if end == 0 {
        return Err(npy_err("npy fortran_order is missing"));
    }
    Ok(&s[..end])
}

fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| npy_err("npy shape overflow"))
    })
}

fn read_f32_payload(payload: &[u8], n: usize) -> Result<Vec<f32>> {
    let need = n
        .checked_mul(4)
        .ok_or_else(|| npy_err("npy f32 payload overflow"))?;
    if payload.len() != need {
        return Err(npy_err(format!(
            "npy f32 payload is {} bytes, expected {need}",
            payload.len()
        )));
    }
    let mut out = Vec::with_capacity(n);
    for chunk in payload.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn read_f16_payload(payload: &[u8], n: usize) -> Result<Vec<f32>> {
    let need = n
        .checked_mul(2)
        .ok_or_else(|| npy_err("npy f16 payload overflow"))?;
    if payload.len() != need {
        return Err(npy_err(format!(
            "npy f16 payload is {} bytes, expected {need}",
            payload.len()
        )));
    }
    let mut out = Vec::with_capacity(n);
    for chunk in payload.chunks_exact(2) {
        out.push(f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
    }
    Ok(out)
}

fn read_i64_payload(payload: &[u8], n: usize) -> Result<Vec<i64>> {
    let need = n
        .checked_mul(8)
        .ok_or_else(|| npy_err("npy i64 payload overflow"))?;
    if payload.len() != need {
        return Err(npy_err(format!(
            "npy i64 payload is {} bytes, expected {need}",
            payload.len()
        )));
    }
    let mut out = Vec::with_capacity(n);
    for chunk in payload.chunks_exact(8) {
        out.push(i64::from_le_bytes(chunk.try_into().expect("8 bytes")));
    }
    Ok(out)
}

/// IEEE-754 binary16 → f32 for packed `embed.npy` rows.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15);
    let exp = (bits >> 10) & 0x1f;
    let frac = u32::from(bits & 0x3ff);
    let out = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = frac;
            while m < 0x400 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((u32::from(exp) - 15 + 127) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_header(major: u8, descr: &str, fortran: bool, shape: &str, data: &[u8]) -> Vec<u8> {
        let order = if fortran { "True" } else { "False" };
        let dict = format!("{{'descr': '{descr}', 'fortran_order': {order}, 'shape': {shape}, }}");
        let mut header = dict.into_bytes();
        header.push(b'\n');
        let mut out = Vec::from(*MAGIC);
        out.push(major);
        out.push(0);
        if major == 1 {
            let len = u16::try_from(header.len()).expect("v1 header");
            out.extend_from_slice(&len.to_le_bytes());
        } else {
            let len = u32::try_from(header.len()).expect("v2 header");
            out.extend_from_slice(&len.to_le_bytes());
        }
        out.extend_from_slice(&header);
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn reads_little_endian_f32_c_order() {
        let mut data = Vec::new();
        for v in [1.0f32, -2.5, 0.0, 4.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let bytes = encode_header(1, "<f4", false, "(2, 2)", &data);
        let arr = read_f32(&bytes).expect("f32");
        assert_eq!(arr.shape, [2, 2]);
        assert_eq!(arr.data, [1.0, -2.5, 0.0, 4.0]);
    }

    #[test]
    fn reads_little_endian_f16_as_f32() {
        // 1.0, -2.0 as IEEE f16.
        let data: Vec<u8> = [0x00, 0x3c, 0x00, 0xc0].into();
        let bytes = encode_header(1, "<f2", false, "(2,)", &data);
        let arr = read_f16_as_f32(&bytes).expect("f16");
        assert_eq!(arr.shape, [2]);
        assert_eq!(arr.data[0], 1.0);
        assert_eq!(arr.data[1], -2.0);
    }

    #[test]
    fn reads_v2_i64() {
        let mut data = Vec::new();
        data.extend_from_slice(&42i64.to_le_bytes());
        data.extend_from_slice(&(-7i64).to_le_bytes());
        let bytes = encode_header(2, "<i8", false, "(2,)", &data);
        let arr = read_i64(&bytes).expect("i64");
        assert_eq!(arr.data, [42, -7]);
    }

    #[test]
    fn rejects_a_bad_header() {
        let err = read(b"not npy").expect_err("magic");
        assert!(err.to_string().contains("not a NumPy .npy file"), "{err}");

        let fortran = encode_header(1, "<f4", true, "(1,)", &0f32.to_le_bytes());
        let err = read(&fortran).expect_err("fortran");
        assert!(err.to_string().contains("C-order"), "{err}");

        let be = encode_header(1, ">f4", false, "(1,)", &0f32.to_be_bytes());
        let err = read(&be).expect_err("endian");
        assert!(err.to_string().contains("unsupported npy dtype"), "{err}");
    }

    #[test]
    fn typed_readers_reject_the_wrong_dtype() {
        let bytes = encode_header(1, "<f4", false, "(1,)", &1f32.to_le_bytes());
        assert!(read_f16_as_f32(&bytes).is_err());
        assert!(read_i64(&bytes).is_err());
        let f16 = encode_header(1, "<f2", false, "(1,)", &[0x00, 0x3c]);
        assert!(read_f32(&f16).is_err());
    }

    #[test]
    fn f16_subnormal_and_inf_promote() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
        let sub = f16_to_f32(0x0001);
        assert!(sub > 0.0 && sub < 1.0e-4);
    }

    #[test]
    fn rejects_truncated_payload_and_version() {
        let mut bytes = encode_header(1, "<f4", false, "(2,)", &1f32.to_le_bytes());
        bytes.pop();
        assert!(read(&bytes).is_err());
        let mut v2 = encode_header(2, "<i8", false, "(1,)", &0i64.to_le_bytes());
        v2.truncate(11);
        assert!(read(&v2).is_err());
        let mut v3 = encode_header(1, "<f4", false, "(1,)", &0f32.to_le_bytes());
        v3[6] = 3;
        let err = read(&v3).expect_err("v3");
        assert!(err.to_string().contains("unsupported npy version"), "{err}");
        let f16 = encode_header(1, "<f2", false, "(2,)", &[0x00, 0x3c]);
        assert!(read(&f16).is_err());
    }

    #[test]
    fn reads_zero_d_and_rejects_a_missing_key() {
        let scalar = encode_header(1, "<f4", false, "()", &1.5f32.to_le_bytes());
        let arr = read_f32(&scalar).expect("0-d");
        assert!(arr.shape.is_empty());
        assert_eq!(arr.data, [1.5]);

        let original = encode_header(1, "<f4", false, "(1,)", &0f32.to_le_bytes());
        let header_len = u16::from_le_bytes([original[8], original[9]]) as usize;
        let header = String::from_utf8(original[10..10 + header_len].to_vec())
            .expect("utf8")
            .replace("'descr':", "'kind':");
        let mut rebuilt = original[..8].to_vec();
        let len = u16::try_from(header.len()).unwrap();
        rebuilt.extend_from_slice(&len.to_le_bytes());
        rebuilt.extend_from_slice(header.as_bytes());
        rebuilt.extend_from_slice(&0f32.to_le_bytes());
        let err = read(&rebuilt).expect_err("missing descr");
        assert!(err.to_string().contains("missing descr"), "{err}");
    }
}
