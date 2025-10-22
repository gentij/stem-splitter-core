use crate::error::Result;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

pub fn verify_sha256(path: &Path, expected_hex: &str) -> Result<bool> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let got = hex::encode(hasher.finalize());
    Ok(got.eq_ignore_ascii_case(expected_hex))
}
