use crate::{error::Result, io::progress::emit_download_progress};
use reqwest::blocking::Client;
use std::{
    fs,
    fs::File,
    io::{Read, Write},
    path::Path,
    time::Duration,
};

pub fn http_client() -> Client {
    Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60 * 60))
        .build()
        .expect("reqwest client build failed")
}

pub fn download_with_progress(client: &Client, url: &str, dest: &Path) -> Result<()> {
    let tmp = dest.with_extension("part");

    let mut resp = client.get(url).send()?.error_for_status()?; // both now convert automatically

    let total = resp.content_length().unwrap_or(0);

    emit_download_progress(0, total);

    let mut file = File::create(&tmp)?;
    let mut downloaded: u64 = 0;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
        emit_download_progress(downloaded, total);
    }
    file.flush()?;

    if dest.exists() {
        fs::remove_file(dest).ok();
    }

    fs::rename(&tmp, dest)?;

    emit_download_progress(total.max(downloaded), total.max(downloaded));

    Ok(())
}
