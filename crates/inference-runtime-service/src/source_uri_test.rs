use std::io::Read;
use std::io::Write;
use std::net::TcpListener;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use inference_runtime_core::Error;
use reqwest::Url;
use tempfile::NamedTempFile;

use super::SourceURIReader;

#[test]
fn test_read_data() {
    let reader = SourceURIReader::new(16).unwrap();

    assert_eq!(reader.read("data:image/png;BASE64,aW1hZ2U=").unwrap(), b"image");
    assert_eq!(reader.read("data:video/mp4;base64,dmlkZW8=").unwrap(), b"video");
}

#[test]
fn test_read_file() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"local source").unwrap();
    let uri = Url::from_file_path(file.path()).unwrap();
    let reader = SourceURIReader::new(16).unwrap();

    assert_eq!(reader.read(uri.as_str()).unwrap(), b"local source");
}

#[test]
fn test_read_http() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0u8; 1024];
        let num_request_bytes = stream.read(&mut request).unwrap();
        assert!(num_request_bytes > 0);
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\nConnection: close\r\n\r\nremote source")
            .unwrap();
    });
    let reader = SourceURIReader::new(16).unwrap();

    assert_eq!(reader.read(&format!("http://{addr}/source")).unwrap(), b"remote source");
    server.join().unwrap();
}

#[test]
fn test_read_size() {
    let bytes = STANDARD.encode([0u8; 17]);
    let reader = SourceURIReader::new(16).unwrap();

    assert!(matches!(
        reader.read(&format!("data:application/octet-stream;base64,{bytes}")),
        Err(Error::InvalidArgument(message)) if message.contains("must not exceed")
    ));
}
