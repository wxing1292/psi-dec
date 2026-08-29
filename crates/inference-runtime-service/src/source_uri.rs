use std::fs::File;
use std::io::Read;
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use inference_runtime_core::Error;
use inference_runtime_core::Result;
use reqwest::Url;
use reqwest::blocking::Client;
use reqwest::redirect::Policy;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

pub struct SourceURIReader {
    http: Client,
    max_source_bytes: u64,
    read_limit: u64,
}

impl SourceURIReader {
    pub fn new(max_source_bytes: usize) -> Result<Self> {
        assert!(
            max_source_bytes < usize::MAX,
            "source byte limit must leave room for the overflow probe"
        );
        let http = Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .redirect(Policy::limited(5))
            .build()
            .map_err(|error| Error::unavailable(format!("unable to initialize source HTTP client: {error}")))?;
        Ok(Self {
            http,
            max_source_bytes: max_source_bytes as u64,
            read_limit: (max_source_bytes + 1) as u64,
        })
    }

    pub fn read(&self, uri: &str) -> Result<Vec<u8>> {
        let url =
            Url::parse(uri).map_err(|error| Error::invalid_argument(format!("source URI is invalid: {error}")))?;
        match url.scheme() {
            "data" => self.read_data(uri),
            "file" => self.read_file(&url),
            "http" | "https" => self.read_http(url),
            scheme => {
                Err(Error::invalid_argument(format!(
                    "source URI scheme {scheme:?} is not supported"
                )))
            },
        }
    }

    fn read_data(&self, uri: &str) -> Result<Vec<u8>> {
        let (metadata, encoded) = uri
            .split_once(',')
            .ok_or_else(|| Error::invalid_argument("source data URI is incomplete"))?;
        let mut parameters = metadata
            .split_once(':')
            .expect("parsed data URI must contain a scheme separator")
            .1
            .split(';');
        parameters.next();
        if !parameters.any(|parameter| parameter.eq_ignore_ascii_case("base64")) {
            return Err(Error::invalid_argument("source data URI must use base64 encoding"));
        }
        let bytes = STANDARD
            .decode(encoded)
            .map_err(|_| Error::invalid_argument("source data URI contains invalid base64 data"))?;
        self.validate_len(bytes.len() as u64)?;
        Ok(bytes)
    }

    fn read_file(&self, url: &Url) -> Result<Vec<u8>> {
        if url.query().is_some() || url.fragment().is_some() {
            return Err(Error::invalid_argument(
                "source file URI must not contain a query or fragment",
            ));
        }
        let path = url
            .to_file_path()
            .map_err(|()| Error::invalid_argument("source file URI is invalid"))?;
        let file = File::open(&path)
            .map_err(|error| Error::invalid_argument(format!("unable to open source file {path:?}: {error}")))?;
        let len = file
            .metadata()
            .map_err(|error| Error::unavailable(format!("unable to inspect source file {path:?}: {error}")))?
            .len();
        self.validate_len(len)?;
        self.read_limited(file)
    }

    fn read_http(&self, url: Url) -> Result<Vec<u8>> {
        let response = self
            .http
            .get(url)
            .send()
            .map_err(|error| Error::unavailable(format!("unable to fetch source URI: {error}")))?;
        let status = response.status();
        if status.is_client_error() {
            return Err(Error::invalid_argument(format!("source URI returned HTTP {status}")));
        }
        if !status.is_success() {
            return Err(Error::unavailable(format!("source URI returned HTTP {status}")));
        }
        if let Some(len) = response.content_length() {
            self.validate_len(len)?;
        }
        self.read_limited(response)
    }

    fn read_limited(&self, reader: impl Read) -> Result<Vec<u8>> {
        let mut bytes = vec![];
        reader
            .take(self.read_limit)
            .read_to_end(&mut bytes)
            .map_err(|error| Error::unavailable(format!("unable to read source: {error}")))?;
        self.validate_len(bytes.len() as u64)?;
        Ok(bytes)
    }

    fn validate_len(&self, len: u64) -> Result<()> {
        if len > self.max_source_bytes {
            return Err(Error::invalid_argument(format!(
                "source must not exceed {} bytes",
                self.max_source_bytes
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "source_uri_test.rs"]
mod tests;
