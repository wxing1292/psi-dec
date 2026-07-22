use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;

use crate::def::ModelExecutorError;

pub struct MappedFile {
    ptr: *mut libc::c_void,
    len: usize,
}

impl MappedFile {
    pub fn open(path: &Path) -> Result<Self, ModelExecutorError> {
        let file = File::open(path).map_err(|err| {
            ModelExecutorError::custom(format!("unable to open safetensors file {:?}, err: {err:?}", path))
        })?;
        let len = file
            .metadata()
            .map_err(|err| {
                ModelExecutorError::custom(format!("unable to stat safetensors file {:?}, err: {err:?}", path))
            })?
            .len() as usize;
        if len == 0 {
            return Err(ModelExecutorError::custom(format!(
                "safetensors file {:?} must not be empty",
                path
            )));
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(ModelExecutorError::custom(format!(
                "unable to mmap safetensors file {:?}, err: {:?}",
                path,
                std::io::Error::last_os_error()
            )));
        }
        unsafe {
            let _ = libc::madvise(ptr, len, libc::MADV_RANDOM);
        }
        Ok(Self { ptr, len })
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr, self.len);
        }
    }
}
