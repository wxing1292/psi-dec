use std::ffi::CString;
use std::num::NonZero;
use std::ops::Deref;
use std::ops::DerefMut;
use std::os::fd::AsFd;
use std::os::fd::OwnedFd;
use std::ptr::NonNull;
use std::slice;

use nix::fcntl::OFlag;
use nix::sys::mman::MapFlags;
use nix::sys::mman::ProtFlags;
use nix::sys::mman::mmap;
use nix::sys::mman::munmap;
use nix::sys::mman::shm_open;
use nix::sys::mman::shm_unlink;
use nix::sys::stat::Mode;
use nix::sys::stat::fstat;
use nix::unistd::ftruncate;

use crate::Error;
use crate::Result;

#[derive(Clone, Copy, Eq, PartialEq)]
enum HandleRole {
    Creator,
    Opener,
}

pub struct SharedMem {
    owned_fd: OwnedFd,
    ptr: NonNull<u8>,
    len: usize,
    c_path: CString,
    role: HandleRole,
}

impl SharedMem {
    pub fn create(path: &str, len: usize) -> Result<Self> {
        Self::create_or_open(
            path,
            len,
            OFlag::O_CREAT | OFlag::O_EXCL | OFlag::O_RDWR,
            Mode::S_IRUSR | Mode::S_IWUSR,
            HandleRole::Creator,
        )
    }

    pub fn open(path: &str, len: usize) -> Result<Self> {
        Self::create_or_open(path, len, OFlag::O_RDWR, Mode::empty(), HandleRole::Opener)
    }

    fn create_or_open(path: &str, len: usize, flag: OFlag, mode: Mode, role: HandleRole) -> Result<Self> {
        assert!(0 < len);
        let c_path = CString::new(path).map_err(|_| {
            let msg = format!("unable to convert path: {path} to cstring");
            tracing::error!(msg);
            Error::InternalError(msg)
        })?;
        let owned_fd = match shm_open(c_path.as_c_str(), flag, mode) {
            Ok(owned_fd) => owned_fd,
            Err(errno) => {
                let msg = format!(
                    "unable to create / open shared mem {path}, flag: {flag:?}, mode: {mode:?}, errno: {errno}"
                );
                tracing::error!(msg);
                return Err(Error::InternalError(msg));
            },
        };
        let cleanup_created_name = || {
            if role == HandleRole::Creator {
                let _ = shm_unlink(c_path.as_ref());
            }
        };
        if role == HandleRole::Creator {
            ftruncate(owned_fd.as_fd(), len as i64).map_err(|errno| {
                cleanup_created_name();
                let msg = format!("unable to ftruncate path: {path}, errno: {errno}");
                tracing::error!(msg);
                Error::InternalError(msg)
            })?;
        } else {
            let stat = fstat(owned_fd.as_fd()).map_err(|errno| {
                let msg = format!("unable to stat shared mem {path}, errno: {errno}");
                tracing::error!(msg);
                Error::InternalError(msg)
            })?;
            if stat.st_size < len as i64 {
                let msg = format!(
                    "shared mem {path} is smaller than requested mapping: actual={}, requested={len}",
                    stat.st_size
                );
                tracing::error!(msg);
                return Err(Error::InternalError(msg));
            }
        }
        let ptr = match unsafe {
            mmap(
                None,
                NonZero::new(len).unwrap(),
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                owned_fd.as_fd(),
                0,
            )
        } {
            Ok(ptr) => ptr,
            Err(errno) => {
                cleanup_created_name();
                let msg = format!("unable to map shared mem {path}, errno: {errno}");
                tracing::error!(msg);
                return Err(Error::InternalError(msg));
            },
        }
        .cast();
        Ok(Self {
            owned_fd,
            ptr,
            len,
            c_path,
            role,
        })
    }

    pub fn ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl AsRef<[u8]> for SharedMem {
    fn as_ref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl AsMut<[u8]> for SharedMem {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Deref for SharedMem {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl DerefMut for SharedMem {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl Drop for SharedMem {
    fn drop(&mut self) {
        let _ = unsafe { munmap(self.ptr.cast(), self.len) };
        if self.role == HandleRole::Creator {
            let _ = shm_unlink(self.c_path.as_ref());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsFd;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;

    use nix::sys::stat::fstat;

    use super::SharedMem;

    static NEXT_NAME: AtomicU64 = AtomicU64::new(0);

    fn unique_path() -> String {
        format!(
            "/pd{:x}{:x}",
            std::process::id(),
            NEXT_NAME.fetch_add(1, Ordering::Relaxed)
        )
    }

    #[test]
    fn test_create_open_close_destroy() {
        let path = unique_path();
        let mut creator = SharedMem::create(&path, 4096).unwrap();
        creator[0] = 42;

        let opener = SharedMem::open(&path, 4096).unwrap();
        assert_eq!(42, opener[0]);
        drop(opener);

        let second_opener = SharedMem::open(&path, 4096).unwrap();
        assert_eq!(42, second_opener[0]);
        drop(second_opener);

        drop(creator);

        assert!(SharedMem::open(&path, 4096).is_err());
    }

    #[test]
    fn test_create_and_open_errors() {
        let missing_path = unique_path();
        assert!(SharedMem::open(&missing_path, 4096).is_err());

        let path = unique_path();
        let creator = SharedMem::create(&path, 4096).unwrap();
        assert!(SharedMem::create(&path, 4096).is_err());

        let backing_len = fstat(creator.owned_fd.as_fd()).unwrap().st_size as usize;
        assert!(SharedMem::open(&path, backing_len + 1).is_err());
    }
}
