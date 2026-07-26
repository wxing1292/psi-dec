use std::sync::Arc;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionReason {
    StopSequence,
    LengthLimit,
    ContextLimit,
}

#[derive(Clone, Debug)]
pub struct AtomicRequestStatus {
    status: Arc<AtomicU8>,
}

impl AtomicRequestStatus {
    #[inline]
    pub fn new() -> Self {
        Self {
            status: Arc::new(AtomicU8::new(RequestStatus::Initialized.to_u8())),
        }
    }

    #[inline]
    pub fn store_running(&self) -> bool {
        loop {
            let status = RequestStatus::from_u8(self.status.load(Ordering::Acquire));
            match status {
                RequestStatus::Initialized | RequestStatus::Swapped => {
                    if self
                        .status
                        .compare_exchange_weak(
                            status.to_u8(),
                            RequestStatus::Running.to_u8(),
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return true;
                    }
                    continue;
                },
                _ => return false,
            }
        }
    }

    #[inline]
    pub fn store_swapped(&self) -> bool {
        loop {
            let status = RequestStatus::from_u8(self.status.load(Ordering::Acquire));
            match status {
                RequestStatus::Running => {
                    if self
                        .status
                        .compare_exchange_weak(
                            status.to_u8(),
                            RequestStatus::Swapped.to_u8(),
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return true;
                    }
                    continue;
                },
                _ => return false,
            }
        }
    }

    #[inline]
    pub fn store_cancelled(&self) -> bool {
        self.store_terminal(RequestStatus::Cancelled)
    }
    #[inline]
    pub fn store_timed_out(&self) -> bool {
        self.store_terminal(RequestStatus::TimedOut)
    }
    #[inline]
    pub fn store_aborted(&self) -> bool {
        self.store_terminal(RequestStatus::Aborted)
    }
    #[inline]
    pub fn store_completed(&self, completion: CompletionReason) -> bool {
        self.store_terminal(RequestStatus::Completed(completion))
    }

    #[inline]
    fn store_terminal(&self, to: RequestStatus) -> bool {
        assert!(
            to.is_terminal(),
            "terminal status transition requires a terminal target"
        );
        loop {
            let status_u8 = self.status.load(Ordering::Acquire);
            let status = RequestStatus::from_u8(status_u8);
            if status.is_terminal() {
                return false;
            }
            if self
                .status
                .compare_exchange_weak(status_u8, to.to_u8(), Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return true;
            }
        }
    }

    #[inline]
    pub fn load(&self) -> RequestStatus {
        RequestStatus::from_u8(self.status.load(Ordering::Acquire))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestStatus {
    Initialized,
    Running,
    Swapped,

    Cancelled,
    TimedOut,
    Aborted,
    Completed(CompletionReason),
}

impl RequestStatus {
    #[inline]
    pub const fn is_initialized(self) -> bool {
        matches!(self, Self::Initialized)
    }

    #[inline]
    pub const fn is_running(self) -> bool {
        matches!(self, Self::Running)
    }

    #[inline]
    pub const fn is_swapped(self) -> bool {
        matches!(self, Self::Swapped)
    }

    #[inline]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Cancelled | Self::TimedOut | Self::Aborted | Self::Completed(_)
        )
    }

    #[inline]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Initialized => 0,
            Self::Running => 1,
            Self::Swapped => 2,
            Self::Cancelled => 3,
            Self::TimedOut => 4,
            Self::Aborted => 5,
            Self::Completed(CompletionReason::StopSequence) => 6,
            Self::Completed(CompletionReason::LengthLimit) => 7,
            Self::Completed(CompletionReason::ContextLimit) => 8,
        }
    }

    #[inline]
    pub const fn from_u8(value: u8) -> RequestStatus {
        match value {
            0 => RequestStatus::Initialized,
            1 => RequestStatus::Running,
            2 => RequestStatus::Swapped,
            3 => RequestStatus::Cancelled,
            4 => RequestStatus::TimedOut,
            5 => RequestStatus::Aborted,
            6 => RequestStatus::Completed(CompletionReason::StopSequence),
            7 => RequestStatus::Completed(CompletionReason::LengthLimit),
            8 => RequestStatus::Completed(CompletionReason::ContextLimit),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_status_follows_initialized_running_terminal_lifecycle() {
        let status = AtomicRequestStatus::new();
        assert_eq!(status.load(), RequestStatus::Initialized);
        assert!(status.store_running());
        assert!(!status.store_running());
        assert_eq!(status.load(), RequestStatus::Running);
        assert!(status.store_completed(CompletionReason::StopSequence));
        assert!(!status.store_cancelled());
        assert_eq!(status.load(), RequestStatus::Completed(CompletionReason::StopSequence));
    }

    #[test]
    fn test_cancelled_request_cannot_be_started() {
        let status = AtomicRequestStatus::new();
        assert!(status.store_cancelled());
        assert!(!status.store_running());
        assert_eq!(status.load(), RequestStatus::Cancelled);
    }

    #[test]
    fn test_context_limit_is_terminal() {
        let status = AtomicRequestStatus::new();
        assert!(status.store_completed(CompletionReason::ContextLimit));
        assert_eq!(status.load(), RequestStatus::Completed(CompletionReason::ContextLimit));
    }

    #[test]
    fn test_swapped_request_can_return_to_running() {
        let status = AtomicRequestStatus::new();
        assert!(status.store_running());
        assert!(status.store_swapped());
        assert_eq!(status.load(), RequestStatus::Swapped);
        assert!(status.store_running());
        assert_eq!(status.load(), RequestStatus::Running);
    }
}
