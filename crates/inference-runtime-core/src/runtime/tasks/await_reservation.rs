use std::marker::PhantomData;

use futures_lite::future::Boxed;

use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::scheduler::UserRequest;

pub struct AwaitReservation<UserReq, DeviceReq, DeviceResp> {
    user_req: UserReq,
    wait: Boxed<()>,
    phantom_data: PhantomData<fn() -> (DeviceReq, DeviceResp)>,
}

impl<UserReq, DeviceReq, DeviceResp> AwaitReservation<UserReq, DeviceReq, DeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    pub fn new(user_req: UserReq, wait: Boxed<()>) -> Self {
        Self {
            user_req,
            wait,
            phantom_data: PhantomData,
        }
    }

    pub async fn run(self) -> UserReq {
        let Self { user_req, wait, .. } = self;
        if !user_req.store_swapped() {
            assert!(
                user_req.is_terminal(),
                "swap-out task requires a running or terminal request"
            );
            return user_req;
        }
        wait.await;
        if !user_req.store_running() {
            assert!(
                user_req.is_terminal(),
                "completed swap-out task requires a swapped or terminal request"
            );
        }
        user_req
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::compute::MockDevReq;
    use crate::compute::MockDevResp;
    use crate::runtime::scheduler::MockUserRequest;

    type TestUserReq = MockUserRequest<MockDevReq, MockDevResp>;
    type TestAwaitReservation = AwaitReservation<TestUserReq, MockDevReq, MockDevResp>;

    #[tokio::test]
    async fn test_complete() {
        let mut user_req = TestUserReq::new();
        user_req.expect_store_swapped().once().return_const(true);
        user_req.expect_store_running().once().return_const(true);

        let _user_req = TestAwaitReservation::new(user_req, Box::pin(async {})).run().await;
    }

    #[tokio::test]
    async fn test_terminal_before_wait() {
        let waited = Arc::new(AtomicBool::new(false));
        let waited_by_task = waited.clone();
        let mut user_req = TestUserReq::new();
        user_req.expect_store_swapped().once().return_const(false);
        user_req.expect_is_terminal().once().return_const(true);

        let _user_req = TestAwaitReservation::new(
            user_req,
            Box::pin(async move {
                waited_by_task.store(true, Ordering::Release);
            }),
        )
        .run()
        .await;

        assert!(!waited.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn test_terminal_after_wait() {
        let waited = Arc::new(AtomicBool::new(false));
        let waited_by_task = waited.clone();
        let mut user_req = TestUserReq::new();
        user_req.expect_store_swapped().once().return_const(true);
        user_req.expect_store_running().once().return_const(false);
        user_req.expect_is_terminal().once().return_const(true);

        let _user_req = TestAwaitReservation::new(
            user_req,
            Box::pin(async move {
                waited_by_task.store(true, Ordering::Release);
            }),
        )
        .run()
        .await;

        assert!(waited.load(Ordering::Acquire));
    }
}
