use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use futures_lite::future::Boxed;

use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::scheduler::UserRequest;

pub struct AwaitReservation<UserReq, DeviceReq, DeviceResp> {
    future: Boxed<UserReq>,
    phantom_data: PhantomData<fn() -> (DeviceReq, DeviceResp)>,
}

impl<UserReq, DeviceReq, DeviceResp> AwaitReservation<UserReq, DeviceReq, DeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    pub fn new(user_req: UserReq, wait: Boxed<()>) -> Self {
        let future = Box::pin(async move {
            if user_req.is_terminal() {
                return user_req;
            }
            wait.await;
            user_req
        });
        Self {
            future,
            phantom_data: PhantomData,
        }
    }
}

impl<UserReq, DeviceReq, DeviceResp> Future for AwaitReservation<UserReq, DeviceReq, DeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    type Output = UserReq;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.future.as_mut().poll(cx)
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
        user_req.expect_is_terminal().once().return_const(false);

        let _user_req = TestAwaitReservation::new(user_req, Box::pin(async {})).await;
    }

    #[tokio::test]
    async fn test_terminal_before_wait() {
        let waited = Arc::new(AtomicBool::new(false));
        let waited_by_task = waited.clone();
        let mut user_req = TestUserReq::new();
        user_req.expect_is_terminal().once().return_const(true);

        let _user_req = TestAwaitReservation::new(
            user_req,
            Box::pin(async move {
                waited_by_task.store(true, Ordering::Release);
            }),
        )
        .await;

        assert!(!waited.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn test_wait() {
        let waited = Arc::new(AtomicBool::new(false));
        let waited_by_task = waited.clone();
        let mut user_req = TestUserReq::new();
        user_req.expect_is_terminal().once().return_const(false);

        let _user_req = TestAwaitReservation::new(
            user_req,
            Box::pin(async move {
                waited_by_task.store(true, Ordering::Release);
            }),
        )
        .await;

        assert!(waited.load(Ordering::Acquire));
    }
}
