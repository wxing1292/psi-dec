use futures_lite::future::Boxed;

use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::scheduler::UserRequest;
use crate::runtime::tasks::AsyncTask;
use crate::runtime::tasks::AwaitReservation;

pub enum SwapOutTask<UserReq, DeviceReq, DeviceResp> {
    AwaitReservation(AwaitReservation<UserReq, DeviceReq, DeviceResp>),
}

impl<UserReq, DeviceReq, DeviceResp> SwapOutTask<UserReq, DeviceReq, DeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    pub fn await_reservation(user_req: UserReq, wait: Boxed<()>) -> Self {
        Self::AwaitReservation(AwaitReservation::new(user_req, wait))
    }
}

impl<UserReq, DeviceReq, DeviceResp> AsyncTask for SwapOutTask<UserReq, DeviceReq, DeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    type Output = UserReq;

    async fn run(self) -> Self::Output {
        match self {
            Self::AwaitReservation(task) => task.run().await,
        }
    }
}
