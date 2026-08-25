use std::any::Any;

use futures_lite::future::Boxed;

use crate::runtime::RawRequestID;

pub trait AsyncTaskReq: Send + 'static {
    type Resp: AsyncTaskResp + ?Sized;

    fn request_id(&self) -> RawRequestID;
    fn run(self: Box<Self>) -> Boxed<Box<Self::Resp>>;
}

pub trait AsyncTaskResp: Any + Send + 'static {
    fn request_id(&self) -> RawRequestID;
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send>;
}
