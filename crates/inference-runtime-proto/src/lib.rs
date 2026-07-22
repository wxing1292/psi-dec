pub mod inference_runtime_service {
    tonic::include_proto!("psi_dec.inference.v1");
}

pub const INFERENCE_RUNTIME_FILE_DESCRIPTOR_SET: &[u8] =
    tonic::include_file_descriptor_set!("inference_runtime_descriptor");
