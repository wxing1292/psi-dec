use inference_backend_metal::components::residual_add;

/// Selects stable capture destinations for Main layer residual outputs.
///
/// Capture selection and destinations are fixed replay topology. The owner
/// must keep returned buffers and column ranges stable for the lifetime of
/// Main. Capture destinations must not alias Main workspaces.
pub trait MainResidualCapture {
    fn capture_for_model_layer(&self, model_layer_index: usize) -> Option<residual_add::CaptureTarget<'_>>;
}
