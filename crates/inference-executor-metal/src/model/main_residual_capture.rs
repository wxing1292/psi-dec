use inference_backend_metal::components::residual_add;
use inference_backend_metal::metal::Buffer;

#[derive(Clone, Copy)]
pub enum MainResidualRows<'a> {
    Prefix,
    Indices(&'a Buffer),
}

impl MainResidualRows<'_> {
    pub fn gathers(self) -> bool {
        matches!(self, Self::Indices(_))
    }
}

/// Selects stable capture destinations for Main layer residual outputs.
///
/// Capture selection and destinations are fixed replay topology. The owner
/// must keep returned buffers and column ranges stable for the lifetime of
/// Main. Capture destinations must not alias Main workspaces.
pub trait MainResidualCapture {
    fn capture_for_model_layer(&self, model_layer_index: usize) -> Option<residual_add::CaptureTarget<'_>>;
}
