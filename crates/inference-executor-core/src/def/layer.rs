/// Semantic abstraction for model layers/components.
///
/// This trait is intentionally lightweight: each concrete layer owns its core,
/// backend, weights, and any internal resource/cache manager. Request-specific
/// routing keys or external caches should be part of the typed input rather than
/// hidden behind a fake `Array -> Array` interface.
pub trait Layer {
    type Input<'a>
    where
        Self: 'a;
    type Output<'a>
    where
        Self: 'a;

    type InputShape;
    type OutputShape;

    fn input_shape(&self) -> Self::InputShape;
    fn output_shape(&self) -> Self::OutputShape;
}
