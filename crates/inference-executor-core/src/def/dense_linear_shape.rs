#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseLinearShape {
    pub out_dim: usize,
    pub in_dim: usize,
}

impl DenseLinearShape {
    pub fn validate(&self) {
        assert!(self.out_dim > 0);
        assert!(self.in_dim > 0);
    }

    pub fn weight_shape(&self) -> [usize; 2] {
        [self.out_dim, self.in_dim]
    }
}
