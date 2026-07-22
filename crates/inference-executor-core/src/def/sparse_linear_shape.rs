use crate::def::DenseLinearShape;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseLinearShape {
    pub num_experts: usize,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl SparseLinearShape {
    pub fn validate(&self) {
        assert!(self.num_experts > 0);
        assert!(self.out_dim > 0);
        assert!(self.in_dim > 0);
    }

    pub fn flat_dense_shape(&self) -> DenseLinearShape {
        DenseLinearShape {
            out_dim: self.num_experts * self.out_dim,
            in_dim: self.in_dim,
        }
    }
}
