#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum SamplingDomain {
    Target = 0x243f_6a88,
    Draft = 0x85a3_08d3,

    /// Bernoulli draw deciding whether to accept a draft token.
    Accept = 0x1319_8a2e,

    /// Replacement draw from normalized max(target - draft, 0).
    Resample = 0x0370_7344,
}

impl From<SamplingDomain> for u32 {
    fn from(domain: SamplingDomain) -> Self {
        domain as Self
    }
}
