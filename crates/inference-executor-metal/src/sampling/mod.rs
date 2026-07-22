use std::cell::Cell;

pub mod rejection_sampling;
pub mod spec_probs;
pub mod top_k_replay;
pub mod top_k_sampling;

#[derive(Default)]
struct RuntimeParamRows {
    configured: Cell<Option<u32>>,
}

impl RuntimeParamRows {
    fn set(&self, num_rows: usize, name: &str) {
        let num_rows = num_rows
            .try_into()
            .unwrap_or_else(|_| panic!("{name} runtime parameter row count must fit u32"));
        self.configured.set(Some(num_rows));
    }

    fn consume(&self, num_active_rows: u32, name: &str) {
        assert_eq!(
            self.configured.take(),
            Some(num_active_rows),
            "{name} runtime parameter rows must be freshly configured for the active replay rows"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeParamRows;

    #[test]
    #[should_panic(expected = "runtime parameter rows must be freshly configured")]
    fn test_runtime_param_rows_require_fresh_generation() {
        let rows = RuntimeParamRows::default();
        rows.set(1, "test sampling");
        rows.consume(1, "test sampling");
        rows.consume(1, "test sampling");
    }
}
