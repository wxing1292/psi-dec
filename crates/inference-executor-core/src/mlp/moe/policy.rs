#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MoEExecutionPolicy {
    #[default]
    Auto,
    TokenMajor,
    ExpertMajor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoEExecutionPolicyConfig {
    pub policy: MoEExecutionPolicy,
    pub auto_token_major_max_tokens: u32,
}

impl Default for MoEExecutionPolicyConfig {
    fn default() -> Self {
        Self {
            policy: MoEExecutionPolicy::Auto,
            auto_token_major_max_tokens: 4,
        }
    }
}

impl MoEExecutionPolicyConfig {
    pub fn new(policy: MoEExecutionPolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    pub fn auto_with_token_major_max_tokens(auto_token_major_max_tokens: u32) -> Self {
        Self {
            policy: MoEExecutionPolicy::Auto,
            auto_token_major_max_tokens,
        }
    }

    pub fn resolve(self, num_tokens: u32) -> MoEExecutionPolicy {
        assert!(num_tokens > 0);
        match self.policy {
            MoEExecutionPolicy::Auto => {
                if num_tokens <= self.auto_token_major_max_tokens {
                    MoEExecutionPolicy::TokenMajor
                } else {
                    MoEExecutionPolicy::ExpertMajor
                }
            },
            MoEExecutionPolicy::TokenMajor => MoEExecutionPolicy::TokenMajor,
            MoEExecutionPolicy::ExpertMajor => MoEExecutionPolicy::ExpertMajor,
        }
    }
}
