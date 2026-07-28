use crate::checkpoint::QuantizedTensorBindings;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xGDNWeightBindings {
    pub qkv: QuantizedTensorBindings,
    pub a: QuantizedTensorBindings,
    pub b: QuantizedTensorBindings,
    pub z: QuantizedTensorBindings,
    pub conv_weight: String,
    pub norm_weight: String,
    pub a_log: String,
    pub dt_bias: String,
    pub output: QuantizedTensorBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xGQAWeightBindings {
    pub q: QuantizedTensorBindings,
    pub k: QuantizedTensorBindings,
    pub v: QuantizedTensorBindings,
    pub q_norm_weight: String,
    pub k_norm_weight: String,
    pub output: QuantizedTensorBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xDenseMLPWeightBindings {
    pub gate: QuantizedTensorBindings,
    pub up: QuantizedTensorBindings,
    pub down: QuantizedTensorBindings,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xMoEWeightBindings {
    pub router: QuantizedTensorBindings,
    pub experts: Qwen3xSparseExpertWeightBindings,
    pub shared_expert_gate: Option<QuantizedTensorBindings>,
    pub shared_expert: Option<Qwen3xDenseMLPWeightBindings>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3xSparseExpertWeightBindings {
    pub gate: QuantizedTensorBindings,
    pub up: QuantizedTensorBindings,
    pub down: QuantizedTensorBindings,
}

impl Qwen3xGDNWeightBindings {
    pub fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        push_quantized_tensor_names(&self.qkv, names);
        push_quantized_tensor_names(&self.a, names);
        push_quantized_tensor_names(&self.b, names);
        push_quantized_tensor_names(&self.z, names);
        names.extend([
            self.conv_weight.as_str(),
            self.norm_weight.as_str(),
            self.a_log.as_str(),
            self.dt_bias.as_str(),
        ]);
        push_quantized_tensor_names(&self.output, names);
    }
}

impl Qwen3xGQAWeightBindings {
    pub fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        push_quantized_tensor_names(&self.q, names);
        push_quantized_tensor_names(&self.k, names);
        push_quantized_tensor_names(&self.v, names);
        names.extend([self.q_norm_weight.as_str(), self.k_norm_weight.as_str()]);
        push_quantized_tensor_names(&self.output, names);
    }
}

impl Qwen3xDenseMLPWeightBindings {
    pub fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        push_quantized_tensor_names(&self.gate, names);
        push_quantized_tensor_names(&self.up, names);
        push_quantized_tensor_names(&self.down, names);
    }
}

impl Qwen3xMoEWeightBindings {
    pub fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        push_quantized_tensor_names(&self.router, names);
        self.experts.push_tensor_names(names);
        if let Some(bindings) = &self.shared_expert_gate {
            push_quantized_tensor_names(bindings, names);
        }
        if let Some(bindings) = &self.shared_expert {
            bindings.push_tensor_names(names);
        }
    }
}

impl Qwen3xSparseExpertWeightBindings {
    pub fn push_tensor_names<'a>(&'a self, names: &mut Vec<&'a str>) {
        push_quantized_tensor_names(&self.gate, names);
        push_quantized_tensor_names(&self.up, names);
        push_quantized_tensor_names(&self.down, names);
    }
}

pub fn quantized(prefix: &str, relative_name: &str) -> QuantizedTensorBindings {
    quantized_path(format!("{prefix}.{relative_name}"))
}

pub fn quantized_path(prefix: String) -> QuantizedTensorBindings {
    QuantizedTensorBindings {
        weight: format!("{prefix}.weight"),
        scales: format!("{prefix}.scales"),
        biases: format!("{prefix}.biases"),
    }
}

pub fn push_quantized_tensor_names<'a>(bindings: &'a QuantizedTensorBindings, names: &mut Vec<&'a str>) {
    names.extend([
        bindings.weight.as_str(),
        bindings.scales.as_str(),
        bindings.biases.as_str(),
    ]);
}

pub fn dense_mlp_bindings(prefix: &str) -> Qwen3xDenseMLPWeightBindings {
    Qwen3xDenseMLPWeightBindings {
        gate: quantized(prefix, "gate_proj"),
        up: quantized(prefix, "up_proj"),
        down: quantized(prefix, "down_proj"),
    }
}
