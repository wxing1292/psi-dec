use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::Path;

use inference_executor_core::model::qwen::v3_x::dspark::init_qwen3x_dspark_config;
use inference_executor_core::model::qwen::v3_x::dspark::resolve_qwen3x_dspark_source_weight_bindings;

use crate::checkpoint::ConversionPlan;
use crate::checkpoint::Result;
use crate::checkpoint::convert_checkpoint;
use crate::checkpoint::error;

pub fn convert(
    input_dir: &Path,
    output_dir: &Path,
    group_size: usize,
    bits: usize,
    markov_w2_bits: usize,
) -> Result<()> {
    convert_checkpoint(input_dir, output_dir, |input_dir, header, mut config_value| {
        let config_path = input_dir.join("config.json");
        let config = init_qwen3x_dspark_config(input_dir)
            .map_err(|err| error(format!("invalid Qwen3x DSpark config {config_path:?}: {err}")))?;
        let bindings = resolve_qwen3x_dspark_source_weight_bindings(&config, header.tensors.keys().map(String::as_str))
            .map_err(|err| error(format!("Qwen3x DSpark tensor set mismatch: {err}")))?;

        let mut bit_overrides = BTreeMap::new();
        let mut quantization = serde_json::Map::from_iter([
            ("group_size".to_string(), serde_json::Value::from(group_size)),
            ("bits".to_string(), serde_json::Value::from(bits)),
            ("mode".to_string(), serde_json::Value::from("affine")),
        ]);
        if markov_w2_bits != bits {
            bit_overrides.insert("markov_head.markov_w2.weight".to_string(), markov_w2_bits);
            quantization.insert(
                "markov_head.markov_w2".to_string(),
                serde_json::json!({ "bits": markov_w2_bits }),
            );
        }
        config_value
            .as_object_mut()
            .ok_or_else(|| error("DSpark config root must be a JSON object"))?
            .insert("quantization".to_string(), serde_json::Value::Object(quantization));

        Ok(ConversionPlan {
            model_name: "DSpark",
            format: "psi-dec-dspark-affine",
            source_max_rank: 2,
            group_size,
            bits,
            bit_overrides,
            unquantized_matrices: BTreeSet::from(["confidence_head.proj.weight".to_string()]),
            renamed_tensors: BTreeMap::new(),
            expected_output_names: bindings.tensor_names().into_iter().map(str::to_string).collect(),
            metadata: HashMap::from([("markov_w2_bits".to_string(), markov_w2_bits.to_string())]),
            config: config_value,
        })
    })
}
