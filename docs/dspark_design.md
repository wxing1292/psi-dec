# DSpark Components

Status: retained low-level Qwen3.5-era component contract; not wired into the current Qwen3.5 executor or service.
Detailed DSpark integration is deferred to a separate Qwen3 milestone using the official DeepSeek contract and
compatible weights.

## Current scope

The repository keeps the already-implemented DSpark configuration, exact tensor binding, checkpoint conversion, Metal
components, component-level plans, weights, and focused tests. This preserves useful low-level work without committing
the Qwen3.5 runtime to unsupported behavior.

The current Qwen3.5 path deliberately has none of the following:

- a DSpark field or replay in `Qwen35Executor`;
- DSpark load, target-capture, or proposal methods;
- a Main residual-duplication hook or target-context input;
- `--hf-dspark-model-dir` service selection;
- shared Main/MTP/DSpark speculation abstraction;
- a DSpark end-to-end correctness or performance claim.

MTP is the only optional speculator wired into Qwen3.5.

## Retained source

```text
crates/inference-executor-core/src/model/qwen/v3_5/
  dspark_config.rs          upstream configuration normalization and target checks
  dspark_weight_layout.rs   exact source/runtime tensor manifests

crates/inference-executor-core/src/bin/
  qwen35_dspark_quantize.rs retained BF16 -> affine checkpoint converter

crates/inference-executor-metal/src/model/qwen/v3_5/
  plan.rs                   retained Qwen35DSparkPlan and direct geometry conversion
  dspark/
    attention.rs            DSpark attention composition
    block_request.rs        request-local block metadata
    context.rs              target-context append path
    layer.rs                DSpark layer owner
    markov.rs               Markov proposal head
    speculator.rs           low-level DSpark speculator composition
    target.rs               selected target-residual handling
    weights.rs              DSpark-owned checkpoint reads and conversion
```

These files are not referenced by the Qwen3.5 executor load or forward path. Their public low-level contracts remain
available to focused tests and future model-specific integration.

## Preserved low-level contract

The retained implementation continues to model:

- exact Hikari/DSpark tensor names and affine quantization layouts;
- selected target residual geometry;
- request-local block metadata and bounded context task coverage;
- GQA context append using the existing target cache lane layout;
- dense DSpark layer and scratch geometry;
- Markov proposal positions, sampling capacities, and output correction;
- component-owned weights and backend buffer contracts.

The standalone quantizer remains available:

```sh
cargo run -p inference-executor-core --bin qwen35_dspark_quantize -- \
  --input-dir /path/to/source \
  --output-dir /path/to/output \
  --group-size 64 --bits 4 --markov-w2-bits 8
```

The output directory must not already exist. Producing converted weights does not make them selectable by the current
Qwen3.5 server.

## Verification boundary

Current verification is compilation plus the existing low-level configuration, binding, geometry, quantization, and
component tests. No compatible DSpark model weights are available in the repository environment, so there is no DSpark
server/decode validation and no DSpark throughput result.

Future wiring must be reviewed as a new model-specific design. It must not infer compatibility from the retained
Qwen3.5 components, silently revive the old Main capture hooks, or broaden the current Qwen3.5 milestone.
