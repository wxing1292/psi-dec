# DSpark Components

Current status: The repository retains the low-level Qwen3.5-era component contract.
The current Qwen3.5 executor and service do not use it.
Future status: A separate Qwen3 milestone owns detailed DSpark integration.
That milestone must use the official DeepSeek contract and compatible weights.

## Current scope

The repository retains the implemented DSpark configuration, exact tensor bindings, checkpoint conversion, Metal
components, plans, weights, and focused tests.
This scope preserves useful low-level work.
It does not commit the Qwen3.5 runtime to unsupported behavior.

The current Qwen3.5 path deliberately has none of the following:

- A DSpark field or replay in `Qwen35Executor`.
- DSpark load, target-capture, or proposal methods.
- A DSpark capture policy or target-context input.
- `--hf-dspark-model-dir` service selection.
- A shared Main/MTP/DSpark speculation abstraction.
- A DSpark end-to-end correctness or performance claim.

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

The Qwen3.5 executor load and forward paths do not reference these files.
Their public low-level contracts remain available to focused tests and future model-specific integration.

`Qwen35Main` retains the narrow, object-safe Qwen3.5-era residual-capture seam.
It can ask an optional `Rc<dyn Qwen35MainResidualCapture>` for a capture target.
It makes this request at each model layer's final post-MLP residual add.
The capture contract returns only an opaque `ResidualCaptureTarget`.
It has no recorder method.

The Qwen3.5 loader passes `None` for the capture owner.
Thus, the seam adds no Qwen3.5 operator or output.

The separate Qwen3 milestone uses `Qwen3MainResidualCapture`.
That milestone must not bind a semantic component to `ReplayRecorder`.

## Preserved low-level contract

The retained implementation models:

- Exact Hikari/DSpark tensor names and affine quantization layouts.
- Selected target residual geometry.
- Request-local block metadata and bounded context task coverage.
- GQA context append with the existing target cache lane layout.
- Dense DSpark layer and scratch geometry.
- Markov proposal positions, sampling capacities, and output correction.
- Component-owned weights and backend buffer contracts.

The standalone quantizer remains available:

```sh
cargo run -p inference-executor-core --bin qwen35_dspark_quantize -- \
  --input-dir /path/to/source \
  --output-dir /path/to/output \
  --group-size 64 --bits 4 --markov-w2-bits 8
```

The output directory must not already exist.
Converted weights are not selectable by the current Qwen3.5 server.

## Verification boundary

Current verification includes compilation and existing low-level tests.
These tests cover configuration, bindings, geometry, quantization, and components.
The repository environment has no compatible DSpark model weights.
Thus, there is no DSpark server/decode validation or DSpark throughput result.

Review future wiring as a new model-specific design.
It must not infer compatibility from the retained Qwen3.5 components.
It must not add recorder semantics to the Main capture contract.
It must not broaden the current Qwen3.5 milestone.
