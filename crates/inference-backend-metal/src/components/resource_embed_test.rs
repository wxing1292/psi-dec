use half::bf16;

use super::Buffers;
use super::Compute;
use super::Config;
use super::Mapping;
use super::MappingTable;
use crate::metal::Buffer;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;
use crate::metal::Stream;
use crate::test_support::ReplayTestCache;

const NUM_ACTIVE_MAPPINGS: ReplayParameterKey = ReplayParameterKey::new("test.resource_embed.num_active_mappings");

#[test]
fn test_replay_matches_reference_across_active_counts() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        hidden_dim: 4,
        io_dtype: Dtype::Bfloat16,
    };
    let arena_values = (0..20)
        .map(|value| bf16::from_f32(value as f32 + 100.0).to_bits())
        .collect::<Vec<_>>();
    let resource_arena = Buffer::from_slice(&device, &arena_values);
    let table = MappingTable::new(
        config,
        vec![
            Mapping {
                destination_row: 4,
                source_offset_bytes: 2 * config.hidden_dim_bytes(),
            },
            Mapping {
                destination_row: 1,
                source_offset_bytes: 3 * config.hidden_dim_bytes(),
            },
            Mapping {
                destination_row: 3,
                source_offset_bytes: 0,
            },
        ],
    );
    let mappings = Buffer::from_slice(&device, table.encoded_u32s());
    let initial_hidden = (0..20)
        .map(|value| bf16::from_f32(value as f32).to_bits())
        .collect::<Vec<_>>();
    let hidden = Buffer::from_slice(&device, &initial_hidden);
    let compute = Compute::new(&device, config);
    let mut cache = ReplayTestCache::new();
    let (_, cache_hit) = cache.record(table.shape(), || {
        let mut recorder = stream.create_replay_program();
        recorder.record(compute.invoke(
            &table,
            ReplayU32::Parameter(NUM_ACTIVE_MAPPINGS),
            Buffers {
                resource_arena: &resource_arena,
                mappings: &mappings,
                hidden: &hidden,
            },
        ));
        recorder.build()
    });
    assert!(!cache_hit);

    for num_active_mappings in [1, 3, 2] {
        hidden.write_typed(0, &initial_hidden);
        let (replay, cache_hit) = cache.record(table.shape(), || unreachable!());
        assert!(cache_hit);
        stream
            .submit_replay_with_arguments(
                replay,
                &ReplayArguments::new().with_u32(NUM_ACTIVE_MAPPINGS, num_active_mappings),
            )
            .wait();

        let mut expected = initial_hidden.clone();
        let reference_mappings = [(1usize, 3usize), (3, 0), (4, 2)];
        for &(destination_row, source_row) in &reference_mappings[..num_active_mappings as usize] {
            expected[destination_row * 4..destination_row * 4 + 4]
                .copy_from_slice(&arena_values[source_row * 4..source_row * 4 + 4]);
        }
        assert_eq!(expected, hidden.read_typed::<u16>(0, expected.len()));
    }

    let smaller_table = MappingTable::new(
        config,
        vec![
            Mapping {
                destination_row: 1,
                source_offset_bytes: 3 * config.hidden_dim_bytes(),
            },
            Mapping {
                destination_row: 3,
                source_offset_bytes: 0,
            },
        ],
    );
    let smaller_mappings = Buffer::from_slice(&device, smaller_table.encoded_u32s());
    let (_, cache_hit) = cache.record(smaller_table.shape(), || {
        let mut recorder = stream.create_replay_program();
        recorder.record(compute.invoke(
            &smaller_table,
            ReplayU32::Parameter(NUM_ACTIVE_MAPPINGS),
            Buffers {
                resource_arena: &resource_arena,
                mappings: &smaller_mappings,
                hidden: &hidden,
            },
        ));
        recorder.build()
    });
    assert!(!cache_hit);
}

#[test]
#[should_panic(expected = "destination rows must be unique")]
fn test_mapping_table_rejects_duplicate_destination_rows() {
    let config = Config {
        hidden_dim: 4,
        io_dtype: Dtype::Bfloat16,
    };
    MappingTable::new(
        config,
        vec![
            Mapping {
                destination_row: 1,
                source_offset_bytes: 0,
            },
            Mapping {
                destination_row: 1,
                source_offset_bytes: 8,
            },
        ],
    );
}
