use std::ops::Range;

// This scan is intentionally not linearizable. An ID allocated after its bitmap word is read has no
// executor-visible payload before Stop. Executor publication is ordered behind Stop on the event-loop FIFO. An ID
// freed after its bitmap word is read can remain in the plan because saving unused state is safe.
//
// TODO(per-request-state-I/O): Before per-request I/O can publish state through an independent executor stream,
// EventLoop must track those tasks and defer whole-executor Stop until their completion is ordered before this scan.
pub fn collect_allocated_id_ranges(words: impl Iterator<Item = u64>) -> Vec<Range<u32>> {
    let mut ranges: Vec<Range<u32>> = Vec::new();

    for (index, mut bitmap) in words.enumerate() {
        let mut bit_base = u32::try_from(index << 6).expect("allocated-ID bitmap index must fit u32");

        while bitmap != 0 {
            let unset_count = bitmap.trailing_zeros();
            bitmap >>= unset_count;
            bit_base += unset_count;

            let set_count = bitmap.trailing_ones();
            let end = bit_base
                .checked_add(set_count)
                .expect("allocated-ID range end must fit u32");
            if let Some(last) = ranges.last_mut()
                && last.end == bit_base
            {
                last.end = end;
            } else {
                ranges.push(bit_base..end);
            }

            if set_count == u64::BITS {
                bitmap = 0;
            } else {
                bitmap >>= set_count;
            }
            bit_base = end;
        }
    }

    ranges
}

#[cfg(test)]
mod tests {
    use super::collect_allocated_id_ranges;

    #[test]
    fn test_collect_allocated_id_ranges() {
        let words = [
            0x0000_0000_0000_000b,
            0x0000_0000_0000_0013,
            0x0000_0000_0000_0000,
            0xffff_ffff_ffff_ffff,
        ];

        assert_eq!(
            collect_allocated_id_ranges(words.into_iter()),
            vec![0..2, 3..4, 64..66, 68..69, 192..256]
        );
    }
}
