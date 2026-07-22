use std::cmp::max;
use std::time::Instant;

use comfy_table::Cell;
use comfy_table::Table;
use comfy_table::presets::UTF8_FULL;
use hdrhistogram::Histogram;

use crate::compute::BatchDevReq;
use crate::compute::BatchDevResp;
use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::RawComputeSlotSeq;
use crate::runtime::scheduler::ScheduleDecision;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::scheduler::UserRequest;

pub struct InstrumentedScheduler<Sch> {
    hist_enqueue: Histogram<u64>,
    hist_decision: Histogram<u64>,
    hist_prepare: Histogram<u64>,
    hist_cancel: Histogram<u64>,
    hist_commit: Histogram<u64>,

    scheduler: Sch,
}

impl<Sch> InstrumentedScheduler<Sch> {
    pub fn new(scheduler: Sch) -> Self {
        Self {
            hist_enqueue: Histogram::<u64>::new(4).unwrap(),
            hist_decision: Histogram::<u64>::new(4).unwrap(),
            hist_prepare: Histogram::<u64>::new(4).unwrap(),
            hist_cancel: Histogram::<u64>::new(4).unwrap(),
            hist_commit: Histogram::<u64>::new(4).unwrap(),

            scheduler,
        }
    }

    pub fn latency_table(&self) -> Table {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);

        let col_count = 2 + COLUMNS.len();

        // header
        let mut header = vec![Cell::new("scheduler api"), Cell::new("count")];
        header.extend(COLUMNS.iter().map(|(name, _)| Cell::new(*name)));
        table.set_header(header);

        // rows
        for (name, histogram) in [
            ("enqueue", &self.hist_enqueue),
            ("decision", &self.hist_decision),
            ("prepare", &self.hist_prepare),
            ("cancel", &self.hist_cancel),
            ("commit", &self.hist_commit),
        ] {
            let mut row = vec![Cell::new(name), Cell::new(histogram.len().to_string())];
            row.extend(COLUMNS.iter().map(|(_, col)| Cell::new(cell(histogram, *col))));
            table.add_row(row);
        }

        table
    }
}

impl<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, Sch>
    Scheduler<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp> for InstrumentedScheduler<Sch>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
    BatchDeviceReq: BatchDevReq<DeviceReq>,
    BatchDeviceResp: BatchDevResp<DeviceResp>,
    Sch: Scheduler<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp>,
{
    fn enqueue(&mut self, user_req: UserReq) {
        let instant = Instant::now();
        self.scheduler.enqueue(user_req);
        let latency = instant.elapsed().as_micros() as u64;
        let _ = self.hist_enqueue.record(max(1, latency));
    }

    fn decision(&mut self) -> ScheduleDecision {
        let instant = Instant::now();
        let result = self.scheduler.decision();
        let latency = instant.elapsed().as_micros() as u64;
        let _ = self.hist_decision.record(max(1, latency));
        result
    }

    fn prepare(&mut self) -> BatchDeviceReq {
        let instant = Instant::now();
        let result = self.scheduler.prepare();
        let latency = instant.elapsed().as_micros() as u64;
        let _ = self.hist_prepare.record(max(1, latency));
        result
    }

    fn cancel(&mut self, batch_dev_req: BatchDeviceReq) {
        let instant = Instant::now();
        self.scheduler.cancel(batch_dev_req);
        let latency = instant.elapsed().as_micros() as u64;
        let _ = self.hist_cancel.record(max(1, latency));
    }

    fn commit(&mut self, batch_dev_resp: BatchDeviceResp) {
        let instant = Instant::now();
        self.scheduler.commit(batch_dev_resp);
        let latency = instant.elapsed().as_micros() as u64;
        let _ = self.hist_commit.record(max(1, latency));
    }

    delegate::delegate! {
        to self.scheduler {
            fn last_compute_slot_seq(&self) -> RawComputeSlotSeq;
            fn next_compute_slot_seq(&self) -> Option<RawComputeSlotSeq>;
            fn run_queue_size(&self) -> usize;
            fn new_queue_size(&self) -> usize;
            fn swap_in_queue_size(&self) -> usize;
            fn swap_out_queue_size(&self) -> usize;
            fn queue_size(&self) -> usize;
        }
    }
}

#[derive(Clone, Copy)]
enum Column {
    P0,
    P(f64),
    P100,
    Avg,
}

const COLUMNS: &[(&str, Column)] = &[
    ("p0", Column::P0),
    ("p10", Column::P(0.10)),
    ("p20", Column::P(0.20)),
    ("p30", Column::P(0.30)),
    ("p40", Column::P(0.40)),
    ("p50", Column::P(0.50)),
    ("p60", Column::P(0.60)),
    ("p70", Column::P(0.70)),
    ("p80", Column::P(0.80)),
    ("p90", Column::P(0.90)),
    ("p95", Column::P(0.95)),
    ("p99", Column::P(0.99)),
    ("p999", Column::P(0.999)),
    ("p100", Column::P100),
    ("avg", Column::Avg),
];

fn cell(hist: &Histogram<u64>, kind: Column) -> String {
    if hist.is_empty() {
        return "-".into();
    }

    fn fmt_us(us: u64) -> String {
        if us < 1_000 {
            format!("{us}us")
        } else if us < 1_000_000 {
            format!("{:.3}ms", us as f64 / 1_000.0)
        } else {
            format!("{:.3}s", us as f64 / 1_000_000.0)
        }
    }
    match kind {
        Column::Avg => fmt_us(hist.mean().round() as u64),
        Column::P0 => fmt_us(hist.min()),
        Column::P100 => fmt_us(hist.max()),
        Column::P(q) => fmt_us(hist.value_at_quantile(q)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::MockBatchDevReq;
    use crate::compute::MockBatchDevResp;
    use crate::compute::MockDevReq;
    use crate::compute::MockDevResp;
    use crate::runtime::scheduler::MockScheduler;
    use crate::runtime::scheduler::MockUserRequest;

    #[test]
    fn test_print_latency_table() {
        let mut scheduler = InstrumentedScheduler::new(MockScheduler::<
            MockUserRequest<MockDevReq, MockDevResp>,
            MockDevReq,
            MockDevResp,
            MockBatchDevReq<MockDevReq>,
            MockBatchDevResp<MockDevResp>,
        >::new());

        for us in [5, 6, 7, 8, 10, 12, 15, 20, 50] {
            scheduler.hist_enqueue.record(us).unwrap();
        }

        for us in [1, 1, 2, 2, 3, 3, 4, 7, 12] {
            scheduler.hist_decision.record(us).unwrap();
        }

        for us in [80, 120, 150, 180, 200, 220, 300, 1200] {
            scheduler.hist_prepare.record(us).unwrap();
        }

        for us in [30, 50, 60, 70, 90, 120, 220, 600] {
            scheduler.hist_commit.record(us).unwrap();
        }

        let table = scheduler.latency_table();
        println!("{}", table);
    }
}
