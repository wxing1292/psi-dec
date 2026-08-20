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
use crate::runtime::scheduler::Scheduler;
use crate::runtime::scheduler::UserRequest;

pub struct InstrumentedScheduler<Sch> {
    num_enqueue: u64,
    num_swap_in: u64,
    hist_prepare: Histogram<u64>,
    hist_cancel: Histogram<u64>,
    hist_commit: Histogram<u64>,

    scheduler: Sch,
}

impl<Sch> InstrumentedScheduler<Sch> {
    pub fn new(scheduler: Sch) -> Self {
        Self {
            num_enqueue: 0,
            num_swap_in: 0,
            hist_prepare: Histogram::<u64>::new(4).unwrap(),
            hist_cancel: Histogram::<u64>::new(4).unwrap(),
            hist_commit: Histogram::<u64>::new(4).unwrap(),

            scheduler,
        }
    }

    pub fn stats_table(&self) -> Table {
        let mut table = Table::new();
        table.load_style(UTF8_FULL.with_rounded_corners());

        let mut header = vec![Cell::new("scheduler api"), Cell::new("count")];
        header.extend(COLUMNS.iter().map(|(name, _)| Cell::new(*name)));
        table.set_header(header);

        for (name, count) in [("enqueue", self.num_enqueue), ("swap_in", self.num_swap_in)] {
            let mut row = vec![Cell::new(name), Cell::new(count)];
            row.extend(COLUMNS.iter().map(|_| Cell::new("-")));
            table.add_row(row);
        }

        for (name, histogram) in [
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
        self.scheduler.enqueue(user_req);
        self.num_enqueue += 1;
    }

    fn swap_in(&mut self, user_req: UserReq) {
        self.scheduler.swap_in(user_req);
        self.num_swap_in += 1;
    }

    fn pop_ready_reqs(&mut self) -> Option<UserReq> {
        self.scheduler.pop_ready_reqs()
    }

    fn can_flush(&self) -> bool {
        self.scheduler.can_flush()
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
    fn test_metrics() {
        type TestUserReq = MockUserRequest<MockDevReq, MockDevResp>;
        type TestBatchDeviceReq = MockBatchDevReq<MockDevReq>;
        type TestBatchDeviceResp = MockBatchDevResp<MockDevResp>;

        let mut inner =
            MockScheduler::<TestUserReq, MockDevReq, MockDevResp, TestBatchDeviceReq, TestBatchDeviceResp>::new();
        inner.expect_enqueue().once().return_once(drop);
        inner.expect_swap_in().once().return_once(drop);
        inner.expect_pop_ready_reqs().once().return_once(|| None);
        inner.expect_can_flush().once().return_const(true);
        inner.expect_prepare().once().return_once(TestBatchDeviceReq::new);
        inner.expect_cancel().once().return_once(drop);
        inner.expect_commit().once().return_once(drop);

        let mut scheduler = InstrumentedScheduler::new(inner);
        scheduler.enqueue(TestUserReq::new());
        scheduler.swap_in(TestUserReq::new());
        assert!(scheduler.pop_ready_reqs().is_none());
        assert!(scheduler.can_flush());
        let batch_dev_req = scheduler.prepare();
        scheduler.cancel(batch_dev_req);
        scheduler.commit(TestBatchDeviceResp::new());

        assert_eq!(scheduler.num_enqueue, 1);
        assert_eq!(scheduler.num_swap_in, 1);
        assert_eq!(scheduler.hist_prepare.len(), 1);
        assert_eq!(scheduler.hist_cancel.len(), 1);
        assert_eq!(scheduler.hist_commit.len(), 1);
    }
}
