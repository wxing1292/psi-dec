use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);
static PROFILING_SUMMARY_EVERY: AtomicU64 = AtomicU64::new(0);
static TREE_PROFILE_SUMMARY: OnceLock<Mutex<TreeProfileSummary>> = OnceLock::new();

thread_local! {
    static TREE_PROFILE_STACK: std::cell::RefCell<Vec<TreeProfileFrame>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub fn set_profiling_summary_every(every: u64) {
    PROFILING_SUMMARY_EVERY.store(every, Ordering::Relaxed);
    PROFILING_ENABLED.store(every > 0, Ordering::Relaxed);
}

pub fn span(name: &'static str) -> TreeProfileGuard {
    if !PROFILING_ENABLED.load(Ordering::Relaxed) {
        return TreeProfileGuard::inactive();
    }

    TREE_PROFILE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if matches!(stack.last(), Some(frame) if frame.name == name) {
            return TreeProfileGuard::inactive();
        }

        let path = stack
            .last()
            .map(|parent| format!("{}.{}", parent.path, name))
            .unwrap_or_else(|| name.to_string());
        stack.push(TreeProfileFrame {
            name,
            path,
            start: Instant::now(),
            child_ms: 0.0,
        });

        TreeProfileGuard::active()
    })
}

pub fn maybe_emit_tree_profile_summary(label: &'static str, model_total_count: u64) {
    let every = PROFILING_SUMMARY_EVERY.load(Ordering::Relaxed).max(32);
    if !PROFILING_ENABLED.load(Ordering::Relaxed) || model_total_count == 0 || !model_total_count.is_multiple_of(every)
    {
        return;
    }

    let Some(summary) = TREE_PROFILE_SUMMARY.get() else {
        return;
    };
    let summary = summary
        .lock()
        .expect("runtime service tree profile summary mutex should not be poisoned");
    emit_tree_profile_summary(label, model_total_count, &summary.stats_by_path);
}

pub struct TreeProfileGuard {
    active: bool,
}

impl TreeProfileGuard {
    fn active() -> Self {
        Self { active: true }
    }

    fn inactive() -> Self {
        Self { active: false }
    }
}

impl Drop for TreeProfileGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        TREE_PROFILE_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            let Some(frame) = stack.pop() else {
                return;
            };
            let inclusive_ms = frame.start.elapsed().as_secs_f64() * 1_000.0;
            let exclusive_ms = (inclusive_ms - frame.child_ms).max(0.0);
            if let Some(parent) = stack.last_mut() {
                parent.child_ms += inclusive_ms;
            }
            record_tree_duration(&frame.path, inclusive_ms, exclusive_ms);
        });
    }
}

struct TreeProfileFrame {
    name: &'static str,
    path: String,
    start: Instant,
    child_ms: f64,
}

#[derive(Default)]
struct TreeProfileSummary {
    stats_by_path: HashMap<String, TreeProfileStats>,
}

#[derive(Clone, Copy)]
struct TreeProfileStats {
    count: u64,
    inclusive_total_ms: f64,
    exclusive_total_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl TreeProfileStats {
    fn new(inclusive_ms: f64, exclusive_ms: f64) -> Self {
        Self {
            count: 1,
            inclusive_total_ms: inclusive_ms,
            exclusive_total_ms: exclusive_ms,
            min_ms: inclusive_ms,
            max_ms: inclusive_ms,
        }
    }

    fn record(&mut self, inclusive_ms: f64, exclusive_ms: f64) {
        self.count += 1;
        self.inclusive_total_ms += inclusive_ms;
        self.exclusive_total_ms += exclusive_ms;
        self.min_ms = self.min_ms.min(inclusive_ms);
        self.max_ms = self.max_ms.max(inclusive_ms);
    }

    fn inclusive_avg_ms(self) -> f64 {
        self.inclusive_total_ms / self.count as f64
    }

    fn exclusive_avg_ms(self) -> f64 {
        self.exclusive_total_ms / self.count as f64
    }
}

fn record_tree_duration(path: &str, inclusive_ms: f64, exclusive_ms: f64) {
    let mut summary = TREE_PROFILE_SUMMARY
        .get_or_init(|| Mutex::new(TreeProfileSummary::default()))
        .lock()
        .expect("runtime service tree profile summary mutex should not be poisoned");
    summary
        .stats_by_path
        .entry(path.to_string())
        .and_modify(|stats| stats.record(inclusive_ms, exclusive_ms))
        .or_insert_with(|| TreeProfileStats::new(inclusive_ms, exclusive_ms));
}

fn emit_tree_profile_summary(
    label: &'static str,
    model_total_count: u64,
    stats_by_path: &HashMap<String, TreeProfileStats>,
) {
    let mut rows = stats_by_path
        .iter()
        .map(|(path, stats)| (path.clone(), *stats))
        .collect::<Vec<_>>();
    rows.sort_by(|(left, _), (right, _)| left.cmp(right));

    for (path, stats) in rows {
        tracing::debug!(
            target: "inference-runtime-service::profile",
            phase = "profile.tree_summary",
            label,
            seq = model_total_count,
            path = %path,
            n = stats.count,
            incl_ms = stats.inclusive_total_ms,
            incl_avg_ms = stats.inclusive_avg_ms(),
            excl_ms = stats.exclusive_total_ms,
            excl_avg_ms = stats.exclusive_avg_ms(),
            min_ms = stats.min_ms,
            max_ms = stats.max_ms,
        );
    }
}
