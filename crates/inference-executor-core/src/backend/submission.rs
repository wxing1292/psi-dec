/// Submitted or in-flight backend work.
pub trait Submission {
    fn wait(&self);
}
