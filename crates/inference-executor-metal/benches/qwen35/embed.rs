#[path = "output.rs"]
mod head;

fn main() {
    head::run(vec![head::Case::Embed], "qwen35_embed");
}
