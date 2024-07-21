use ahash::HashSet;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Result;
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize)]
struct Entry {
    article: String,
}

fn read_articles_from_file(file_path: &str) -> Result<Vec<Entry>> {
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    let articles: Vec<Entry> = serde_json::from_reader(reader)?;
    Ok(articles)
}

fn generate_ngrams(text: &str, n: usize) -> HashSet<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    words.windows(n).map(|window| window.join(" ")).collect()
}

fn jaccard(set1: &HashSet<String>, set2: &HashSet<String>) -> f64 {
    set1.intersection(set2).count() as f64 / set1.union(set2).count() as f64
}

fn main() {
    let articles = read_articles_from_file("data.json").unwrap();
    let n = 1;
    let threshold = 0.05;

    let ngrams: &Vec<HashSet<String>> = &articles
        .par_iter()
        .map(|a| generate_ngrams(&a.article, n))
        .collect();

    let num_comparisons = (ngrams.len() * (ngrams.len() - 1)) / 2;
    let pb = &ProgressBar::new(num_comparisons as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let results: Vec<(usize, usize, f64)> = (0..ngrams.len())
        .into_par_iter()
        .flat_map(|i| {
            (i + 1..ngrams.len()).into_par_iter().map(move |j| {
                pb.inc(1);
                (i, j, jaccard(&ngrams[i], &ngrams[j]))
            })
        })
        .collect();
    pb.finish();

    println!("Article pairs with similarity above {}: ", threshold);
    for (i, j, score) in results {
        if score > threshold {
            println!("Pair: ({}, {}), Jaccard: {}", i, j, score);
        }
    }
}
