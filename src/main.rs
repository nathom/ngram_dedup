use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::Result;
use std::fs::File;
use std::io::{BufReader, Write};

use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::Bfs;

fn find_connected_components(edges: &[(usize, usize)]) -> Vec<HashSet<usize>> {
    let mut graph = UnGraph::<usize, ()>::new_undirected();

    let mut node_map: HashMap<usize, NodeIndex> = HashMap::new();
    let mut reverse_map: HashMap<NodeIndex, usize> = HashMap::new();

    for &(i, j) in edges {
        let node_i = *node_map.entry(i).or_insert_with(|| {
            let idx = graph.add_node(i);
            reverse_map.insert(idx, i);
            idx
        });
        let node_j = *node_map.entry(j).or_insert_with(|| {
            let idx = graph.add_node(j);
            reverse_map.insert(idx, j);
            idx
        });
        graph.add_edge(node_i, node_j, ());
    }

    let mut visited = HashSet::new();
    let mut components = Vec::new();

    for node in graph.node_indices() {
        if !visited.contains(&node) {
            let mut component = HashSet::new();
            let mut bfs = Bfs::new(&graph, node);
            while let Some(nx) = bfs.next(&graph) {
                visited.insert(nx);
                component.insert(*reverse_map.get(&nx).unwrap());
            }
            components.push(component);
        }
    }

    components
}

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    data_file: String,

    /// Name of the person to greet
    #[arg(short, long)]
    output_path: String,

    /// Number of times to greet
    #[arg(short, long, default_value_t = 0.7)]
    threshold: f64,

    /// N-gram width
    #[arg(short, long, default_value_t = 3)]
    n: usize,
}

type Entry = String;

#[derive(Serialize)]
struct MergedEntry {
    article: String,
    ids: Vec<usize>,
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

fn get_pair_scores(ngrams: Vec<HashSet<String>>) -> Vec<(usize, usize, f64)> {
    let ngrams = &ngrams;

    let num_comparisons = (ngrams.len() * (ngrams.len() - 1)) / 2;
    let pb = &ProgressBar::new(num_comparisons as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed}/{duration}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
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
    results
}

fn print_histogram(scores: &Vec<(usize, usize, f64)>) {
    let mut histogram = histo::Histogram::with_buckets(100);
    for (_, _, score) in scores {
        histogram.add((score * 100.0).round() as u64);
    }
    println!("{histogram}");
}

fn preprocess(s: &Entry) -> Entry {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect()
}

fn preprocess_entries(entries: Vec<Entry>) -> Vec<Entry> {
    entries.par_iter().map(preprocess).collect()
}

fn main() {
    let args = Args::parse();

    let input_entries = read_articles_from_file(&args.data_file).unwrap();
    let input_entries = preprocess_entries(input_entries);
    let n = args.n;
    let threshold = args.threshold;
    println!("Merging articles with {n}-gram similarity > {threshold}");

    let ngrams: Vec<HashSet<String>> = input_entries
        .par_iter()
        .map(|a| generate_ngrams(&a, n))
        .collect();

    let results = get_pair_scores(ngrams);

    print_histogram(&results);

    let edge_list: Vec<_> = results
        .into_iter()
        .filter(|(_, _, score)| *score > threshold)
        .map(|(i, j, _)| (i, j))
        .collect();

    let mut inds_written: HashSet<usize> = HashSet::new();
    let components = find_connected_components(&edge_list);

    let mut merged_articles = vec![];

    // Write merged articles
    for connected_inds in &components {
        let mut ids: Vec<usize> = vec![];
        ids.extend(connected_inds);
        assert!(ids.len() >= 1);
        inds_written.extend(connected_inds);

        merged_articles.push(MergedEntry {
            article: input_entries[ids[0]].clone(),
            ids,
        });
    }

    let input_len = input_entries.len() as f64;
    // Write rest of articles
    let remaining_articles = input_entries
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !inds_written.contains(i))
        .map(|(i, entry)| MergedEntry {
            article: entry,
            ids: vec![i],
        });

    merged_articles.extend(remaining_articles);

    let output_len = merged_articles.len() as f64;
    println!(
        "Shrunk corpus from {} entries to {} ({}%)",
        input_len,
        merged_articles.len(),
        100.0 * (input_len - output_len) / input_len
    );

    let s = serde_json::to_string_pretty(&merged_articles).unwrap();
    let mut f = File::create(args.output_path).unwrap();
    f.write_all(s.as_bytes()).unwrap();
}
