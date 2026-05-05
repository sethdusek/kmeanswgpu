//! Initialization methods for k-means (Forgy, K-means++)

use rand::{Rng, rng};
use rand_distr::{Distribution, Uniform};
use rayon::prelude::*;

use crate::Image;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum InitializationMethod {
    Forgy,
    Kmeanspp,
}

impl InitializationMethod {
    pub fn initialize(&self, input_image: &Image, k: u32) -> Vec<[u8; 4]> {
        match self {
            InitializationMethod::Forgy => forgy(input_image, k),
            InitializationMethod::Kmeanspp => kmeans_pp(input_image, k),
        }
    }
}

pub fn forgy(image: &Image, k: u32) -> Vec<[u8; 4]> {
    (0..k)
        .map(|_| {
            image
                .get_pixel(
                    rng().random_range(0..image.width()),
                    rng().random_range(0..image.height()),
                )
                .0
        })
        .collect()
}

fn dist_euclid_squared(a: [u8; 4], b: [u8; 4]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a.abs_diff(b) as u64).pow(2))
        .sum()
}

// weighted distribution based on rand WeightedIndex but with some optimizations to avoid re-allocating when updating all weights
struct WeightedDistr {
    cumulative_weights: Vec<u64>,
    weight_distribution: rand::distr::uniform::Uniform<u64>,
}

impl WeightedDistr {
    fn new(weights: &[u64]) -> Self {
        let mut weights_prefix = Vec::with_capacity(weights.len());
        let sampler = Self::update_inner(&mut weights_prefix, weights);
        Self {
            cumulative_weights: weights_prefix,
            weight_distribution: sampler,
        }
    }
    fn update(&mut self, new_weights: &[u64]) {
        self.weight_distribution = Self::update_inner(&mut self.cumulative_weights, new_weights);
    }
    fn update_inner(weights_prefix: &mut Vec<u64>, new_weights: &[u64]) -> Uniform<u64> {
        weights_prefix.clear();
        weights_prefix.reserve(new_weights.len() - 1);
        let mut total_weight = new_weights[0];
        for w in &new_weights[1..] {
            // safety: weights_prefix.capacity() == new_weights.len() - 1
            // for some reason rustc doesn't see the above invariant holds \
            // and adds a branch checking for spare capacity each iteration, this assert lets it "see" that this is always true
            // makes loop over twice as fast (4.2 billion elements/second), yay
            unsafe {
                std::hint::assert_unchecked(weights_prefix.capacity() > weights_prefix.len());
            }
            weights_prefix.push(total_weight);

            // safety: rand uses checked addition, but we don't need it here (for performance)
            // some quick math: the maximum distance two u8x4 vectors can have is 255.pow(2) * 4 = 260,100
            // with a u64 we could accumulate 70,921,738,076,545 such elements before overflowing
            total_weight += w;
        }
        Uniform::new(0, total_weight).unwrap()
    }
}

impl Distribution<usize> for WeightedDistr {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let chosen_weight = self.weight_distribution.sample(rng);
        // Find the first item which has a weight *higher* than the chosen weight.
        self.cumulative_weights
            .partition_point(|w| w <= &chosen_weight)
    }
}

pub fn kmeans_pp(image: &Image, k: u32) -> Vec<[u8; 4]> {
    let mut rng = rand::rng();
    let mut centroids = Vec::with_capacity(k as usize);
    centroids.push(
        image
            .get_pixel(
                rng.random_range(0..image.width()),
                rng.random_range(0..image.height()),
            )
            .0,
    );
    let mut distr = WeightedDistr::new(&[1]);
    let mut weights = vec![];
    for _ in 1..k {
        image
            .par_pixels()
            .map(|p| {
                centroids
                    .iter()
                    .map(|c| dist_euclid_squared(p.0, *c))
                    .min()
                    .unwrap()
            })
            .collect_into_vec(&mut weights);
        distr.update(&weights);
        let idx = distr.sample(&mut rng) as u32;
        centroids.push(image.get_pixel(idx % image.width(), idx / image.width()).0);
        weights.clear();
    }
    centroids
}
