#![feature(portable_simd, slice_as_chunks)]

use image::{ImageBuffer, Rgba};
mod init;
pub mod renderer;

pub type Image = ImageBuffer<Rgba<u8>, Vec<u8>>;
