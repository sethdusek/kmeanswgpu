use image::{ImageBuffer, Rgba};
mod init;
pub mod renderer;

pub type Image = ImageBuffer<Rgba<u8>, Vec<u8>>;
