#![warn(missing_docs)]

//! Use Vulkan buffers as video frames / images!

// TODO(quadrupleslap): Use stable Vulkan.

extern crate framing;
extern crate vulkano;

mod buffer;
mod upload;

pub use buffer::*;
pub use upload::*;
