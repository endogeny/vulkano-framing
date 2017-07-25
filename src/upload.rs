use framing::video::VideoFrame;
use std::iter;
use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::image::{Dimensions, ImmutableImage, ImageCreationError};
use vulkano::format::{AcceptsPixels, FormatDesc};
use vulkano::command_buffer::{AutoCommandBuffer, CommandBufferExecFuture};
use vulkano::sync::NowFuture;

/// Upload a frame to the GPU.
pub fn upload<W, T, F>(
    queue: Arc<Queue>,
    format: F,
    image: T,
) -> Result<
    (Arc<ImmutableImage<F>>, CommandBufferExecFuture<NowFuture, AutoCommandBuffer>),
    ImageCreationError,
>
where
    T: VideoFrame,
    F: FormatDesc + AcceptsPixels<W> + Send + Sync + 'static,
    W: From<T::Pixel> + Clone + Send + Sync + 'static,
{

    let (w, h) = (image.width(), image.height());
    ImmutableImage::from_iter(
        (0..w * h).map(|i| unsafe {
            let (x, y) = (i % w, i / w);
            image.pixel(x, y).into()
        }),
        Dimensions::Dim2d {
            width: w as u32,
            height: h as u32,
        },
        format,
        iter::once(queue.family()),
        queue.clone(),
    )
}
