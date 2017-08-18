use framing::{self, Image};
use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::image::{Dimensions, ImmutableImage, ImageCreationError};
use vulkano::format::{AcceptsPixels, Format, FormatDesc};
use vulkano::command_buffer::{AutoCommandBuffer, CommandBufferExecFuture};
use vulkano::sync::NowFuture;

/// Upload a frame to the GPU as a texture.
pub fn upload<W, T, F>(
    queue: Arc<Queue>,
    format: F,
    image: T,
) -> Result<(
        Arc<ImmutableImage<F>>,
        CommandBufferExecFuture<NowFuture, AutoCommandBuffer>
    ), ImageCreationError>
where
    T: Image,
    F: FormatDesc + AcceptsPixels<W> + Send + Sync + 'static,
    W: From<T::Pixel> + Clone + Send + Sync + 'static,
    Format: AcceptsPixels<W>
{
    let (w, h) = (image.width(), image.height());
    ImmutableImage::from_iter(
        (0..w * h).map(|i| unsafe {
            let (x, y) = (i % w, i / w);
            image.pixel(x, y).into()
        }),
        Dimensions::Dim2d {
            width: w as u32,
            height: h as u32
        },
        format,
        queue.clone()
    )
}

/// Upload a bunch of frames to the GPU as a texture array.
///
/// It is important to ensure that all the images are of the same size, because
/// otherwise this function will just return an `UnsupportedUsage` error.
///
/// # Panics
///
/// Panics if there are no images in the array.
pub fn upload_array<W, T, F>(
    queue: Arc<Queue>,
    format: F,
    images: &[T],
) -> Result<(
        Arc<ImmutableImage<F>>,
        CommandBufferExecFuture<NowFuture, AutoCommandBuffer>
    ), ImageCreationError>
where
    T: Image,
    F: FormatDesc + AcceptsPixels<W> + Send + Sync + 'static,
    W: From<T::Pixel> + Clone + Send + Sync + 'static,
    Format: AcceptsPixels<W>
{
    let (w, h) = (images[0].width(), images[0].height());
    let size = w * h;

    for image in images {
        if image.width() != w || image.height() != h {
            return Err(ImageCreationError::UnsupportedUsage);
        }
    }

    let mut iter = images.into_iter()
        .flat_map(|image| framing::iter(image))
        .map(|(_, _, pixel)| pixel.into());

    ImmutableImage::from_iter(
        // TODO(quadrupleslap): This is a horrifying hack.
        (0..size * images.len()).map(|_| iter.next().unwrap()),
        Dimensions::Dim2dArray {
            width: w as u32,
            height: h as u32,
            array_layers: images.len() as u32
        },
        format,
        queue.clone()
    )
}
