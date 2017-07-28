use std::error::Error as StdError;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use vulkano::buffer::{BufferAccess, CpuAccessibleBuffer};
use vulkano::buffer::cpu_access::{
    ReadLock,
    WriteLock,
    ReadLockError,
    WriteLockError
};
use framing::video::VideoFrame;

/// Wraps a buffer, providing functions that allow the user to interpret it as
/// a frame instead of as a bunch of pixels.
#[derive(Clone, Debug)]
pub struct Buffer<T> {
    inner: Arc<CpuAccessibleBuffer<[T]>>,
    width: usize,
    height: usize
}

impl<T> Buffer<T> where T: Send + Sync + 'static {
    /// Creates the wrapper.
    ///
    /// Expects a width and height in pixels, and a buffer of length
    /// `width * height`. A buffer of incorrect length will cause a `BadLength`
    /// error to be returned.
    pub fn new(
        inner: Arc<CpuAccessibleBuffer<[T]>>,
        width: usize,
        height: usize
    ) -> Result<Self, BadLength> {
        let expected_len = width * height;
        let actual_len = inner.len();

        if expected_len == actual_len {
            Ok(Self {
                inner,
                width,
                height
            })
        } else {
            Err(BadLength {
                expected_len,
                actual_len
            })
        }
    }
}

impl<T: 'static> Buffer<T> {
    /// Try to get a read-only frame from the buffer.
    pub fn read(&self) -> Result<Reader<T>, ReadLockError> {
        let inner = self.inner.read()?;
        let (width, height) = (self.width, self.height);
        Ok(Reader { inner, width, height })
    }

    /// Try to get a mutable frame from the buffer.
    pub fn write(&self) -> Result<Writer<T>, WriteLockError> {
        let inner = self.inner.write()?;
        let (width, height) = (self.width, self.height);
        Ok(Writer { inner, width, height })
    }
}

impl<T> Buffer<T> {
    /// Get back the underlying buffer.
    pub fn buffer(&self) -> &Arc<CpuAccessibleBuffer<[T]>> {
        &self.inner
    }

    /// The width of the image, in pixels.
    pub fn width(&self) -> usize { self.width }

    /// The height of the image, in pixels.
    pub fn height(&self) -> usize { self.height }
}

/// A read-only frame.
pub struct Reader<'a, T: 'a> {
    inner: ReadLock<'a, [T]>,
    width: usize,
    height: usize
}

impl<'a, T: 'a> VideoFrame for Reader<'a, T> where T: Clone {
    type Pixel = T;

    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }

    unsafe fn pixel(&self, x: usize, y: usize) -> Self::Pixel {
        self.inner.get_unchecked(y * self.width + x).clone()
    }
}

impl<'a, T: 'a> Deref for Reader<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// A mutable frame.
pub struct Writer<'a, T: 'a> {
    inner: WriteLock<'a, [T]>,
    width: usize,
    height: usize
}

impl<'a, T: 'a> VideoFrame for Writer<'a, T> where T: Clone {
    type Pixel = T;

    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }

    unsafe fn pixel(&self, x: usize, y: usize) -> Self::Pixel {
        self.inner.get_unchecked(y * self.width + x).clone()
    }
}

impl<'a, T: 'a> Deref for Writer<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, T: 'a> DerefMut for Writer<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// A struct representing a buffer length mismatch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BadLength {
    expected_len: usize,
    actual_len: usize
}

impl fmt::Display for BadLength {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "the buffer had to have {} pixels, but had {} pixels",
            self.expected_len,
            self.actual_len
        )
    }
}

impl StdError for BadLength {
    fn description(&self) -> &str {
        "incorrect buffer length"
    }
}
