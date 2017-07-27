//! Puts the image in a triangle and overlays a silly-looking gradient.

#[macro_use] extern crate vulkano;
#[macro_use] extern crate vulkano_shader_derive;
extern crate vulkano_framing;
extern crate png_framing;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::image::AttachmentImage;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, Features};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::sampler::{Filter, MipmapMode, SamplerAddressMode, Sampler};
use vulkano::sync::GpuFuture;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

use png_framing::Png;
use vulkano_framing::Buffer;
use std::sync::Arc;

#[derive(Clone, Debug)]
struct Vertex {
    pos: [f32; 3],
    tex: [f32; 2]
}
impl_vertex!(Vertex, pos, tex);

fn main() {
    let image = Png::decode(include_bytes!("todoroki.png")).unwrap();

    let format = Format::R8G8B8A8Unorm;
    let (w, h) = (1920, 1200);
    let instance_ext = InstanceExtensions::none();
    let device_ext = DeviceExtensions::none();
    let features = Features::none();

    let instance = Instance::new(None, &instance_ext, None).unwrap();
    let phy = PhysicalDevice::enumerate(&instance).next().unwrap();

    let queue_family =
        phy.queue_families()
            .find(|&q| q.supports_graphics())
            .unwrap();

    let (device, mut queues) =
        Device::new(
            phy,
            &features,
            &device_ext,
            [(queue_family, 0.5)].iter().cloned()
        ).unwrap();

    let queue = queues.next().unwrap();

    let (texture, texture_future) =
        vulkano_framing::upload::<[u8; 4], _, _>(
            queue.clone(),
            format,
            image
        ).unwrap();

    texture_future.then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let output: Arc<CpuAccessibleBuffer<[[u8; 4]]>> = unsafe {
        CpuAccessibleBuffer::uninitialized_array(
            device.clone(),
            w * h,
            BufferUsage::all(),
            Some(queue_family)
        ).unwrap()
    };

    let output = Buffer::new(output, w, h).unwrap();

    macro_rules! vertices {
        ($($pos:expr, $tex:expr),*) => {
            [$( Vertex { pos: $pos, tex: $tex } ),*]
        }
    }

    let vertices = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        Some(queue.family()),
        vertices![
            [-0.8,  0.8, 0.0], [0.0, 1.0],
            [ 0.8,  0.8, 0.0], [1.0, 1.0],
            [ 0.0, -0.8, 0.0], [0.5, 0.0]
        ].iter().cloned()
    ).unwrap();

    let vs = VertexShader::load(device.clone()).unwrap();
    let fs = FragmentShader::load(device.clone()).unwrap();

    let render_pass = Arc::new(single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: DontCare,
                store: Store,
                format: format,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    ).unwrap());

    let staging = Arc::new(
        AttachmentImage::new(
            device.clone(),
            [w as u32, h as u32],
            format
        ).unwrap()
    );

    let framebuffer = Arc::new(
        Framebuffer::start(render_pass.clone())
            .add(staging.clone())
            .unwrap()
            .build()
            .unwrap()
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports(Some(Viewport {
                origin: [0.0, 0.0],
                depth_range: 0.0..1.0,
                dimensions: [w as f32, h as f32],
            }))
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap()
    );

    let sampler = Sampler::new(
        device.clone(),
        Filter::Linear,
        Filter::Linear,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0, 1.0, 0.0, 0.0
    ).unwrap();

    let set = Arc::new(
        PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_sampled_image(texture.clone(), sampler.clone())
            .unwrap()
            .build()
            .unwrap()
    );

    let commands =
        AutoCommandBufferBuilder::primary(device.clone(), queue.family())
            .unwrap()

            .begin_render_pass(
                framebuffer.clone(),
                false,
                vec![[0.0, 1.0, 0.0, 1.0].into()]
            )
            .unwrap()

            .draw(
                pipeline.clone(),
                DynamicState::none(),
                vertices.clone(),
                set.clone(),
                ()
            )
            .unwrap()

            .end_render_pass()
            .unwrap()

            .copy_image_to_buffer(staging.clone(), output.buffer().clone())
            .unwrap()

            .build()
            .unwrap();

    commands.execute(queue)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    Png::new(output.read().unwrap()).save("output.png").unwrap();
    println!("Saved image to `output.png`!");
}

macro_rules! shaders {
    ($(#[ kind = $kind:expr ] $name:ident $code:expr)*) => {
        #[doc(hidden)]
        mod shaders {$(
            #[allow(dead_code, non_snake_case)]
            pub mod $name {
                #[derive(VulkanoShader)]
                #[src = $code]
                #[ty = $kind]
                struct Dummy;
            }
        )*}

        $(use self::shaders::$name::Shader as $name;)*
    }
}

shaders! {
    #[kind = "vertex"]
    VertexShader "
        #version 450

        layout(location = 0) in vec3 pos;
        layout(location = 1) in vec2 tex;

        layout(location = 0) out vec2 tex_coords;

        void main() {
            gl_Position = vec4(pos, 1.0);
            tex_coords = tex;
        }
    "

    #[kind = "fragment"]
    FragmentShader "
        #version 450

        layout(location = 0) in vec2 tex_coords;
        layout(location = 0) out vec4 color;

        layout(set = 0, binding = 0) uniform sampler2D tex;

        void main() {
            color = texture(tex, tex_coords) + vec4(tex_coords, 0.0, 1.0);
        }
    "
}
