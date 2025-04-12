use std::sync::Arc;

use image::{ImageBuffer, ImageReader, Rgba};
use pollster::FutureExt;
use renderer::Renderer;
use winit::{
    application::ApplicationHandler,
    dpi::Size,
    event::MouseButton,
    event_loop::EventLoop,
    window::{Window, WindowAttributes},
};

mod init;
mod renderer;

const K: u32 = 10;
fn get_args() -> anyhow::Result<(Image, u32)> {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap();
    let image = ImageReader::open(path)?.decode()?.to_rgba8();
    let k = args
        .next()
        .map(|k| k.parse::<u32>())
        .transpose()?
        .unwrap_or(K);
    Ok((image, k))
}

pub type Image = ImageBuffer<Rgba<u8>, Vec<u8>>;

struct App<'a> {
    image: Image,
    k: u32,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer<'a>>,
}

impl<'a> App<'a> {
    fn new(image: Image, k: u32) -> Self {
        App {
            image,
            k,
            window: None,
            renderer: None,
        }
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(WindowAttributes::default().with_inner_size(Size::Physical(
                winit::dpi::PhysicalSize {
                    width: self.image.width(),
                    height: self.image.height(),
                },
            )))
            .unwrap();
        let window = Arc::new(window);
        let renderer = Renderer::new(window.clone(), self.image.clone(), self.k)
            .block_on()
            .unwrap();
        self.renderer = Some(renderer);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                self.renderer.as_mut().map(|r| r.render());
            }
            winit::event::WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                self.renderer.as_mut().map(|r| {
                    r.update_mouse_position(position.x as f32, position.y as f32);
                });
                self.window.as_mut().map(|window| window.request_redraw());
            }
            winit::event::WindowEvent::MouseInput {
                device_id: _,
                state,
                button: MouseButton::Left,
            } => {
                self.renderer
                    .as_mut()
                    .map(|r| r.mouse_clicked(state.is_pressed()));
                self.window.as_mut().map(|window| window.request_redraw());
            }

            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(8)
    //     .build_global()
    //     .unwrap();
    let (image, k) = get_args()?;
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let mut app = App::new(image, k);
    event_loop.run_app(&mut app)?;
    Ok(())
}
