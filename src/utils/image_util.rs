use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb, ImageError};
use ndarray::{Array, ArrayBase, Dim, OwnedRepr, s};
use raqote::SolidSource;
use std::path::Path;
use std::fmt;
use std::collections::HashMap;

// FIXME: should it be imagenet mean and std?
const DEFAULT_MEAN: [f32; 3] = [0.0, 0.0, 0.0];
const DEFAULT_STD: [f32; 3] = [1.0, 1.0, 1.0];

#[derive(Debug)]
pub enum ImageLoadError {
    ImageError(ImageError),
    InvalidPath(String),
}

impl fmt::Display for ImageLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageLoadError::ImageError(err) => write!(f, "Image error: {}", err),
            ImageLoadError::InvalidPath(path) => write!(f, "Invalid image path: {}", path),
        }
    }
}

impl From<ImageError> for ImageLoadError {
    fn from(err: ImageError) -> ImageLoadError {
        ImageLoadError::ImageError(err)
    }
}

#[derive(Clone, Copy)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone)]
pub struct LoadedImageU8 {
    pub image_array: ArrayBase<OwnedRepr<u8>, Dim<[usize; 4]>>,
    pub size: ImageSize,
}

#[derive(Clone)]
pub struct LoadedImageF32 {
    pub image_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
    pub size: ImageSize,
}

impl fmt::Debug for ImageSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.width, self.height)
    }
}


pub fn load_image_u8(
    image_path: &str,
    target_size: (u32, u32)
) -> Result<LoadedImageU8, ImageLoadError> {

    if !Path::new(image_path).exists() {
        return Err(ImageLoadError::InvalidPath(image_path.to_string()));
    }

    let image = image::open(image_path).map_err(ImageLoadError::from)?;

    let (orig_width, orig_height) = (image.width(), image.height());
    let (target_width, target_height) = target_size;

    let scale = (target_width as f32 / orig_width as f32)
        .min(target_height as f32 / orig_height as f32);

    let new_width = (orig_width as f32 * scale).round() as u32;
    let new_height = (orig_height as f32 * scale).round() as u32;

    let resized_image = image.resize_exact(new_width, new_height, FilterType::Nearest).to_rgb8();

    let pad_left = (target_width - new_width) / 2;
    let pad_top = (target_height - new_height) / 2;

    let mut padded_image = ImageBuffer::from_pixel(target_width, target_height, Rgb([112, 112, 112]));
    for (x, y, pixel) in resized_image.enumerate_pixels() {
        padded_image.put_pixel(x + pad_left, y + pad_top, *pixel);
    }

    let array = Array::from_shape_fn((1, 3, target_height as usize, target_width as usize), |(_, c, j, i)| {
        let pixel = padded_image.get_pixel(i as u32, j as u32);
        pixel.channels()[c]
    });

    let size = ImageSize {
        width: target_width,
        height: target_height,
    };

    Ok(LoadedImageU8 {
        image_array: array,
        size,
    })
}



pub fn normalize_image_f32(
    loaded_image: &LoadedImageU8,  
    mean: Option<[f32; 3]>,
    std: Option<[f32; 3]>
) -> LoadedImageF32 {

    let mean = mean.unwrap_or(DEFAULT_MEAN);
    let std = std.unwrap_or(DEFAULT_STD);

    let mut array = loaded_image.image_array.mapv(|x| x as f32 / 255.0);

    for c in 0..3 {
        array.slice_mut(s![0, c, .., ..]).mapv_inplace(|x| (x - mean[c]) / std[c]);
    }

    LoadedImageF32 {
        image_array: array,
        size: loaded_image.size.clone(),  
    }
}

pub fn generate_color_for_classes(num_classes: usize) -> HashMap<usize, SolidSource> {
    let mut class_colors: HashMap<usize, SolidSource> = HashMap::new();
    for class_id in 0..num_classes {
        let hue = (class_id as f32 / num_classes as f32) * 360.0;
        let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0); // Full saturation and brightness
        class_colors.insert(class_id, SolidSource {
            r: (r * 255.0) as u8,
            g: (g * 255.0) as u8,
            b: (b * 255.0) as u8,
            a: 0xFF,
        });
    }
    class_colors
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match h as u32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        300..=359 => (c, 0.0, x),
        _ => (0.0, 0.0, 0.0),
    };

    (r + m, g + m, b + m)
}
