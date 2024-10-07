use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use image::{DynamicImage, Rgb, RgbImage, RgbaImage};

use crate::utils::image_util::generate_color_for_classes;

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_id: usize,
    pub probability: f32,
}

impl BoundingBox {
    pub fn intersection(&self, other: &BoundingBox) -> f32 {
        (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0) * (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0)
    }

    pub fn union(&self, other: &BoundingBox) -> f32 {
        ((self.x2 - self.x1) * (self.y2 - self.y1)) + ((other.x2 - other.x1) * (other.y2 - other.y1)) - self.intersection(other)
    }

    pub fn iou(&self, other: &BoundingBox) -> f32 {
        self.intersection(other) / self.union(other)
    }
}


pub fn nms(boxes: Vec<BoundingBox>, iou_threshold: f32) -> Vec<BoundingBox> {
    let mut sorted_boxes = boxes.clone();
    sorted_boxes.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
    let mut result = Vec::new();
    while !sorted_boxes.is_empty() {
        let best_box = sorted_boxes.remove(0);
        result.push(best_box);
        sorted_boxes.retain(|bbox| best_box.iou(&bbox) < iou_threshold);
    }
    result
}


pub fn draw_boxes(image: &DynamicImage, boxes: &Vec<BoundingBox>, input_size: (u32, u32)) -> RgbImage {
    let (img_width, img_height) = (image.width(), image.height());
    let mut dt = DrawTarget::new(img_width as i32, img_height as i32);

    // Generate or retrieve the class colors based on the number of unique class_ids in the boxes
    let num_classes = boxes.iter().map(|bbox| bbox.class_id).max().unwrap_or(0) + 1;
    let class_colors = generate_color_for_classes(num_classes);

    for bbox in boxes {
        let mut pb = PathBuilder::new();
        pb.rect(
            bbox.x1 * img_width as f32 / input_size.0 as f32,
            bbox.y1 * img_height as f32 / input_size.1 as f32,
            (bbox.x2 - bbox.x1) * img_width as f32 / input_size.0 as f32,
            (bbox.y2 - bbox.y1) * img_height as f32 / input_size.1 as f32,
        );
        let path = pb.finish();

        // Retrieve the color for this class_id
        let color = class_colors.get(&bbox.class_id).unwrap_or(&SolidSource { r: 0x80, g: 0x10, b: 0x40, a: 0xFF });

        dt.stroke(
            &path,
            &Source::Solid(*color),
            &StrokeStyle {
                join: LineJoin::Round,
                width: 4.0,
                ..StrokeStyle::default()
            },
            &DrawOptions::new(),
        );
    }

    let box_image_rgba = RgbaImage::from_raw(img_width, img_height, dt.into_vec().iter().flat_map(|&x| x.to_ne_bytes()).collect()).expect("Failed to create image from raw data");
    let mut original_rgb = image.to_rgb8();
    for (x, y, pixel) in box_image_rgba.enumerate_pixels() {
        let orig_pixel = original_rgb.get_pixel_mut(x, y);
        let blend_pixel = Rgb([
            ((pixel[0] as u16 * pixel[3] as u16 + orig_pixel[0] as u16 * (255 - pixel[3]) as u16) / 255) as u8,
            ((pixel[1] as u16 * pixel[3] as u16 + orig_pixel[1] as u16 * (255 - pixel[3]) as u16) / 255) as u8,
            ((pixel[2] as u16 * pixel[3] as u16 + orig_pixel[2] as u16 * (255 - pixel[3]) as u16) / 255) as u8,
        ]);
        *orig_pixel = blend_pixel;
    }

    original_rgb
}


pub fn output_to_yolo_txt(boxes: Vec<BoundingBox>, image_width: u32, image_height: u32, output_path: &str) {
    let mut yolo_output = String::new();
    for bbox in boxes {
        let x_center = (bbox.x1 + bbox.x2) / 2.0;
        let y_center = (bbox.y1 + bbox.y2) / 2.0;
        let width = bbox.x2 - bbox.x1;
        let height = bbox.y2 - bbox.y1;
        yolo_output.push_str(&format!("{} {} {} {} {}\n", bbox.class_id, x_center / image_width as f32, y_center / image_height as f32, width / image_width as f32, height / image_height as f32));
    }
    std::fs::write(output_path, yolo_output).expect("Failed to write YOLO output to file");
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_intersection() {
        let a = BoundingBox { x1: -2.0, y1: -2.0, x2: -1.0, y2: -1.0, class_id: 0, probability: 0.0 };
        let b = BoundingBox { x1: 1.0, y1: 1.0, x2: 2.0, y2: 2.0, class_id: 0, probability: 0.0 };
        assert_eq!(a.intersection(&b), 0.0); //should be zero
    }
}