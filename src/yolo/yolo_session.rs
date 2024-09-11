use image::{DynamicImage, RgbImage};
use ndarray::Array4;
use std::path::Path;

use super::yolo_utils::*;
use crate::{
    dnn::ort_inference_session::OrtInferenceSession,
    utils::image_util::{load_image_u8, normalize_image_f32, LoadedImageU8},
};

pub struct YoloSession {
    session: OrtInferenceSession,
    input_size: (u32, u32),
    use_nms: bool,
    model_name: String,
}

impl YoloSession {
    pub fn new(
        model_path: &Path,
        input_size: (u32, u32),
        use_nms: bool,
        model_name: String,
    ) -> ort::Result<Self> {
        let session = OrtInferenceSession::new(model_path)?;
        Ok(YoloSession {
            session,
            input_size,
            use_nms,
            model_name,
        })
    }

    pub fn run_inference(&self, input_tensor: Array4<f32>) -> Vec<BoundingBox> {
        let outputs = self
            .session
            .run_inference(input_tensor)
            .expect("Inference failed");
        let output = outputs["output0"]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract tensor")
            .into_owned();
        let mut boxes = Vec::new();
        println!("output shape: {:?}", output.shape());

        let original_output_shape = output.shape();


        // FIXME: This is a temporary solution to handle different output shapes

        if self.model_name == "yolov10" {
            let reshaped_output = output
                .to_shape((original_output_shape[1], original_output_shape[2]))
                .expect("Failed to reshape the output");
            for detection in reshaped_output.outer_iter() {
                let bbox_coords = [detection[0], detection[1], detection[2], detection[3]];
                let object_confidence = detection[4];
                let class_id = detection[5] as usize;

                let bbox = BoundingBox {
                    x1: bbox_coords[0],
                    y1: bbox_coords[1],
                    x2: bbox_coords[2],
                    y2: bbox_coords[3],
                    class_id,
                    probability: object_confidence,
                };

                if object_confidence >= 0.25 {
                    boxes.push(bbox);
                }
            }
        } else if self.model_name == "yolov8" {
            let reshaped_output = output
                .to_shape((original_output_shape[1], original_output_shape[2]))
                .expect("Failed to reshape the output");

            // FIXME: it needs to be implemented in a more generic way
            // and should be optimized
            // loop for each detection (total 8400)
            for i in 0..reshaped_output.shape()[1]{
                // get x,y,w,h
                let x = reshaped_output[[0, i]];
                let y = reshaped_output[[1, i]];
                let w = reshaped_output[[2, i]];
                let h = reshaped_output[[3, i]];

                // get the highest class probability
                let mut max_class_prob = 0.0;
                let mut max_class_id = 0;

                for j in 4..reshaped_output.shape()[0]{
                    if reshaped_output[[j, i]] > max_class_prob{
                        max_class_prob = reshaped_output[[j, i]];
                        max_class_id = j - 4;
                    }
                }

                if max_class_prob > 0.25{
                    let bbox = BoundingBox {
                        x1: x - w / 2.0,
                        y1: y - h / 2.0,
                        x2: x + w / 2.0,
                        y2: y + h / 2.0,
                        class_id: max_class_id,
                        probability: max_class_prob,
                    };
                    boxes.push(bbox);
                }
        }
     } else {
            panic!(
                "Unsupported model name: {}.\n Model should be 'yolov10' or 'yolov8'.",
                self.model_name
            );
        }
        boxes
    }

    pub fn load_and_preprocess_image(&self, image_path: &str) -> (RgbImage, LoadedImageU8) {
        let loaded_image: LoadedImageU8 =
            load_image_u8(image_path, self.input_size).expect("Failed to preprocess image");

        let interleaved_data: Vec<u8> = loaded_image
            .image_array
            .view()
            .to_shape((
                3,
                loaded_image.size.height as usize,
                loaded_image.size.width as usize,
            ))
            .unwrap()
            .permuted_axes((1, 2, 0))
            .iter()
            .cloned()
            .collect();

        let img = RgbImage::from_raw(
            loaded_image.size.width,
            loaded_image.size.height,
            interleaved_data,
        )
        .expect("Failed to create image from raw data");

        (img, loaded_image)
    }

    pub fn save_outputs(
        &self,
        image: RgbImage,
        boxes: Vec<BoundingBox>,
        image_path: &str,
        output_dir: Option<&str>,
    ) {
        let output_dir_str = output_dir.unwrap_or("output");
        let output_dir = Path::new(output_dir_str);
        if !output_dir.exists() {
            std::fs::create_dir(output_dir).expect("Failed to create output directory");
        }

        let file_name = Path::new(image_path).file_stem().unwrap();
        let image_output_path = output_dir.join(format!("{}.jpg", file_name.to_string_lossy()));
        let txt_output_path = output_dir.join(format!("{}.txt", file_name.to_string_lossy()));

        image
            .save(image_output_path)
            .expect("Failed to save output image");

        let (image_width, image_height) = image.dimensions();
        output_to_yolo_txt(
            boxes,
            image_width,
            image_height,
            txt_output_path.to_str().unwrap(),
        );
    }

    pub fn process_image(&self, image_path: &str) {
        let (original_image, loaded_image) = self.load_and_preprocess_image(image_path);

        let normalized_image = normalize_image_f32(&loaded_image, None, None);
        let mut inferred_boxes = self.run_inference(normalized_image.image_array);

        // YOLOv10 does not require NMS
        // you can use them for other models
        // XXX: it's not hard coded as "0.45", which is the default value for Ultralytics YOLOv8
        if self.use_nms {
            inferred_boxes = nms(inferred_boxes, 0.45);
        }

        let result_image = draw_boxes(
            &DynamicImage::ImageRgb8(original_image),
            &inferred_boxes,
            self.input_size,
        );

        self.save_outputs(result_image, inferred_boxes, image_path, None);
    }
}
