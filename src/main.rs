use std::env;
use std::path::Path;
use yolo_rust_ort::yolo::yolo_session::YoloSession;

fn main() {
    let args: Vec<String> = env::args().collect();
    let image_path = match args.get(1) {
        Some(arg) => arg,
        None => {
            println!("Warning: No image path provided. Using default image 'sample/horses0.jpg.");
            "sample/horses0.jpg"
        }
    };
    let model_path = match args.get(2) {
        Some(arg) => Path::new(arg),
        None => {
            println!("Warning: No model path provided. Using default model 'onnx/yolov10n.onnx'.");
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("onnx").join("yolov10n.onnx")
        }
    };
    let model_name = match args.get(3) {
        Some(arg) => arg.clone(),
        None => {
            println!("Warning: No model name provided. Using default model name 'yolov10'.");
            "yolov10".to_string()
        }
    };
    let use_nms = match args.get(4) {
        Some(arg) => arg.parse::<bool>()
            .map_err(|e| println!("Error parsing nms argument: {}. Defaulting to false.", e))
            .unwrap_or(false),
        None => {
            println!("Warning: No NMS flag provided. Using default NMS flag 'false'.");
            false
        }
    };

    let yolo_model = YoloSession::new(&model_path, (640, 640), use_nms, model_name)
        .expect("Failed to create YOLO model");

    yolo_model.process_image(image_path);
}
