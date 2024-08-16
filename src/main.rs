use std::env;
use std::path::Path;
use yolo_v10_rust::yolo::yolo_session::YoloSession;

fn main() {
    let args: Vec<String> = env::args().collect();
    let image_path = if args.len() > 1 {
        &args[1]
    } else {
        println!("Warning: No image path provided. Using default image 'sample/people0.jpg'.");
        "sample/horses0.jpg"
    };

    let model_path = if args.len() > 2 {
        Path::new(&args[2])
    } else {
        println!("Warning: No model path provided. Using default model 'onnx/yolov10n.onnx'.");
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("onnx")
            .join("yolov10n.onnx")
    };

    let yolo_model = YoloSession::new(
        &model_path,
        (640, 640),
    ).expect("Failed to create YOLO model");

    yolo_model.process_image(image_path);
}
