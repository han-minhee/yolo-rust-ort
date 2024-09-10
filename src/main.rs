use std::env;
use std::path::Path;
use yolo_rust_ort::yolo::yolo_session::YoloSession;

fn main() {
    let args: Vec<String> = env::args().collect();
    let image_path = if args.len() > 1 {
        &args[1]
    } else {
        println!("Warning: No image path provided. Using default image 'sample/horses0.jpg'.");
        "sample/horses0.jpg"
    };

    let default_model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("onnx")
        .join("yolov10n.onnx");

    let model_path = if args.len() > 2 {
        Path::new(&args[2])
    } else {
        println!("Warning: No model path provided. Using default model 'onnx/yolov10n.onnx'.");
        &default_model_path
    };

    let model_name = if args.len() > 3 {
        args[3].clone()
    } else {
        println!("Warning: No model name provided. Using default model name 'yolov10'.");
        "yolov10".to_string()
    };

    let use_nms = if args.len() > 4 {
        args[4].parse::<bool>().unwrap_or(false)
    } else {
        println!("Warning: No NMS flag provided. Using default NMS flag 'false'.");
        false
    };

    let yolo_model = YoloSession::new(
        &model_path,
        (640, 640),
        use_nms,
        model_name,
    ).expect("Failed to create YOLO model");

    yolo_model.process_image(image_path);
}
