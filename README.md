# yolo-v10-rust
A Rust implementation of YOLOv10 using ONNX Runtime (ort)

![people sitting around a table](./output/people1.jpg)

## Overview
This small example demonstrates how to implement YOLOv10 in Rust using [ort](https://github.com/pykeio/ort) crate as a backend for running ONNX models. `ort` is a wrapper around [ONNX Runtime](https://onnxruntime.ai/).

## Getting Started

### Prerequisites
Ensure you have a basic Rust development environment set up. If you want to download a specific YOLO model, you'll also need Python with the `ultralytics` package installed.

### Running the demo
1. **(Optional) Download the YOLO Model**  
   To download a specific YOLOv10 model, use the following command:
   ```bash
   python ./download_model.py --model yolov10n.pt
   ```
   Alternatively, you can use the pre-included model at ./onnx/yolov10n.pt.

2. **Run the Rust Application**
```bash
cargo run -- path/to/your/image.jpg path/to/your/model.onnx
```

3. Check the output
The processed image and its corresponding detection results will be saved in the ./output directory. The output will include:
- A JPEG image with bounding boxes drawn with colors according to the class IDs.
- A text file containing the detection results in COCO format.

## Disclaimer
This project is my first attempt at Rust, so the code can be really messy.

## References
[ort](https://github.com/pykeio/ort)
[ultralytics](https://github.com/ultralytics/ultralytics)
[THU-MIG](https://github.com/THU-MIG/yolov10)
