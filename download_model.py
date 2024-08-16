import argparse
import shutil
from ultralytics import YOLO
import os

def main(model_name : str) -> None:
    if not model_name.endswith('.pt'):
        model_name += '.pt'

    try:
        model = YOLO(model_name)
        model.export(format="onnx")

        onnx_file_name : str = model_name.replace(".pt", ".onnx")

        onnx_dir = os.path.join("./", "onnx")
        if not os.path.exists(onnx_dir):
            os.makedirs(onnx_dir)
        shutil.copy(onnx_file_name, onnx_dir)

        print(f"Model {model_name} converted to ONNX and saved to {onnx_file_name}")
        os.remove(onnx_file_name)

    except Exception as e:
        print(f"Error: {e}")
        print(f"Model {model_name} not converted to ONNX")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and convert YOLO model to ONNX')
    parser.add_argument('--model', type=str, default='yolov10n.pt', help='Model name or weights path')
    args = parser.parse_args()

    main(args.model)

