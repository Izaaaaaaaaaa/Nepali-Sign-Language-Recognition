import os
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_YAML = os.path.join(ROOT_DIR, 'hand_dataset', 'data.yaml')

# Load base model
model = YOLO('yolov8n.pt') 

if __name__ == '__main__':
    print("Training YOLO hand detector...")
    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        device='cpu', # Change to 0 if you have an NVIDIA GPU
        project=os.path.join(ROOT_DIR, 'runs'),
        name='hand_model'
    )
    print("Training complete. Move 'best.pt' from runs/hand_model/weights/ to models/yolov8_hand.pt")