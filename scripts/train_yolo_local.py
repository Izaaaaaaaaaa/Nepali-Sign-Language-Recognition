import os
from ultralytics import YOLO

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# This is the standard path where YOLO saves the "last" checkpoint
CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'runs', 'detect', 'hand_model', 'weights', 'last.pt')
DATA_YAML = os.path.join(ROOT_DIR, 'hand_dataset', 'data.yaml')

def train_or_resume():
    # 1. Check if the 'last.pt' file exists (meaning it crashed or paused)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"--- CHECKPOINT FOUND ---")
        print(f"Resuming from: {CHECKPOINT_PATH}")
        
        # Load the interrupted model
        model = YOLO(CHECKPOINT_PATH)
        
        # Resume the training
        # Note: You don't need to pass 'data' or 'epochs' again; 
        # YOLO remembers them from the checkpoint!
        model.train(resume=True)
        
    else:
        # 2. If no checkpoint exists, start a new training session
        print("--- NO CHECKPOINT FOUND ---")
        print("Starting a new training session from scratch...")
        
        model = YOLO('yolov8n.pt') # Start with the Nano base model
        model.train(
            data=DATA_YAML,
            epochs=50,
            imgsz=640,
            device='cpu',  # Change to 0 if you have a GPU
            workers=0,
            project=os.path.join(ROOT_DIR, 'runs', 'detect'),
            name='hand_model',
            resume=False
        )

if __name__ == '__main__':
    train_or_resume()