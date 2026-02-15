from ultralytics import YOLO
import os

# 1. Load the model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'yolov8_hand.pt')
model = YOLO(model_path)

# 2. Pick an image from your hand_dataset (Choose one you know has a hand)
# CHANGE THIS PATH to a real image on your computer
img_test = r"D:\Nepali_SLR_Project\data\sequences\naam\3\39.jpg" 

if os.path.exists(img_test):
    results = model(img_test, conf=0.1)
    # This will pop up a window using YOLO's built-in viewer
    results[0].show() 
    print(f"Detected {len(results[0].boxes)} objects.")
else:
    print("Image not found. Please paste a correct path to a JPG.")