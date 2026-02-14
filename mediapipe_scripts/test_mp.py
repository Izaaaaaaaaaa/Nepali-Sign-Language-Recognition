# Test script
try:
    import mediapipe as mp
    print(f"MediaPipe version: {mp.__version__}")
    print(f"Available: {dir(mp)}")
    
    # Try to access hands
    hands = mp.solutions.hands
    print("✓ SUCCESS: MediaPipe working!")
except Exception as e:
    print(f"✗ ERROR: {e}")