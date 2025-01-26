import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

# Function to overlay sunglasses
def overlay_sunglasses(frame, sunglasses, landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    width = int(abs(right_eye[0] - left_eye[0]) * 2)
    height = int(width * sunglasses.size[1] / sunglasses.size[0])
    x = int((left_eye[0] + right_eye[0]) / 2 - width / 2)
    y = int((left_eye[1] + right_eye[1]) / 2 - height / 2)

    # Clamp to frame bounds
    x = max(0, x)
    y = max(0, y)
    width = min(width, frame.shape[1] - x)
    height = min(height, frame.shape[0] - y)

    if width > 0 and height > 0:
        sunglasses_resized = sunglasses.resize((width, height), Image.Resampling.LANCZOS)
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        frame_pil.paste(sunglasses_resized, (x, y), sunglasses_resized)
        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    print("Sunglasses placement out of bounds.")
    return frame

# Load sunglasses image
sunglasses = Image.open("sunglasses.png").convert("RGBA")

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Start webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Process face landmarks
    if results.multi_face_landmarks:
        h, w, c = frame.shape
        landmarks = [
            (int(lm.x * w), int(lm.y * h))
            for lm in results.multi_face_landmarks[0].landmark
        ]
        frame = overlay_sunglasses(frame, sunglasses, landmarks)

    # Display the frame
    cv2.imshow("Sunglasses Try-On", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
