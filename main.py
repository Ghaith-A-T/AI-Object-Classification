import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load model
model = load_model("keras_model.h5", compile=False)

# Load labels
class_names = [line.strip().split(" ", 1)[1] for line in open("labels.txt", "r")]

print("Model loaded successfully!")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Copy image for display
    display = frame.copy()

    # Resize image for model
    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.expand_dims(normalized_image_array, axis=0)

    prediction = model.predict(data, verbose=0)

    index = np.argmax(prediction)
    confidence = float(prediction[0][index])

    object_name = class_names[index]

    # Confidence threshold
    if confidence < 0.80:
        object_name = "Unknown"
        color = (0, 0, 255)      # Red
    else:
        color = (0, 255, 0)      # Green

    # Draw border
    cv2.rectangle(display, (10, 10), (630, 470), color, 3)

    # Object name
    cv2.putText(
        display,
        f"Object: {object_name}",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    # Confidence
    cv2.putText(
        display,
        f"Confidence: {confidence*100:.2f}%",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    # Exit message
    cv2.putText(
        display,
        "Press Q to Exit",
        (25, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow("Object Classification using Teachable Machine", display)

    key = cv2.waitKey(1)

    if key == ord("q") or key == ord("Q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()