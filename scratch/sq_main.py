import os
import cv2
from matplotlib import pyplot as plt

# Input image
img = cv2.imread("image.jpg")
if img is None:
    raise FileNotFoundError("image.jpg not found. Place an input image in the repo root.")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb_or = img_rgb.copy()

# Try YOLO (Ultralytics). If unavailable, fall back to CascadeClassifier.
detections = []  # list of (x, y, w, h)
used_detector = None

try:
    # Ultralytics YOLO (pip install ultralytics). Uses a local or auto-downloaded weight.
    from ultralytics import YOLO  # type: ignore

    # You can replace 'yolov8n.pt' with a local path to a custom model.
    model_path = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")
    model = YOLO(model_path)

    # Run inference
    results = model.predict(source=img, conf=0.25, verbose=False)
    if results and len(results) > 0:
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None:
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append((x1, y1, x2 - x1, y2 - y1))
    used_detector = "YOLOv8"
except Exception as e:
    # Could be ImportError or weights download failure. We'll fall back to cascade.
    # print(f"YOLO not used due to: {e}")
    pass

if not detections:
    # Fallback to Haar cascade if YOLO unavailable or no detections
    cascade_path = "stop_data.xml"
    if os.path.exists(cascade_path):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stop_cascade = cv2.CascadeClassifier(cascade_path)
        found = stop_cascade.detectMultiScale(img_gray, minSize=(20, 20))
        for (x, y, w, h) in found:
            detections.append((x, y, w, h))
        used_detector = "Cascade"
    else:
        used_detector = "None"

# Draw detections
img_out = img_rgb.copy()
for (x, y, w, h) in detections:
    cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_rgb_or)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(img_out)
axes[1].set_title(f"Detected ({used_detector})")
axes[1].axis("off")

print(f"Detections: {len(detections)} using {used_detector}")

plt.tight_layout()
plt.show()

