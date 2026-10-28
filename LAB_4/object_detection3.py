
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import cv2

# Load the pretrained VGG16 model
model = VGG16(weights="imagenet", include_top=True)

# Generate region proposals using Selective Search
def get_region_proposals(img, max_regions=200):
    try:
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        boxes = [
            (x, y, w, h)
            for x, y, w, h in ss.process()
            if w > 40 and h > 40
        ]
        if boxes:
            return boxes[:max_regions]
    except Exception:
        pass

    # Fallback: sliding windows
    H, W = img.shape[:2]
    boxes = []

    for scale in (0.6, 0.4, 0.25):
        bw, bh = int(W * scale), int(H * scale)

        for y in range(0, H - bh + 1, max(1, bh // 2)):
            for x in range(0, W - bw + 1, max(1, bw // 2)):
                boxes.append((x, y, bw, bh))

    return boxes[:max_regions]


# Remove duplicate overlapping boxes
def non_max_suppression(dets, iou_thr=0.3):
    if not dets:
        return []

    boxes = np.array([
        [x, y, x + w, y + h]
        for (x, y, w, h), _, _ in dets
    ], dtype=float)

    scores = np.array([score for _, _, score in dets])
    order = scores.argsort()[::-1]
    keep = []

    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]

        if len(rest) == 0:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (
            boxes[i, 3] - boxes[i, 1]
        )
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (
            boxes[rest, 3] - boxes[rest, 1]
        )

        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou < iou_thr]

    return [dets[i] for i in keep]


# Load the image from the LAB_4 folder
image_path = "image.jpg"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(
        f"Could not find {image_path}. Keep it in the same folder as this script."
    )

# Resize the image
H0, W0 = img.shape[:2]
img = cv2.resize(img, (600, int(H0 * 600.0 / W0)))

# Generate region proposals
boxes = get_region_proposals(img, max_regions=200)
print("Region proposals:", len(boxes))

# Prepare image crops for VGG16
crops = np.array([
    cv2.resize(img[y:y+h, x:x+w], (224, 224))
    for x, y, w, h in boxes
], dtype=np.float32)

# Classify all regions in batches
preds = model.predict(
    preprocess_input(crops),
    batch_size=32,
    verbose=0
)

# Keep high-confidence predictions
detections = []

for (x, y, w, h), p in zip(boxes, preds):
    _, label, prob = decode_predictions(p[None], top=1)[0][0]

    if prob > 0.50:
        detections.append(((x, y, w, h), label, float(prob)))

# Remove duplicate detections
detections = non_max_suppression(detections, iou_thr=0.3)

# Draw the final detections
for (x, y, w, h), label, score in detections:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"{label}: {score:.2f}",
        (x, max(18, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# Save and display the result
cv2.imwrite("detected_output2.jpg", img)
print("Detected objects:", len(detections))
print("Output saved as detected_output2.jpg")

cv2.imshow("VGG16 Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
