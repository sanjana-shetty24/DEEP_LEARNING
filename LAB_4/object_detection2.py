
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
import os

# 1. Load the pre-trained object detection model
MODEL_URL = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
print("Loading object detection model...")
detector = hub.load(MODEL_URL)
print("Model loaded successfully!")

# 2. COCO class names (80 objects)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "TV",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


# 3. Detect objects and draw bounding boxes
def detect_objects_with_boxes(image_path, detector):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not open image: {image_path}. "
            "Check that the image exists in the project folder."
        )

    orig_image = image.copy()

    # Convert image to TensorFlow tensor
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run object detection
    detections = detector(input_tensor)

    # Extract detection results
    num_detections = int(detections["num_detections"].numpy()[0])
    detection_boxes = detections["detection_boxes"].numpy()[0]
    detection_classes = (
        detections["detection_classes"].numpy()[0].astype(np.int32)
    )
    detection_scores = detections["detection_scores"].numpy()[0]

    # Draw boxes for detections above 50% confidence
    confidence_threshold = 0.5
    detected_count = 0

    for i in range(num_detections):
        score = detection_scores[i]

        if score > confidence_threshold:
            box = detection_boxes[i]
            h, w, _ = orig_image.shape

            y_min, x_min, y_max, x_max = box

            x_min = int(x_min * w)
            x_max = int(x_max * w)
            y_min = int(y_min * h)
            y_max = int(y_max * h)

            class_id = detection_classes[i]
            if 1 <= class_id <= len(COCO_CLASSES):
                class_name = COCO_CLASSES[class_id - 1]
            else:
                class_name = "Unknown"

            label = f"{class_name}: {score:.2f}"

            cv2.rectangle(
                orig_image,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            cv2.putText(
                orig_image,
                label,
                (x_min, max(y_min - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            detected_count += 1
            print(f"Detected: {label}")

    print(f"\nDetection complete! Objects detected: {detected_count}")
    return orig_image


# 4. Run object detection using image.jpg
if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(__file__), "image.jpg")

    result_image = detect_objects_with_boxes(image_path, detector)

    # Save the output image in the same folder
    output_path = os.path.join(
        os.path.dirname(__file__), "detected_output.jpg"
    )
    cv2.imwrite(output_path, result_image)

    print(f"Result saved to: {output_path}")

    # Display the result
    cv2.imshow("Object Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
