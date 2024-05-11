from collections import defaultdict

import cv2
import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor


def get_objects(image: Image.Image) -> defaultdict[str, int]:
    # you can specify the revision tag if you don't want the timm dependency
    model = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(model, revision="no_timm")
    model = DetrForObjectDetection.from_pretrained(model, revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]

    vehicle_labels = ["car", "bus", "truck", "motorcycle"]
    freq: defaultdict[str, int] = defaultdict(int)
    for score, label in zip(results["scores"], results["labels"]):
        item = model.config.id2label[label.item()]
        confidence = round(score.item() * 100, 3)
        if (
            isinstance(item, str)
            and isinstance(confidence, float)
            and item in vehicle_labels
        ):
            # print(f"Detected a [{item:<10}] with confidence {confidence}%")
            freq[item] += 1
    return freq


# load the video
video = cv2.VideoCapture(
    "./assets/2103099-uhd_3840_2160_30fps.mp4", apiPreference=cv2.CAP_FFMPEG
)
fps = video.get(cv2.CAP_PROP_FPS)
interval = fps * 5  # 5 seconds

frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    # load the image
    image = Image.fromarray(frame)
    if frame_count % interval == 0:
        print(f"Frame: {frame_count}")
        freq = get_objects(image)
        for item, count in freq.items():
            print(f"-> {item}: {count}")
        print()
    frame_count += 1
