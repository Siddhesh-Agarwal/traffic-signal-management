import time

import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor


def get_objects(image: Image.Image):
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

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        item = model.config.id2label[label.item()]
        confidence = round(score.item() * 100, 3)
        coords = tuple(map(int, box.tolist()))
        if item in vehicle_labels and isinstance(item, str):
            print(f"Detected a [{item:<10}] with confidence {confidence}% at {coords}")


if __name__ == "__main__":
    print("[INFO] Script Started")

    print("Loading image...")
    image_path = "./assets/pexels-freestockpro-1031698.jpg"
    image = Image.open(image_path, mode="r").convert("RGB")
    print("Image loaded")

    print("Getting objects...")
    t1 = time.perf_counter_ns()
    get_objects(image)
    t2 = time.perf_counter_ns()
    print(f"Time taken: {(t2 - t1)/1e9} s")

    print("Script Finished")
