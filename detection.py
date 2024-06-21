import os
import cv2
import shutil
from PIL import Image, ImageDraw
from collections import defaultdict
import yolov5

yolo_model = yolov5.load("yolov5s.pt")


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def detect_objects(model, image, threshold=0.5, min_area=10000):
    results = model(image, size=640)
    preds = results.pred[0]
    boxes = preds[:, :4].cpu().numpy()
    scores = preds[:, 4].cpu().numpy()
    labels = preds[:, 5].cpu().numpy().astype(int)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    valid_indices = (scores >= threshold) & (areas >= min_area)
    return labels[valid_indices], boxes[valid_indices], scores[valid_indices]


def save_detected_objects(image, boxes, output_dir):
    for i, box in enumerate(boxes):
        cropped_image = image.crop((box[0], box[1], box[2], box[3]))
        cropped_image_path = os.path.join(output_dir, f"object_{i+1}.jpg")
        cropped_image.save(cropped_image_path)
        print(f"Saved: {cropped_image_path}")


def draw_boxes(image, boxes, labels, scores):
    draw = ImageDraw.Draw(image)
    class_names = model.names # type: ignore
    for box, label, score in zip(boxes, labels, scores):
        draw.rectangle([(box[0], box[1]), (box[2], box[3])],
                       outline="red", width=3)
        draw.text((box[0], box[1]),
                  f"{class_names[label]}: {score:.2f}", fill="red")
    return image


def extract_frames(video_path, output_folder="frames"):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    success = True

    while success:
        success, frame = video_capture.read()
        if success:
            if frame_count % int(fps) == 0:
                frame_path = os.path.join(
                    output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
            frame_count += 1

    video_capture.release()


def main(video_path):
    frames_folder = "frames"
    extract_frames(video_path, frames_folder)

    for frame_file in os.listdir(frames_folder):
        frame_path = os.path.join(frames_folder, frame_file)
        image = load_image(frame_path)

        labels, boxes, scores = detect_objects(yolo_model, image)

        output_dir = os.path.join(
            "detected_objects", os.path.splitext(frame_file)[0])
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        save_detected_objects(image, boxes, output_dir)

        class_names = yolo_model.names
        object_counts = defaultdict(int)
        for label in labels:
            object_counts[class_names[label]] += 1

        for object_name, count in object_counts.items():
            print(f"{object_name} - {count}")
