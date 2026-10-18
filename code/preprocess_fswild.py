import os
import cv2
import pandas as pd
import mediapipe as mp
from PIL import Image
from tqdm import tqdm

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeHandCropper:
    def __init__(self, model_path="hand_landmarker.task", padding=20):
        self.padding = padding

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5
        )

        self.hands_detector = vision.HandLandmarker.create_from_options(
            options
        )

    def crop(self, image_path):
        image = cv2.imread(image_path)

        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        results = self.hands_detector.detect(mp_image)

        if results.hand_landmarks:
            h, w, _ = image.shape
            
            all_x_coords = []
            all_y_coords = []

            for hand_landmarks in results.hand_landmarks:
                all_x_coords.extend([landmark.x * w for landmark in hand_landmarks])
                all_y_coords.extend([landmark.y * h for landmark in hand_landmarks])

            xmin = max(0, int(min(all_x_coords)) - self.padding)
            ymin = max(0, int(min(all_y_coords)) - self.padding)
            xmax = min(w, int(max(all_x_coords)) + self.padding)
            ymax = min(h, int(max(all_y_coords)) + self.padding)

            if xmax > xmin and ymax > ymin:
                cropped_array = image_rgb[ymin:ymax, xmin:xmax]
                return Image.fromarray(cropped_array)

        return Image.fromarray(image_rgb)

    def close(self):
        self.hands_detector.close()


def process_dataset(csv_path, source_dir, target_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(target_dir, exist_ok=True)

    sequences = df["filename"].dropna().unique()
    print(f"Processing {len(sequences)} sequences...")

    cropper = MediaPipeHandCropper(
        model_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "hand_landmarker.task"
        ),
        padding=20
    )

    try:
        for seq in tqdm(sequences):
            seq_source_path = os.path.join(source_dir, str(seq))
            seq_target_path = os.path.join(target_dir, str(seq))

            if not os.path.isdir(seq_source_path):
                continue

            os.makedirs(seq_target_path, exist_ok=True)

            frame_files = sorted(
                f for f in os.listdir(seq_source_path)
                if f.lower().endswith(".jpg")
            )

            for filename in frame_files:
                source_img_path = os.path.join(
                    seq_source_path, filename
                )
                target_img_path = os.path.join(
                    seq_target_path, filename
                )

                if os.path.exists(target_img_path):
                    continue

                img_pil = cropper.crop(source_img_path)

                if img_pil is not None:
                    img_pil.save(target_img_path)

    finally:
        cropper.close()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    CSV_FILE = os.path.join(BASE_DIR, "ChicagoFSWild.csv")
    SOURCE_DIR = os.path.join(BASE_DIR, "ChicagoFSWild-Frames")
    TARGET_DIR = os.path.join(BASE_DIR, "ChicagoFSWild-Cropped")

    print("CSV exists:", os.path.isfile(CSV_FILE), CSV_FILE)
    print("Source exists:", os.path.isdir(SOURCE_DIR), SOURCE_DIR)
    print("Target:", TARGET_DIR)

    if not os.path.isfile(CSV_FILE):
        raise FileNotFoundError(f"CSV not found: {CSV_FILE}")

    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(f"Frames directory not found: {SOURCE_DIR}")

    process_dataset(CSV_FILE, SOURCE_DIR, TARGET_DIR)
    print("Cropping complete!")