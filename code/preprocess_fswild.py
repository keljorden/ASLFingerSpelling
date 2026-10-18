import os
import cv2
import pandas as pd
import mediapipe as mp
from PIL import Image
from tqdm import tqdm
import shutil

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

        self.hands_detector = vision.HandLandmarker.create_from_options(options)

    def get_hand_coordinates(self, image_path):
       
        image = cv2.imread(image_path)
        if image is None:
            return [], [], None, None

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        results = self.hands_detector.detect(mp_image)
        
        frame_x_coords = []
        frame_y_coords = []

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                frame_x_coords.extend([landmark.x * w for landmark in hand_landmarks])
                frame_y_coords.extend([landmark.y * h for landmark in hand_landmarks])
                
        return frame_x_coords, frame_y_coords, w, h

    def apply_crop_and_save(self, image_path, target_path, bbox):
        
        image = cv2.imread(image_path)
        if image is None:
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        xmin, ymin, xmax, ymax = bbox
        
        cropped_array = image_rgb[ymin:ymax, xmin:xmax]
        
        img_pil = Image.fromarray(cropped_array)
        img_pil.save(target_path)

    def close(self):
        self.hands_detector.close()


def process_dataset(csv_path, source_dir, target_dir, padding=20):
    df = pd.read_csv(csv_path)
    os.makedirs(target_dir, exist_ok=True)

    sequences = df["filename"].dropna().unique()
    print(f"Processing {len(sequences)} sequences using Global Sequence Cropping...")

    cropper = MediaPipeHandCropper(
        model_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "hand_landmarker.task"
        ),
        padding=padding
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
            
            if not frame_files:
                continue

            first_target_img = os.path.join(seq_target_path, frame_files[0])
            if os.path.exists(first_target_img):
                continue

            seq_all_x = []
            seq_all_y = []
            img_w, img_h = 0, 0

            for filename in frame_files:
                source_img_path = os.path.join(seq_source_path, filename)
                x_coords, y_coords, w, h = cropper.get_hand_coordinates(source_img_path)
                
                seq_all_x.extend(x_coords)
                seq_all_y.extend(y_coords)
                
                if w is not None and img_w == 0:
                    img_w, img_h = w, h

            if not seq_all_x or not seq_all_y:
                for filename in frame_files:
                    shutil.copy2(
                        os.path.join(seq_source_path, filename),
                        os.path.join(seq_target_path, filename)
                    )
                continue

            xmin = max(0, int(min(seq_all_x)) - padding)
            ymin = max(0, int(min(seq_all_y)) - padding)
            xmax = min(img_w, int(max(seq_all_x)) + padding)
            ymax = min(img_h, int(max(seq_all_y)) + padding)
            
            if xmax <= xmin or ymax <= ymin:
                continue
                
            global_bbox = (xmin, ymin, xmax, ymax)

            for filename in frame_files:
                source_img_path = os.path.join(seq_source_path, filename)
                target_img_path = os.path.join(seq_target_path, filename)
                
                cropper.apply_crop_and_save(source_img_path, target_img_path, global_bbox)

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

    process_dataset(CSV_FILE, SOURCE_DIR, TARGET_DIR, padding=30)
    print("Cropping complete")