# ASL Fingerspelling Recognition (CSLR) on ChicagoFSWild

This repository contains a PyTorch-based Continuous Sign Language Recognition (CSLR) pipeline designed to translate unaligned sequences of American Sign Language (ASL) fingerspelling frames into text. 

The project is specifically engineered to train efficiently on a **6GB VRAM GPU** using a 5,000-sample dataset, implementing offline MediaPipe hand cropping, custom jitter-free video augmentations, and a two-phase "shock treatment" training strategy.

## Key Features

* **ResNet18 + BiGRU Architecture**: A hybrid CNN-RNN model that extracts spatial features via a pretrained ResNet18 and models temporal dependencies using a 2-layer Bidirectional GRU.
* **CTC Loss & Greedy Decoding**: Handles unaligned sequence-to-sequence translation where the input frame count and the output string length vary.
* **6GB VRAM Optimization**: Utilizes Gradient Accumulation (simulating a batch size of 16 with a physical batch size of 2-4) to prevent CUDA Out-of-Memory errors during sequence modeling.
* **Offline MediaPipe Preprocessing**: Uses the `mediapipe.tasks.vision` API to pre-crop bounding boxes around the signing hand, eliminating temporal jitter and drastically speeding up DataLoader bottlenecks.
* **Jitter-Free Video Augmentations**: Custom PyTorch transforms (`SpatialVideoAugmentation`, `TemporalFrameDrop`) that apply consistent spatial transformations across the entire sequence to preserve optical flow and simulate varying signing speeds.
* **Two-Phase Training**: 
  * *Phase 1:* Frozen backbone with `ReduceLROnPlateau` tracking Character Error Rate (CER).
  * *Phase 2 (Shock Treatment):* Partial unfreezing of deep semantic layers with `CosineAnnealingWarmRestarts` to shatter performance plateaus.

## Dataset

This project uses the **ChicagoFSWild** dataset. 
* Place your raw extracted frames into a directory named `FSWild`.
* **Inputs:** Variable-length sequences of `.jpg` frames.
* **Targets:** Variable-length text strings (`label_proc`), which include spaces and special characters.

## Installation & Requirements

```bash
pip install torch torchvision
pip install pandas numpy opencv-python mediapipe Pillow tqdm
```
*Note: A GPU supporting CUDA is highly recommended.*

## Model Architecture

1. **Spatial Backbone**: `ResNet18` (ImageNet pretrained). Early layers (`conv1`, `layer1`, `layer2`) are frozen to prevent overfitting on the small dataset.
2. **Feature Compression**: A `Linear` + `BatchNorm1d` bottleneck to compress 512-dim ResNet features down to 256-dim.
3. **Temporal Module**: A 2-layer `BiGRU` (Hidden Size: 256). *Note: BiGRU was empirically found to outperform BiLSTM for this dataset size.*
4. **Classification Head**: Maps the RNN output to the vocabulary size (26 letters + space/special characters + CTC Blank Token).

## Pipeline Usage

### 1. Offline Preprocessing (Hand Cropping)
To prevent the CPU from bottlenecking the GPU during training, the dataset is pre-processed to isolate the signing hands using MediaPipe.
You must download the MediaPipe hand landmarker model first:
```bash
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
Then run the extraction script on your `FSWild` folder:
```bash
python preprocess_fswild.py
```
*This generates a `FSWild-Cropped` directory mirroring the original dataset structure.*

### 2. Training Phase 1 (Foundation)
Trains the BiGRU and the projection layers while keeping the CNN backbone frozen. Uses AdamW and tracks Character Error Rate (CER).
```bash
python training.py
```

### 3. Training Phase 2 (Shock Treatment)
Once Phase 1 plateaus (typically around ~0.50 CER), this script unfreezes ResNet `layer3` and `layer4` and applies a Cosine Annealing scheduler to kick the model out of its local minimum.
```bash
python shockTraining.py
```

### 4. Inference (Live Video Translation)
Test the trained model on a brand-new `.mp4` video. The script handles the frame extraction, MediaPipe cropping, tensor padding, and Greedy CTC Decoding automatically.
```bash
python inference.py --video "my_test_spelling.mp4" --weights "modelWeights/gru_shock_best_model.pth"
```

## Evaluation Metric

The model is evaluated using **CER (Character Error Rate)** based on the Levenshtein distance between the predicted text and the ground truth string, preventing division by zero and accurately reflecting the unaligned translation accuracy.
