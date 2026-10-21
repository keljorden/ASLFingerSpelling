import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

from preprocess_fswild import MediaPipeHandCropper
from modelMaker import ResNetBiGRU
from datasetMaker import ASLVulcanTokenizer, PadToSquare
from greedyDecoder import greedy_decode

def predict_video(video_path, model_path, device='cuda'):

    tokenizer = ASLVulcanTokenizer()
    cropper = MediaPipeHandCropper(model_asset_path='hand_landmarker.task', padding=20)
    
    transform = transforms.Compose([
        PadToSquare(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model = ResNetBiGRU(num_classes=27, hidden_size=256, num_layers=2, dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    cap = cv2.VideoCapture(video_path)
    processed_frames = []
    
    print("Extracting and cropping frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        cropped_pil = cropper.crop(temp_path)
        
        if cropped_pil is not None:
            tensor_frame = transform(cropped_pil)
            processed_frames.append(tensor_frame)
            
    cap.release()
    if Path("temp_frame.jpg").exists():
        Path("temp_frame.jpg").unlink()
        
    if not processed_frames:
        return "Error: No hands detected in video."
        
    frames_tensor = torch.stack(processed_frames).unsqueeze(0).to(device)
    
    print("Running sequence through the model...")
    with torch.inference_mode():
        predictions = model(frames_tensor)
        
        decoded_text = greedy_decode(predictions, tokenizer)[0]
        
    return decoded_text

if __name__ == "__main__":
    VIDEO_FILE = "my_test_spelling.mp4"
    MODEL_WEIGHTS = Path.cwd().parent / "modelWeights" / "gru_shock_model.pth"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    result = predict_video(VIDEO_FILE, MODEL_WEIGHTS, device)
    print(f"\nPredicted Sign Language Text: '{result}'")