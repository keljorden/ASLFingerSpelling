import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

class ASLVulcanTokenizer:
    def __init__(self):
        self.chars = "abcdefghijklmnopqrstuvwxyz"
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.chars)}
        self.blank_idx = 0
        
    def encode(self, text):
        text = str(text).lower().replace(" ", "") 
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

class ChicagoFSWildDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, tokenizer, partition='train', transform=None):
        
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['partition'] == partition].reset_index(drop=True)
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder_name = row['filename']
        num_frames = row['number_of_frames']
        text_label = row['label_proc']
        
        frames = []
        sequence_path = os.path.join(self.root_dir, folder_name)
        
        if os.path.exists(sequence_path):
            frame_files = sorted(os.listdir(sequence_path))
            for f in frame_files:
                img_path = os.path.join(sequence_path, f)
                try:
                    img = Image.open(img_path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    frames.append(img)
                except Exception as e:
                    continue
                    
        if len(frames) == 0:
            frames = [torch.zeros(3, 224, 224) for _ in range(num_frames)]
            
        frames_tensor = torch.stack(frames) 
        
        target = self.tokenizer.encode(text_label)
        target_tensor = torch.tensor(target, dtype=torch.long)
        
        return frames_tensor, target_tensor

if __name__ == "__main__":

    root = r"C:\Users\offsh\Documents\project\ChicagoFSWild\ChicagoFSWild\ChicagoFSWild-Cropped"
    f = "ChicagoFSWild.csv"
    
    tokenizer = ASLVulcanTokenizer()
    
    dataset = ChicagoFSWildDataset(csv_file=f, root_dir=root, tokenizer=tokenizer)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        frames, tag = dataset[0]
        print(f"Frames shape: {frames.shape}")
        print(f"Tag (encoded label): {tag}")