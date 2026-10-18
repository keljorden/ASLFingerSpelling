import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
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

def pad_collate(batch):
    frames_list, targets_list = zip(*batch)
    
    frames_padded = pad_sequence(frames_list, batch_first=True)
    
    targets_padded = pad_sequence(targets_list, batch_first=True, padding_value=0)
    
    frame_lengths = torch.tensor([len(f) for f in frames_list], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets_list], dtype=torch.long)
    
    return frames_padded, targets_padded, frame_lengths, target_lengths

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

    batch_size = 4
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=pad_collate
    )
        
    print("\nTesting DataLoader...")
    for batch_idx, (frames_batch, targets_batch, f_lens, t_lens) in enumerate(dataloader):
        print(f"Batch Frames Shape: {frames_batch.shape}")
        print(f"Batch Targets Shape: {targets_batch.shape}")
        print(f"Frame Lengths: {f_lens}")
        print(f"Target Lengths: {t_lens}")
        break