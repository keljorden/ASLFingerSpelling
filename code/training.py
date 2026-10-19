import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def train_and_validate(model, train_loader, val_loader, max_epochs=100, learning_rate=1e-3, device='cuda', save_path = "best_model.pth"):
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=True
    )
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    early_stopping_patience = 10
    epochs_without_improvement = 0
    best_val_loss = float('inf')
    
    print(f"Starting training on {device}...")
    
    for epoch in range(max_epochs):
        
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for frames, targets, f_lens, t_lens in train_bar:
            frames, targets = frames.to(device), targets.to(device)
            f_lens, t_lens = f_lens.to(device), t_lens.to(device)
            
            optimizer.zero_grad()
            predictions = model(frames)
            
            log_probs = F.log_softmax(predictions, dim=2).permute(1, 0, 2)
            loss = criterion(log_probs, targets, f_lens, t_lens)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        with torch.inference_mode():
            for frames, targets, f_lens, t_lens in val_bar:
                frames, targets = frames.to(device), targets.to(device)
                f_lens, t_lens = f_lens.to(device), t_lens.to(device)
                
                predictions = model(frames)
                log_probs = F.log_softmax(predictions, dim=2).permute(1, 0, 2)
                loss = criterion(log_probs, targets, f_lens, t_lens)
                
                val_loss += loss.item()
                val_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print("  -> Best model saved!")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epoch(s).")
            
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n[!] Validation loss plateaued for {early_stopping_patience} epochs. Stopping training early to prevent overfitting.")
            break
