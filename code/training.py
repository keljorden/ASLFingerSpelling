import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from greedyDecoder import greedy_decode
from CER import decode_targets, calculate_cer

def train_and_validate(model, train_loader, val_loader, tokenizer, max_epochs=100, learning_rate=1e-3, accumulation_steps=4, device='cuda', save_path="best_model.pth"):
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    early_stopping_patience = 10
    epochs_without_improvement = 0
    best_val_cer = float('inf')
    
    print(f"Starting training on {device}...")
    
    for epoch in range(max_epochs):
        
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        optimizer.zero_grad()
        
        for batch_idx, (frames, targets, f_lens, t_lens) in enumerate(train_bar):
            frames, targets = frames.to(device), targets.to(device)
            f_lens, t_lens = f_lens.to(device), t_lens.to(device)
            
            predictions = model(frames)
            
            log_probs = F.log_softmax(predictions, dim=2).permute(1, 0, 2)
            loss = criterion(log_probs, targets, f_lens, t_lens)
            
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                optimizer.zero_grad()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_cer = 0.0
        total_val_samples = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        with torch.inference_mode():
            for frames, targets, f_lens, t_lens in val_bar:
                frames, targets = frames.to(device), targets.to(device)
                f_lens, t_lens = f_lens.to(device), t_lens.to(device)
                
                predictions = model(frames)
                log_probs = F.log_softmax(predictions, dim=2).permute(1, 0, 2)
                loss = criterion(log_probs, targets, f_lens, t_lens)
                val_loss += loss.item()
                
                pred_strings = greedy_decode(predictions, tokenizer)
                target_strings = decode_targets(targets, t_lens, tokenizer)
                
                batch_cer = 0.0
                for ref, hyp in zip(target_strings, pred_strings):
                    cer = calculate_cer(ref, hyp)
                    batch_cer += cer
                        
                val_cer += batch_cer
                total_val_samples += len(targets)
                
                val_bar.set_postfix({'loss': f"{loss.item():.4f}", 'cer': f"{(batch_cer/len(targets)):.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_cer = val_cer / total_val_samples
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val CER: {avg_val_cer:.4f} | LR: {current_lr:.6f}")
        
        scheduler.step(avg_val_cer)
        
        if avg_val_cer < best_val_cer:
            best_val_cer = avg_val_cer
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_cer": best_val_cer,
                "epochs_without_improvement": epochs_without_improvement,
            }, save_path)
            print("  -> Best model saved!")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epoch(s).")
            
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n[!] Validation CER plateaued for {early_stopping_patience} epochs. Stopping training early.")
            break