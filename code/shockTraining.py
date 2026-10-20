import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from greedyDecoder import greedy_decode
from CER import decode_targets, calculate_cer

def shock_treatment_train(model, train_loader, val_loader, tokenizer, best_model_path, save_path, accumulation_steps=4, device='cuda'):
    print("Loading best weights for Shock Treatment...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    for param in model.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5, 
        T_mult=1, 
        eta_min=1e-6
    )
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    best_val_cer = float('inf')
    
    max_epochs = 15
    
    print(f"Starting Phase 2 Shock Training on {device}...")
    
    for epoch in range(max_epochs):
        
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Shock Epoch {epoch+1} [Train]")
        
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
        val_cer = 0.0
        total_val_samples = 0
        
        val_bar = tqdm(val_loader, desc=f"Shock Epoch {epoch+1} [Val]")
        
        with torch.inference_mode():
            for frames, targets, f_lens, t_lens in val_bar:
                frames, targets = frames.to(device), targets.to(device)
                f_lens, t_lens = f_lens.to(device), t_lens.to(device)
                
                predictions = model(frames)
                
                pred_strings = greedy_decode(predictions, tokenizer)
                target_strings = decode_targets(targets, t_lens, tokenizer)
                
                batch_cer = sum(calculate_cer(ref, hyp) for ref, hyp in zip(target_strings, pred_strings))
                val_cer += batch_cer
                total_val_samples += len(targets)
                
                val_bar.set_postfix({'cer': f"{(batch_cer/len(targets)):.4f}"})
                
        avg_val_cer = val_cer / total_val_samples
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val CER: {avg_val_cer:.4f} | LR: {current_lr:.6f}")
        
        scheduler.step()
        
        if avg_val_cer < best_val_cer:
            best_val_cer = avg_val_cer
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_cer": best_val_cer
            }, save_path)
            print("  -> New Best Shock Model saved!")