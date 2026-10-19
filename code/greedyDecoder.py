import torch

def greedy_decode(predictions, tokenizer):
    predicted_indices = torch.argmax(predictions, dim=-1)
    
    decoded_strings = []
    
    for i in range(predicted_indices.size(0)):
        sequence = predicted_indices[i].tolist()
        
        collapsed_sequence = []
        previous_token = -1
        
        for token in sequence:
            if token != previous_token:
                if token != tokenizer.blank_idx:
                    collapsed_sequence.append(token)
            
            previous_token = token
            
        decoded_text = "".join([tokenizer.idx_to_char[idx] for idx in collapsed_sequence])
        decoded_strings.append(decoded_text)
        
    return decoded_strings