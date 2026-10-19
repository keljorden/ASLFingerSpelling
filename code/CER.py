def levenshtein_distance(ref, hyp):
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,
                               dp[i][j - 1] + 1,
                               dp[i - 1][j - 1] + 1)
    return dp[m][n]

def calculate_cer(reference, hypothesis):
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0
    return levenshtein_distance(reference, hypothesis) / len(reference)

def decode_targets(targets, t_lens, tokenizer):
    decoded_strings = []
    for i in range(targets.size(0)):
        actual_length = t_lens[i].item()
        sequence = targets[i][:actual_length].tolist()
        
        text = "".join([tokenizer.idx_to_char[idx] for idx in sequence if idx != tokenizer.blank_idx])
        decoded_strings.append(text)
    return decoded_strings