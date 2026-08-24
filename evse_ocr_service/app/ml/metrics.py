from typing import List, Dict, Any

def levenshtein_distance(seq1: str, seq2: str) -> int:
    """
    Calculates the Levenshtein edit distance (Substitutions, Deletions, Insertions) between two sequences.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1]   # Substitution
                )
                
    return dp[m][n]

def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """
    Calculates Character Error Rate: CER = (S + D + I) / N
    """
    if not predictions or not references:
        return 0.0
        
    total_distance = 0
    total_reference_length = 0
    
    for pred, ref in zip(predictions, references):
        total_distance += levenshtein_distance(pred, ref)
        total_reference_length += len(ref)
        
    if total_reference_length == 0:
        return 0.0
        
    return total_distance / total_reference_length

def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    Calculates Word Error Rate (treating delimiter-separated tokens as words).
    """
    if not predictions or not references:
        return 0.0
        
    total_distance = 0
    total_words = 0
    
    for pred, ref in zip(predictions, references):
        pred_words = pred.replace("*", " ").replace("-", " ").split()
        ref_words = ref.replace("*", " ").replace("-", " ").split()
        
        m, n = len(pred_words), len(ref_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_words[i - 1] == ref_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                    
        total_distance += dp[m][n]
        total_words += len(ref_words)
        
    if total_words == 0:
        return 0.0
    return total_distance / total_words

def evaluate_predictions(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics: CER, WER, and Exact Match Accuracy.
    """
    exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    total = len(predictions) if predictions else 1
    
    return {
        "exact_match_accuracy": round(exact_matches / total, 4),
        "cer": round(calculate_cer(predictions, references), 4),
        "wer": round(calculate_wer(predictions, references), 4),
        "total_samples": len(predictions)
    }
