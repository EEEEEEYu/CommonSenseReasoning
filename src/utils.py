import re
import random
from typing import List, Set

def generate_banlist(event_description: str) -> List[str]:
    """
    Generates a list of banned words from the event description.
    Simple strategy: Split by space, filter stopwords (mocked for now), and return.
    In a real scenario, this would use NLTK or similar for lemmatization.
    """
    stopwords = {"a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "is", "was", "are", "were"}
    words = re.findall(r'\b\w+\b', event_description.lower())
    banlist = [w for w in words if w not in stopwords and len(w) > 2]
    return list(set(banlist))

def check_banlist(text: str, banlist: List[str]) -> bool:
    """
    Checks if any word from the banlist is present in the text.
    Returns True if valid (no banned words), False otherwise.
    """
    text_lower = text.lower()
    for word in banlist:
        # Simple word boundary check
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            return False
    return True

def calculate_set_atom_metrics(gold: dict, predicted: dict) -> dict:
    """
    Calculates Precision, Recall, F1, and Containment for semantic fields.
    Treats values as sets of words/tokens or exact string matches.
    Here we do simple exact string match for simplicity.
    """
    metrics = {}
    total_tokens = 0
    match_tokens = 0
    
    # Simple containment score: 
    # Check if gold values are essentially in predicted values
    # For a real robust metric, we'd need overlap of tokens.
    
    fields = ["agent", "predicate", "patient", "recipient", "location", "time", "instrument"]
    correct_fields = 0
    total_fields = 0
    
    for field in fields:
        g_val = gold.get(field)
        p_val = predicted.get(field)
        
        if g_val:
            total_fields += 1
            if p_val and (g_val.lower() in p_val.lower() or p_val.lower() in g_val.lower()):
                 correct_fields += 1
                 
    metrics["field_accuracy"] = correct_fields / total_fields if total_fields > 0 else 0.0
    
    return metrics
