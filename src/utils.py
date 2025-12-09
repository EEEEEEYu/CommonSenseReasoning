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
    Calculates metrics for semantic fields where each field is a list of strings.
    We gather all atoms (words/phrases) from the gold lists and predicted lists.
    A match is counted if a predicted atom is present in the gold list for that field.
    """
    metrics = {}
    
    fields = ["agent", "predicate", "patient", "recipient", "location", "time", "instrument"]
    
    # Global counts for micro-average
    total_gold_atoms = 0
    total_pred_atoms = 0
    total_correct_atoms = 0
    
    # Per-field metrics
    for field in fields:
        g_list = gold.get(field, [])
        p_list = predicted.get(field, [])
        
        # Normalize: convert to lowercase, strip to ensure robust comparison
        g_set = set(str(x).lower().strip() for x in g_list)
        p_set = set(str(x).lower().strip() for x in p_list)
        
        # Calculate overlap
        # Relaxation: Check if predicted item *contains* or *is contained by* any gold item
        # But user asked: "assume that any extracted object/events that are in the full gold json set ... is a hit"
        # We will check exact match (normalized) against the set for now to be robust. 
        # Or should we do substring? "mismatch" usually implies string distance. 
        # Let's do exact match on the set members.
        
        correct = 0
        for p_item in p_set:
            if p_item in g_set:
                correct += 1
            else:
                 # Optional: substring match fallback?
                 # User said: "match gold json extract all objects ... any extracted ... in the full gold json set ... is a hit"
                 # Let's stick to simple set membership for "in".
                 pass

        total_gold_atoms += len(g_set)
        total_pred_atoms += len(p_set)
        total_correct_atoms += correct
        
        # Field specific recall/precision
        prec = correct / len(p_set) if len(p_set) > 0 else 0.0
        rec = correct / len(g_set) if len(g_set) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        metrics[f"{field}_precision"] = prec
        metrics[f"{field}_recall"] = rec
        metrics[f"{field}_f1"] = f1

    # Micro-averages
    micro_prec = total_correct_atoms / total_pred_atoms if total_pred_atoms > 0 else 0.0
    micro_rec = total_correct_atoms / total_gold_atoms if total_gold_atoms > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    
    metrics["micro_precision"] = micro_prec
    metrics["micro_recall"] = micro_rec
    metrics["micro_f1"] = micro_f1
    
    return metrics
