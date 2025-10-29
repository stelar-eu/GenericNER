import pandas as pd

def evaluate_ner(df, gold_standard, predictions):
    """
    Evaluates the performance of a named entity recognition (NER) model by comparing predicted entities with gold standard entities.

    Args:
        df (pd.DataFrame): DataFrame containing the gold standard and predicted entities
        gold_standard (str): Column name in DataFrame containing gold standard entities
        predictions (str): Column name in DataFrame containing predicted entities
    
    Returns:
        dict: Dictionary containing precision, recall, F1 score, and counts of entities
    """
    # Initialize metrics
    total_gold_standard = 0
    total_predicted = 0
    total_correct = 0
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a DataFrame")
    
    # Iterate through the DataFrame
    for index, row in df.iterrows():
        gold_entities = row[gold_standard] if isinstance(row[gold_standard], list) else []
        pred_entities = row[predictions] if isinstance(row[predictions], list) else []
        
        # Count entities
        total_gold_standard += len(gold_entities)
        total_predicted += len(pred_entities)
        
        # Count correct predictions (exact match)
        correct = len(set(gold_entities) & set(pred_entities))
        total_correct += correct
    
    # Calculate metrics
    precision = total_correct / total_predicted if total_predicted > 0 else 0
    recall = total_correct / total_gold_standard if total_gold_standard > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_gold_standard": total_gold_standard,
        "total_predicted": total_predicted,
        "total_correct": total_correct
    }
    
    
def evaluate_ner_list(df, gold_standard, predictions):
    """
    Evaluates the performance of a named entity recognition (NER) model by comparing predicted entities with gold standard entities.

    Args:
        df (pd.DataFrame): DataFrame containing the gold standard and predicted entities
        gold_standard (str): Column name in DataFrame containing gold standard entities (list of entities)
        predictions (str): Column name in DataFrame containing predicted entities (list of entities)
    
    Returns:
        dict: Dictionary containing precision, recall, F1 score, and counts of entities
    """
    # Initialize metrics
    total_gold_standard = 0
    total_predicted = 0
    total_correct = 0
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a DataFrame")
    
    if gold_standard not in df.columns:
        raise ValueError(f"Column '{gold_standard}' not found in DataFrame")
    
    if predictions not in df.columns:
        raise ValueError(f"Column '{predictions}' not found in DataFrame")
    
    # Iterate through the DataFrame
    for index, row in df.iterrows():
        # Ensure we have lists for both gold standard and predictions
        gold_entities = row[gold_standard] if isinstance(row[gold_standard], list) else []
        pred_entities = row[predictions] if isinstance(row[predictions], list) else []
        
        # Count entities for this row
        total_gold_standard += len(gold_entities)
        total_predicted += len(pred_entities)
        
        # Convert to sets for intersection calculation (handles duplicates)
        gold_set = set(gold_entities)
        pred_set = set(pred_entities)
        
        # Count correct predictions (exact match)
        correct = len(gold_set & pred_set)
        total_correct += correct
    
    # Calculate metrics
    precision = total_correct / total_predicted if total_predicted > 0 else 0
    recall = total_correct / total_gold_standard if total_gold_standard > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_gold_standard": total_gold_standard,
        "total_predicted": total_predicted,
        "total_correct": total_correct
    }
    