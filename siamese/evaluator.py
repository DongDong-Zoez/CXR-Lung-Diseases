import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluator(preds, gts):
    
    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    gts = gts.cpu().numpy() if isinstance(gts, torch.Tensor) else gts
    
    acc = accuracy_score(preds, gts)
    f1 = f1_score(preds, gts, average="macro")
    
    return acc, f1