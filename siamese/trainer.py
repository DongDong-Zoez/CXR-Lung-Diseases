import torch
from evaluator import evaluator

def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    train_loss = .0
    predictions, ground_truths = [], []
    for before_images, after_images, labels in train_loader:
        before_images = before_images.to(device=device, dtype=torch.float)
        after_images = after_images.to(device=device, dtype=torch.float)
        labels = labels.to(device=device, dtype=torch.long)
        
        optimizer.zero_grad()
        logits = model(before_images, after_images)
        loss = criterion(logits, labels)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        
        predictions.append(preds)
        ground_truths.append(labels)
        
    train_loss /= len(train_loader)
    
    predictions = torch.cat(predictions)
    ground_truths = torch.cat(ground_truths)
    train_acc, train_f1 = evaluator(predictions, ground_truths)
    
    return train_loss, 100*train_acc, 100*train_f1
        
def validation(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = .0
    predictions, ground_truths = [], []
    with torch.no_grad():
        for before_images, after_images, labels in valid_loader:
            before_images = before_images.to(device=device, dtype=torch.float)
            after_images = after_images.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)

            logits = model(before_images, after_images)
            loss = criterion(logits, labels)
            
            valid_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
        
            predictions.append(preds)
            ground_truths.append(labels)
        
    valid_loss /= len(valid_loader)
    
    predictions = torch.cat(predictions)
    ground_truths = torch.cat(ground_truths)
    valid_acc, valid_f1 = evaluator(predictions, ground_truths)
    return valid_loss, 100*valid_acc, 100*valid_f1
    