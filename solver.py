import torch


# Evaluate how many parameters there are in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_accuracy(preds, gt):
    correct = (preds.argmax(dim=1) == gt).float()
    return correct.sum() / len(correct)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_one_epoch(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        label = batch.label.sub(1)
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, label)
        acc = calc_accuracy(predictions, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            label = batch.label.sub(1)
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, label)
            acc = calc_accuracy(predictions, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
