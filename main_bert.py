from transformers import BertTokenizer
from torchtext import datasets
from torchtext import data
import torch
import argparse
from models import bert
import solver
import torch.nn as nn
import torch.optim as optim
import time
import pickle
import plot


def save_result(result, name):
    with open("./results/" + name + ".pkl", "wb") as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[: max_input_length - 2]
    return tokens


parser = argparse.ArgumentParser(description="SST")
parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
parser.add_argument("--max-epoch", type=int, default=200, help="Restrict max epochs")
args = parser.parse_args()


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_input_length = tokenizer.max_model_input_sizes["bert-base-uncased"]

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEXT = data.Field(
    batch_first=True,
    use_vocab=False,
    tokenize=tokenize_and_cut,
    preprocessing=tokenizer.convert_tokens_to_ids,
    init_token=init_token_idx,
    eos_token=eos_token_idx,
    pad_token=pad_token_idx,
    unk_token=unk_token_idx,
)

LABEL = data.Field(sequential=False, dtype=torch.long)

train, val, test = datasets.SST.splits(
    TEXT, LABEL, fine_grained=True, train_subtrees=False
)
LABEL.build_vocab(train)
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_size=args.batch_size, device=device
)

HIDDEN_DIM = 768
OUTPUT_DIM = 5
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = bert.BERTGRUSentiment(HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
# Freeze prameters in bert
for name, param in model.named_parameters():
    if name.startswith("bert"):
        param.requires_grad = False

print(f"The model has {solver.count_parameters(model):,} trainable parameters")

# Training
opitmizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

num_of_epochs = args.max_epoch
best_valid_loss = float("inf")
train_loss_list, train_acc_list = [], []
val_loss_list, val_acc_list = [], []
for epoch in range(num_of_epochs):
    start_time = time.time()
    train_loss, train_acc = solver.train_one_epoch(
        model, train_iter, opitmizer, criterion
    )
    valid_loss, valid_acc = solver.evaluate(model, val_iter, criterion)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(valid_loss)
    val_acc_list.append(valid_acc)
    end_time = time.time()
    epoch_mins, epoch_secs = solver.epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "./trained_models/" + "BERT" + "-model.pt")

    print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

# Test
test_loss, test_acc = solver.evaluate(model, test_iter, criterion)
print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")

# Save Result
result_dict = {
    "BERT" + " Training": [train_loss_list, train_acc_list],
    "BERT" + " Validating": [val_loss_list, val_acc_list],
}

save_result(result_dict, "BERT" + "_result")
plot.plot_loss_and_acc_save(result_dict, "./plots/" + "BERT")
