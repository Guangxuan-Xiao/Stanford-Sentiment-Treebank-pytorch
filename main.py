import argparse
import torch
import torch.nn as nn
from models import model as md
import torch.optim as optim
import time
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors  # , GloVe, CharNGram, FastText
import solver


def main():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("--load", help="Load the saved model")
    parser.add_argument("--save-path", default="model.pkl", help="Path to save model")
    parser.add_argument("--model", default="naive_lstm", help="Choose the model")
    parser.add_argument(
        "--max-epoch", type=int, default=200, help="Restrict max epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log interval when training in batches",
    )
    parser.add_argument(
        "--save-interval", type=int, default=1, help="Model save interval when training"
    )
    parser.add_argument(
        "--optim", default="sgd", help="Optimizer. Choose between adam and sgd"
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ################################
    # DataLoader
    ################################

    # set up fields
    TEXT = data.Field()
    LABEL = data.Field(sequential=False, dtype=torch.long)

    # make splits for data
    # DO NOT MODIFY: fine_grained=True, train_subtrees=False
    train, val, test = datasets.SST.splits(
        TEXT, LABEL, fine_grained=True, train_subtrees=False
    )
    # build the vocabulary
    # you can use other pretrained vectors, refer to https://github.com/pytorch/text/blob/master/torchtext/vocab.py
    TEXT.build_vocab(train, vectors=Vectors(name="vector.txt", cache="./data"))
    LABEL.build_vocab(train)
    # We can also see the vocabulary directly using either the stoi (string to int) or itos (int to string) method.
    # make iterator for splits
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=args.batch_size, device=device
    )
    # print batch information

    batch = next(iter(train_iter))  # for batch in train_iter

    # Copy the pre-trained word embeddings we loaded earlier into the embedding layer of our model.
    pretrained_embeddings = TEXT.vocab.vectors
    print("pretrained_embeddings.shape: ", pretrained_embeddings.shape)
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = pretrained_embeddings.shape[1]
    HIDDEN_DIM = 768
    OUTPUT_DIM = 5

    model = md.model(args.model)(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    # you should maintain a nn.embedding layer in your network
    model.embedding.weight.data.copy_(pretrained_embeddings)
    print(f"The model has {count_parameters(model):,} trainable parameters")

    # Training
    opitmizer = optim.SGD(model.parameters(), 0.1)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    num_of_epochs = args.max_epoch
    best_valid_loss = float("inf")

    for epoch in range(num_of_epochs):
        start_time = time.time()
        train_loss, train_acc = solver.train_one_epoch(
            model, train_iter, opitmizer, criterion
        )
        valid_loss, valid_acc = solver.evaluate(model, val_iter, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = solver.epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "tut1-model.pt")

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

    # Test
    test_loss, test_acc = solver.evaluate(model, test_iter, criterion)
    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
