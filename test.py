import torch
import torch.nn as nn
from models import model as md
import os
from torchtext import data
from torchtext import datasets
import solver
from torchtext.vocab import Vectors


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    TEXT = data.Field()
    LABEL = data.Field(sequential=False, dtype=torch.long)

    train, val, test = datasets.SST.splits(
        TEXT, LABEL, fine_grained=True, train_subtrees=False
    )

    LABEL.build_vocab(train)

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=8, device=device
    )
    TEXT.build_vocab(train, vectors=Vectors(name="vector.txt", cache="./data"))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 768
    OUTPUT_DIM = 5
    for file in os.listdir("./trained_models"):
        print(file)
        checkpoint = torch.load("./trained_models/" + file)
        print(checkpoint)
        for k in checkpoint:
            print(k)
        model = md.model(file[:-9])(
            INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM
        ).load_state_dict(checkpoint)
        model = model.to(device)
        # Test
        test_loss, test_acc = solver.evaluate(model, test_iter, criterion)
        print(
            file[:-9] + f" Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%"
        )


if __name__ == "__main__":
    main()
