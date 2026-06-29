import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_data():
    x = np.random.randn(10000, 50).astype(np.float32)

    # Generate labels that depend on the inputs
    weights = np.random.randn(50)
    logits = x @ weights
    y = (logits > 0).astype(np.int64)

    x = torch.tensor(x)
    y = torch.tensor(y)

    dataset = TensorDataset(x, y)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size]
    )

    return train_dataset, test_dataset


class SuperNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = get_data()

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64
    )

    model = SuperNet().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    epochs = 20

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        accuracy = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch + 1:2d} "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {accuracy:.4f}"
        )


if __name__ == "__main__":
    train_model()
