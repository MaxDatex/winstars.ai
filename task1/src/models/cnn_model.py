import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..interface import MnistClassifierInterface


class CNNArchitecture(nn.Module):
    def __init__(self):
        super(CNNArchitecture, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class CNN(MnistClassifierInterface):
    def __init__(
            self,
            epochs: int = 10,
            batch_size: int = 32,
            learning_rate: float = 0.001,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CNNArchitecture().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _prepare_data(self, x, y=None):
        if hasattr(x, 'values'):
            x = x.values
        x = x.reshape(-1, 1, 28, 28)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        if y is not None:
            if hasattr(y, 'values'):
                y = y.values
            y = torch.tensor(y, dtype=torch.long).to(self.device)
            return TensorDataset(x, y)
        return x

    def train(self, x_train, y_train):
        dataset = self._prepare_data(x_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(f"Training | Device: {self.device}")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            training_loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for x_batch, y_batch in training_loop:
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                training_loop.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / len(loader)
            print(f'Epoch {epoch + 1}, loss: {avg_loss:.4f}')

    def predict(self, x_test):
        x = self._prepare_data(x_test)
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            prediction = torch.argmax(output, dim=1)
        return prediction.cpu().numpy()
