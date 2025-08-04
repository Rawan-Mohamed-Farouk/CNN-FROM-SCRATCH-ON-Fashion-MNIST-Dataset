import torch, torch.nn as nn, torch.optim as optim, torchvision, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import FashionCNN
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128)

model = FashionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

def train():
    for epoch in range(1, 11):
        model.train(); loss_total = 0; correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            correct += (out.argmax(1) == y).sum().item()
        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch}: Loss={loss_total:.4f}, Acc={acc:.2%}")
        test()
        scheduler.step(loss_total)

def test():
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
    acc = correct / len(test_loader.dataset)
    print(f"✅ Test Accuracy: {acc:.2%}")
    torch.save(model.state_dict(), "best_model.pth")

train()
