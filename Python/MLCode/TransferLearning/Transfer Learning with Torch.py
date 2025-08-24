#%% 引入模組
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch import nn, optim
#%% Transfer Learning
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data_dir = 'C:/研究所/自學/各模型/CNN圖檔/xray_dataset_covid19_prac/pic'
dataset = datasets.ImageFolder(
    data_dir, transform = data_transforms)

dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(
    val_dataset, batch_size = 32, shuffle = False)
test_loader = DataLoader(
    test_dataset, batch_size = 32, shuffle = False)

model = models.resnet18(pretrained = True)
model.fc = nn.Linear(model.fc.in_features, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.fc.parameters(), lr = 0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    print(f'Epoch {epoch+1} / {num_epochs}')
    print(f' Train Loss : {train_loss:.4f}, Val Loss : {val_loss:.4f}')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy : {100 * correct / total :.2f}%')
