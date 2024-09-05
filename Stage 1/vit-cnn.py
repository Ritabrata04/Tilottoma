import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import timm

class SmartBinSegregationModel(nn.Module):
    """
    Smart Bin Segregation Model where we use Vision Transformer (ViT) as feature extractor and a lightweight CNN for real-time classification.
    This is ammortised for real time. We draw inspiration from the SAM paper that using global image encoder with a local image encoder for classification
    provides better results.
    """
    def __init__(self):
        super(SmartBinSegregationModel, self).__init__()

        # Pretrained Vision Transformer (ViT)
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()  # Remove the classification head, use only feature extractor

        # Lightweight CNN for real-time classification
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Binary classification (menstrual waste vs. non-menstrual waste)
        )

    def forward(self, x):
        # Extract features using ViT
        x = self.vit(x)  # Output from ViT is (batch_size, 768)

        # Convert 1D ViT output to 2D for CNN (batch_size, channels, height, width)
        x = x.view(x.size(0), 768, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # Resize for CNN

        # Perform classification using lightweight CNN
        x = self.cnn(x)
        return x
    # Define transformations for data augmentation and preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = CustomDataset(root='data/train', transform=transform)
val_dataset = CustomDataset(root='data/val', transform=transform)

# Load data into DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model, loss function, and optimizer
model = SmartBinSegregationModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):  # Number of epochs
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

    # Validation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total}%')


