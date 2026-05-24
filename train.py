import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import HotnessDataset
from tqdm import tqdm


def main():
    # Setup CUDA device for faster training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard transformations for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Load datasets from the split folders
    train_dataset = HotnessDataset(csv_file="dataset/train.csv", img_dir="dataset/train", transform=transform)
    val_dataset = HotnessDataset(csv_file="dataset/val.csv", img_dir="dataset/val", transform=transform)

    # 2. Create DataLoaders (Change batch_size to 16 or 8 if CUDA out of memory occurs)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Initialize pre-trained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze base layers to keep pre-trained features
    for param in model.parameters():
        param.requires_grad = False

    # Modify final layer for Regression (1 output node instead of classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)

    # 4. Define Loss and Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    num_epochs = 40

    # 5. Main Training Loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 15)

        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc="Training Phase", leave=False)
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.view(-1, 1).to(device)  # Reshape labels for MSELoss

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_dataset)
        print(f'Train L1 Loss: {train_loss:.4f}')

        # --- VALIDATION PHASE ---
        model.eval()
        running_loss = 0.0

        val_bar = tqdm(val_loader, desc="Validation Phase", leave=False)
        with torch.no_grad():  # Disable gradient calculation for validation
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.view(-1, 1).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                val_bar.set_postfix(loss=loss.item())

        val_loss = running_loss / len(val_dataset)
        print(f'Val L1 Loss: {val_loss:.4f}')

    # 6. Save the trained weights
    torch.save(model.state_dict(), 'hotness_model.pth')
    print("\nModel successfully saved to hotness_model.pth")


if __name__ == '__main__':
    main()