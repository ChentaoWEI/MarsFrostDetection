from data_prepare import prepare_data
from dataloader import CustomDataset, transform
from torch.utils.data import DataLoader
# from model.cnn_model import CustomModel as CNN
# from model.VGG16 import initialize_vgg16 as VGG16
#from model.ResNet50 import initialize_resnet50 as ResNet50
from model.EfficientNetB0 import initialize_efficientnetb0 as EfficientNetB0
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# Load the data
print("Data preparation...")
img_list_train, label_list_train, img_list_test, label_list_test, img_list_val, label_list_val = prepare_data()

print("Data loading...")
train_dataset = CustomDataset(img_list_train, label_list_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = CustomDataset(img_list_test, label_list_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
val_dataset = CustomDataset(img_list_val, label_list_val, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

print("Model loading...")
model = EfficientNetB0(2)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)  # L2 regularization
criterion = nn.CrossEntropyLoss()

# Define the lists to store the training and validation loss
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 5
trigger_times = 0

num_epochs = 20

# Run the training loop
print("Training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    batch_num = 0
    for images, labels in train_loader:
        batch_num += 1
        print(f"Training batch {batch_num}...")
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader: 
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Append the training and validation loss to the lists
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'saved_model/EfficientNetB0_best_model.pth')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break

# Plot the training and validation loss
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('saved_plot/EfficientNetB0_loss.png')
plt.show()







