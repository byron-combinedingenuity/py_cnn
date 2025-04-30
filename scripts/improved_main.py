# Modified main.py
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

def load_data(transform_train, transform_test, batch_size):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Create validation set
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    return trainset, valset, testset

def create_data_loaders(trainset, valset, testset, batch_size):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
    return classes, trainloader, valloader, testloader

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class ImprovedNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

def train_epoch(net, trainloader, optimizer, criterion, device):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 200 == 199:
            print(f'Batch {i+1}, Loss: {running_loss/200:.3f}, Acc: {100*correct/total:.2f}%')
            running_loss = 0.0
            
    return 100 * correct / total

def validate(net, valloader, criterion, device):
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return val_loss / len(valloader), 100 * correct / total

def test(net, testloader, classes, device):
    net.eval()
    correct = 0
    total = 0
    
    # Prepare for per-class accuracy
    class_correct = {classname: 0 for classname in classes}
    class_total = {classname: 0 for classname in classes}
    
    # Confusion matrix
    confusion_matrix = torch.zeros(len(classes), len(classes))
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            
            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
            # Overall accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[classes[label]] += 1
                class_total[classes[label]] += 1
    
    # Print overall accuracy
    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')
    
    # Print per-class accuracy
    for classname in classes:
        if class_total[classname] > 0:
            accuracy = 100 * class_correct[classname] / class_total[classname]
            print(f'Accuracy for class {classname:5s}: {accuracy:.1f}%')
    
    return 100 * correct / total, confusion_matrix

def main():
    # Configuration
    batch_size = 128  # Increased from 4
    num_epochs = 20   # Increased from 2
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 1e-4  # L2 regularization
    
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # No augmentation for validation and test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load data and create data loaders
    trainset, valset, testset = load_data(transform_train, transform_test, batch_size)
    classes, trainloader, valloader, testloader = create_data_loaders(trainset, valset, testset, batch_size)
    
    # Create the network
    net = ImprovedNet()
    net.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
    
    # Lists to store metrics
    train_acc_history = []
    val_acc_history = []
    val_loss_history = []
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Train
        train_acc = train_epoch(net, trainloader, optimizer, criterion, device)
        train_acc_history.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(net, valloader, criterion, device)
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)
        
        print(f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}')
        
        # Learning rate adjustment
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), './cifar_best_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
    
    print('Finished Training')
    
    # Load best model
    net.load_state_dict(torch.load('./cifar_best_model.pth'))
    
    # Test the model
    test_acc, confusion_matrix = test(net, testloader, classes, device)
    
    print(f'Final test accuracy: {test_acc:.2f}%')
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy vs. Epoch')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss vs. Epoch')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    main()