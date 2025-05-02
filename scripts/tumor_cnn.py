# tumor_classifier.py - Modified for binary tumor classification with proper filename parsing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import re
from PIL import Image
import argparse
from sklearn.metrics import roc_curve, auc


class TumorDataset(Dataset):
    def __init__(self, data_dir, file_pattern, csv_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.csv_file = csv_file
        
        # Get all matching files
        self.file_paths = []
        
        # Handle file pattern correctly
        if '*' in file_pattern:
            # If pattern already has a wildcard, use it directly
            self.file_paths.extend(glob.glob(os.path.join(data_dir, f"{file_pattern}")))
        else:
            # Add wildcards around the pattern if needed
            self.file_paths.extend(glob.glob(os.path.join(data_dir, f"*{file_pattern}*")))
            
            # Handle .npy files if no extension in pattern
            if '.' not in file_pattern:
                # Try with common image extensions
                image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy']
                for ext in image_extensions:
                    self.file_paths.extend(glob.glob(os.path.join(data_dir, f"*{file_pattern}*{ext}")))
        
        print(f"Found {len(self.file_paths)} files matching the pattern {file_pattern}")
        
        # Load CSV for labels
        self.df = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.file_paths)  # Changed from self.image_paths to self.file_paths
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]  # Changed from self.image_paths to self.file_paths
        
        # Get exam and slice numbers from filename using regex
        filename = os.path.basename(file_path)
        
        # Use regex to extract exam and slice numbers
        exam_match = re.search(r'exam(\d+)', filename)
        slice_match = re.search(r'slice(\d+)', filename)
        
        if exam_match and slice_match:
            exam_num = int(exam_match.group(1))
            slice_num = int(slice_match.group(1))
        else:
            print(f"Error parsing filename: {filename}. Could not extract exam/slice numbers.")
            exam_num = 0
            slice_num = 0
        
        # Load file based on extension
        if file_path.endswith('.npy'):
            # Load NumPy array
            image_array = np.load(file_path)
            
            # Normalize the array to 0-255 range if needed
            if image_array.max() > 1 and image_array.max() <= 255:
                # Already in 0-255 range
                pass
            elif image_array.max() <= 1.0:
                # Scale from 0-1 to 0-255
                image_array = (image_array * 255).astype(np.uint8)
            else:
                # Scale arbitrary range to 0-255
                image_array = ((image_array - image_array.min()) / 
                            (image_array.max() - image_array.min()) * 255).astype(np.uint8)
            
            # Convert to PIL Image
            if len(image_array.shape) == 2:  # Grayscale
                image = Image.fromarray(image_array, mode='L')
                # Convert to RGB (3 channels)
                image = image.convert('RGB')
            elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB
                image = Image.fromarray(image_array)
            else:
                raise ValueError(f"Unsupported array shape: {image_array.shape}")
        else:
            # Load image file
            image = Image.open(file_path)  # Changed from img_path to file_path
            
            # Convert to RGB if grayscale
            if image.mode != 'RGB':
                image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get label from CSV
        row = self.df[(self.df['exam'] == exam_num) & (self.df['slice'] == slice_num)]
        
        has_tumor = False
        if not row.empty:
            row = row.iloc[0]
            
            # Check if region has tumor (either breast)
            if 'right_has_malignant' in row and row['right_has_malignant'] > 0:
                has_tumor = True
            if 'left_has_malignant' in row and row['left_has_malignant'] > 0:
                has_tumor = True
        
        # Create binary label
        label = 1 if has_tumor else 0
        
        return image, label


class TumorClassifier(nn.Module):
    def __init__(self, dropout_rate=0.3):
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
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 1)

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
        
        # Adaptive pooling to handle various input sizes
        x = self.adaptive_pool(x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x.squeeze()


def train_epoch(net, trainloader, optimizer, criterion, device):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # For binary classification
        predicted = (outputs > 0.0).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 20 == 19:
            print(f'Batch {i+1}, Loss: {running_loss/20:.3f}, Acc: {100*correct/total:.2f}%')
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
            images, labels = images.to(device), labels.to(device).float()
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            predicted = (outputs > 0.0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return val_loss / len(valloader), 100 * correct / total


def test(net, testloader, device):
    net.eval()
    correct = 0
    total = 0
    
    # For confusion matrix
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device).float()
            
            outputs = net(images)
            predicted = (outputs > 0.0).float()
            
            # For ROC curve we need probabilities
            probs = torch.sigmoid(outputs)
            
            # Update confusion matrix
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
            
            # Overall accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for ROC curve
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(probs.cpu().numpy())
    
    # Print overall accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f'True Positives: {true_positives}')
    print(f'False Positives: {false_positives}')
    print(f'True Negatives: {true_negatives}')
    print(f'False Negatives: {false_negatives}')
    
    # Calculate sensitivity and specificity
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    print(f'Sensitivity (TPR): {100 * sensitivity:.2f}%')
    print(f'Specificity (TNR): {100 * specificity:.2f}%')
    
    return accuracy, np.array(all_labels), np.array(all_predictions)


def main():
    # Hardcoded arguments for debugging
    data_dir = "/mnt/d/work/datasets/breast_models_repository/processed_set_v2/output/masked_images/"
    file_pattern = "_masked_magnitude_gray.png"
    # file_pattern = "*_1.00*"
    csv_file = "/mnt/d/work/datasets/breast_models_repository/exam_analysis_results.csv"
    batch_size = 22
    num_epochs = 40
    learning_rate = 0.00011
    image_size = 224
    
    # Configuration
    weight_decay = 1e-5  # L2 regularization
    
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data augmentation for training - adjusted for medical images
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.1, 0.1, 0.1), (0.2, 0.2, 0.2))
    ])
    
    # No augmentation for validation and test
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1, 0.1, 0.1), (0.2, 0.2, 0.2))
    ])
    
    # Create datasets
    full_dataset = TumorDataset(data_dir, file_pattern, csv_file, transform=transform_train)
    
    # Count class distribution
    label_counts = {0: 0, 1: 0}
    for _, label in full_dataset:
        label_counts[label] += 1
    
    print(f"Class distribution - Not Malignant: {label_counts[0]}, Malignant: {label_counts[1]}")
    
    # Set class weights if needed
    class_weight = None
    if label_counts[1] > 0:
        # Calculate based on inverse class frequency
        class_weight = label_counts[0] / label_counts[1]
        print(f"Automatically calculated class weight for malignant: {class_weight:.2f}")
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    trainset, valset, testset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator)
    
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Check dataset sizes
    print(f"Training set size: {len(trainset)}")
    print(f"Validation set size: {len(valset)}")
    print(f"Test set size: {len(testset)}")
    
    # Create the network
    net = TumorClassifier()
    net.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    if class_weight is not None:
        pos_weight = torch.tensor([class_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Lists to store metrics
    train_acc_history = []
    val_acc_history = []
    val_loss_history = []
    
    # Training loop
    best_val_acc = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = 7  # Early stopping patience
    
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
            torch.save(net.state_dict(), './tumor_best_acc_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            epochs_without_improvement = 0
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), './tumor_best_loss_model.pth')
            print(f'New best model saved with validation loss: {val_loss:.4f}')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs without improvement')
            break
    
    print('Finished Training')
    
    # Load best model based on accuracy
    net.load_state_dict(torch.load('./tumor_best_acc_model.pth'))
    
    # Test the model
    test_acc, all_labels, all_predictions = test(net, testloader, device)
    
    print(f'Final test accuracy: {test_acc:.2f}%')
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy vs. Epoch')
    
    plt.subplot(1, 3, 2)
    plt.plot(val_loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss vs. Epoch')
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.subplot(1, 3, 3)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()


if __name__ == "__main__":
    main()