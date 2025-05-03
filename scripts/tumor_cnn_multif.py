# improved_tumor_cnn.py - Enhanced for multi-frequency tumor classification
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import re
from PIL import Image
import argparse
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from collections import defaultdict


class MultiFrequencyTumorDataset(Dataset):
    """Dataset class that handles multiple frequency patterns and combines them into multi-channel inputs."""
    
    def __init__(self, data_dir, file_patterns, csv_file, transform=None, min_patterns_required=None):
        """
        Initialize the multi-frequency dataset.
        
        Args:
            data_dir (str): Directory containing the image files
            file_patterns (list): List of file patterns for different frequencies
            csv_file (str): Path to the CSV file with labels
            transform: Optional transformations to apply to images
            min_patterns_required (int): Minimum number of patterns required for a valid sample
                                         If None, all patterns are required
        """
        self.data_dir = data_dir
        self.file_patterns = file_patterns
        self.transform = transform
        self.csv_file = csv_file
        
        # Set minimum patterns required (default to all if not specified)
        self.min_patterns_required = min_patterns_required if min_patterns_required is not None else len(file_patterns)
        
        # Ensure min_patterns_required is valid
        if self.min_patterns_required <= 0 or self.min_patterns_required > len(file_patterns):
            raise ValueError(f"min_patterns_required must be between 1 and {len(file_patterns)}")
        
        # Load CSV for labels
        self.df = pd.read_csv(csv_file)
        
        # Create a dictionary to store all files grouped by exam and slice numbers
        self.exam_slice_files = defaultdict(dict)
        
        # Process each file pattern
        for pattern_idx, pattern in enumerate(file_patterns):
            # Get all matching files for this pattern
            matching_files = []
            
            # Handle file pattern correctly
            if '*' in pattern:
                # If pattern already has a wildcard, use it directly
                matching_files.extend(glob.glob(os.path.join(data_dir, f"{pattern}")))
            else:
                # Add wildcards around the pattern
                matching_files.extend(glob.glob(os.path.join(data_dir, f"*{pattern}*")))
                
                # Handle .npy files if no extension in pattern
                if '.' not in pattern:
                    # Try with common image extensions
                    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy']
                    for ext in image_extensions:
                        matching_files.extend(glob.glob(os.path.join(data_dir, f"*{pattern}*{ext}")))
            
            print(f"Found {len(matching_files)} files matching pattern: {pattern}")
            
            # Process each file to extract exam and slice numbers
            for file_path in matching_files:
                filename = os.path.basename(file_path)
                
                # Use regex to extract exam and slice numbers
                exam_match = re.search(r'exam(\d+)', filename)
                slice_match = re.search(r'slice(\d+)', filename)
                
                if exam_match and slice_match:
                    exam_num = int(exam_match.group(1))
                    slice_num = int(slice_match.group(1))
                    
                    # Store the file path in the dictionary
                    # Each exam-slice pair will have file paths for different patterns
                    if pattern_idx not in self.exam_slice_files[(exam_num, slice_num)]:
                        self.exam_slice_files[(exam_num, slice_num)][pattern_idx] = file_path
        
        # MODIFIED: Keep exam-slice pairs that have at least min_patterns_required patterns
        self.valid_exam_slices = [
            key for key, value in self.exam_slice_files.items() 
            if len(value) >= self.min_patterns_required
        ]
        
        print(f"Found {len(self.valid_exam_slices)} exam-slice pairs with at least {self.min_patterns_required} of {len(file_patterns)} frequency patterns")
        
    def __len__(self):
        return len(self.valid_exam_slices)
    
    def __getitem__(self, idx):
        exam_num, slice_num = self.valid_exam_slices[idx]
        
        # List to store loaded images for all patterns
        multi_freq_images = []
        
        # Load each frequency image
        for pattern_idx in range(len(self.file_patterns)):
            # Skip patterns that don't exist for this exam-slice pair
            if pattern_idx not in self.exam_slice_files[(exam_num, slice_num)]:
                continue
                
            file_path = self.exam_slice_files[(exam_num, slice_num)][pattern_idx]
            
            # Load image based on file extension
            if file_path.endswith('.npy'):
                # Load NumPy array
                image_array = np.load(file_path)
                
                # Check if the array contains complex numbers
                if np.iscomplexobj(image_array):
                    # For complex data, extract real and imaginary parts
                    real_part = np.real(image_array)
                    imag_part = np.imag(image_array)
                    
                    # Normalize real part
                    if real_part.max() != real_part.min():
                        real_part = ((real_part - real_part.min()) / 
                                    (real_part.max() - real_part.min()) * 255).astype(np.uint8)
                    else:
                        real_part = np.zeros_like(real_part, dtype=np.uint8)
                    
                    # Normalize imaginary part
                    if imag_part.max() != imag_part.min():
                        imag_part = ((imag_part - imag_part.min()) / 
                                    (imag_part.max() - imag_part.min()) * 255).astype(np.uint8)
                    else:
                        imag_part = np.zeros_like(imag_part, dtype=np.uint8)
                    
                    # Create PIL images for both components
                    real_image = Image.fromarray(real_part, mode='L')
                    imag_image = Image.fromarray(imag_part, mode='L')
                    
                    # Apply transform if specified
                    if self.transform:
                        real_image = self.transform(real_image)
                        imag_image = self.transform(imag_image)
                    
                    # Add both real and imaginary components
                    multi_freq_images.append(real_image)
                    multi_freq_images.append(imag_image)
                    
                    # Print info about complex data handling
                    if pattern_idx == 0 and idx < 2:  # Only print for first few samples
                        print(f"Complex data detected in {os.path.basename(file_path)}. "
                            f"Using both real and imaginary components.")
                else:
                    # For non-complex data, handle as before
                    # Normalize the array to 0-255 range if needed
                    if image_array.max() > 1 and image_array.max() <= 255:
                        # Already in 0-255 range
                        pass
                    elif image_array.max() <= 1.0:
                        # Scale from 0-1 to 0-255
                        image_array = (image_array * 255).astype(np.uint8)
                    else:
                        # Scale arbitrary range to 0-255
                        if image_array.max() != image_array.min():
                            image_array = ((image_array - image_array.min()) / 
                                        (image_array.max() - image_array.min()) * 255).astype(np.uint8)
                        else:
                            image_array = np.zeros_like(image_array, dtype=np.uint8)
                    
                    # Convert to PIL Image
                    if len(image_array.shape) == 2:  # Grayscale
                        image = Image.fromarray(image_array, mode='L')
                    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB
                        image = Image.fromarray(image_array)
                    else:
                        raise ValueError(f"Unsupported array shape: {image_array.shape}")
                    
                    # Apply transform if specified
                    if self.transform:
                        image = self.transform(image)
                    
                    # Add the image to our list
                    multi_freq_images.append(image)
            else:
                # Load image file
                image = Image.open(file_path)
                
                # Ensure image is grayscale (single channel) for multi-channel stacking
                if image.mode == 'RGB':
                    # Convert to grayscale using luminosity method
                    image = image.convert('L')
                
                # Apply transform if specified
                if self.transform:
                    image = self.transform(image)
                
                # Add the image to our list
                multi_freq_images.append(image)
        
        # Stack multiple frequencies as channels
        if torch.is_tensor(multi_freq_images[0]):
            # If transform was applied and returned tensors, stack them as channels
            multi_channel_image = torch.cat([img.unsqueeze(0) if img.dim() == 2 else img for img in multi_freq_images], dim=0)
        else:
            # If we still have PIL images, convert to tensors and stack
            multi_channel_image = torch.cat([transforms.ToTensor()(img) for img in multi_freq_images], dim=0)
        
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
        
        return multi_channel_image, label, (exam_num, slice_num)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance more effectively."""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Binary case
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ImprovedTumorClassifier(nn.Module):
    """Improved CNN architecture optimized for complex RF imaging data."""
    
    def __init__(self, in_channels=4, dropout_rate=0.4):
        super(ImprovedTumorClassifier, self).__init__()
        
        # Input channels adjusted for multi-frequency data
        self.in_channels = in_channels
        
        # First convolutional block - larger kernels for initial RF feature extraction
        # Larger kernel captures more spatial context from RF images
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third convolutional block - increased channel depth
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fourth convolutional block - deeper feature extraction
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Add residual connections for stable training
        self.use_residual = True
        self.res_conv1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.res_conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.res_conv3 = nn.Conv2d(64, 128, kernel_size=1)
        self.res_conv4 = nn.Conv2d(128, 256, kernel_size=1)
        
        # Global pooling with multiple pooling types
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Attention mechanism for RF feature weighting
        self.channel_attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )
        
        # Fully connected layers with multi-stage dropout
        self.fc1 = nn.Linear(256 * 2, 256)  # Doubled for avg+max pooling
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(dropout_rate * 1.5)  # Higher dropout for deeper layers
        
        self.fc3 = nn.Linear(128, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)
        self.dropout_fc3 = nn.Dropout(dropout_rate * 1.5)
        
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        # First block with residual connection
        identity1 = self.res_conv1(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.use_residual:
            x = x + identity1
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block with residual connection
        identity2 = self.res_conv2(x)
        identity2 = F.avg_pool2d(identity2, 2)  # Match spatial size after pooling
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        if self.use_residual:
            x = x + identity2
        x = self.dropout2(x)
        
        # Third block with residual connection  
        identity3 = self.res_conv3(x)
        identity3 = F.avg_pool2d(identity3, 2)  # Match spatial size after pooling
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        if self.use_residual:
            x = x + identity3
        x = self.dropout3(x)
        
        # Fourth block with residual connection
        identity4 = self.res_conv4(x)
        identity4 = F.avg_pool2d(identity4, 2)  # Match spatial size after pooling
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        if self.use_residual:
            x = x + identity4
        x = self.dropout4(x)
        
        # Parallel global pooling (captures different statistics)
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        
        # Apply channel attention to avg_pool
        avg_pool_flat = avg_pool.view(avg_pool.size(0), -1)
        channel_weights = self.channel_attention(avg_pool_flat).unsqueeze(-1).unsqueeze(-1)
        weighted_avg_pool = avg_pool * channel_weights
        
        # Combine pools
        avg_pool_flat = weighted_avg_pool.view(avg_pool.size(0), -1)
        max_pool_flat = max_pool.view(max_pool.size(0), -1)
        
        # Concatenate the pooled features
        pooled = torch.cat([avg_pool_flat, max_pool_flat], dim=1)
        
        # Multi-stage fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(pooled)))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout_fc3(x)
        
        x = self.fc4(x)
        return x.squeeze()


def train_epoch(net, trainloader, optimizer, criterion, device):
    """Training function for one epoch with improved logging."""
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels, _ = data
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
        
        # Update confusion matrix metrics
        true_positives += ((predicted == 1) & (labels == 1)).sum().item()
        false_positives += ((predicted == 1) & (labels == 0)).sum().item()
        true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
        false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
        
        if i % 10 == 9:
            # Calculate current metrics
            accuracy = 100 * correct / max(1, total)
            sensitivity = 100 * true_positives / max(1, (true_positives + false_negatives))
            specificity = 100 * true_negatives / max(1, (true_negatives + false_positives))
            
            print(f'Batch {i+1}, Loss: {running_loss/10:.3f}, '
                  f'Acc: {accuracy:.2f}%, '
                  f'Sens: {sensitivity:.2f}%, '
                  f'Spec: {specificity:.2f}%')
            running_loss = 0.0
    
    # Calculate final metrics
    accuracy = 100 * correct / max(1, total)
    sensitivity = 100 * true_positives / max(1, (true_positives + false_negatives))
    specificity = 100 * true_negatives / max(1, (true_negatives + false_positives))
    
    return accuracy, sensitivity, specificity


def validate(net, valloader, criterion, device):
    """Validation function with comprehensive metrics."""
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for data in valloader:
            images, labels, _ = data
            images, labels = images.to(device), labels.to(device).float()
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            predicted = (outputs > 0.0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update confusion matrix metrics
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
            
            # Store all outputs and labels for ROC and PR curves
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * correct / max(1, total)
    sensitivity = 100 * true_positives / max(1, (true_positives + false_negatives))
    specificity = 100 * true_negatives / max(1, (true_negatives + false_positives))
    
    # Calculate AUROC and AUPRC
    if len(set(all_labels)) > 1:  # Ensure we have both classes
        fpr, tpr, _ = roc_curve(all_labels, all_outputs)
        auroc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(all_labels, all_outputs)
        auprc = average_precision_score(all_labels, all_outputs)
    else:
        auroc = 0
        auprc = 0
    
    return val_loss / len(valloader), accuracy, sensitivity, specificity, auroc, auprc


def test(net, testloader, device):
    """Extended test function with comprehensive metrics and visualization."""
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
    exam_slice_errors = []  # To track which exams/slices are misclassified
    
    with torch.no_grad():
        for data in testloader:
            images, labels, metadata = data
            images, labels = images.to(device), labels.to(device).float()
            
            outputs = net(images)
            predicted = (outputs > 0.0).float()
            
            # For ROC curve we need probabilities
            probs = torch.sigmoid(outputs)
            
            # Track misclassified samples
            misclassified_indices = (predicted != labels).cpu().numpy()
            misclassified_metadata = [metadata[i] for i, is_error in enumerate(misclassified_indices) if is_error]
            exam_slice_errors.extend(misclassified_metadata)
            
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
    
    # Print misclassified exam/slice pairs
    if exam_slice_errors:
        print("\nMisclassified samples (exam, slice):")
        for exam_num, slice_num in exam_slice_errors:
            print(f"Exam {exam_num}, Slice {slice_num}")
    
    # Calculate AUROC and AUPRC
    if len(set(all_labels)) > 1:  # Ensure we have both classes
        fpr, tpr, _ = roc_curve(all_labels, all_predictions)
        auroc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
        auprc = average_precision_score(all_labels, all_predictions)
        
        print(f'Area Under ROC Curve (AUROC): {auroc:.4f}')
        print(f'Area Under Precision-Recall Curve (AUPRC): {auprc:.4f}')
    else:
        auroc = 0
        auprc = 0
        print("Warning: Test set contains only one class, ROC and PR metrics are not available.")
    
    return accuracy, np.array(all_labels), np.array(all_predictions), (sensitivity, specificity, auroc, auprc)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Frequency Tumor Classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing image files')
    parser.add_argument('--file_patterns', type=str, nargs='+', required=True, 
                         help='File patterns for each frequency (e.g., _1.00GHz_magnitude _2.00GHz_magnitude)')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with labels')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to')
    parser.add_argument('--weight_decay', type=float, default=2e-5, help='L2 regularization factor')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--output_dir', type=str, default='./', help='Directory to save output files')
    parser.add_argument('--min_patterns', type=int, default=None, 
                       help='Minimum number of patterns required per sample (default: all)')
    
    args = parser.parse_args()
    
       
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data augmentation for training - enhanced for medical images
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    
    # No augmentation for validation and test
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    
    # Create dataset with multiple frequency patterns
    print(f"Loading dataset with file patterns: {args.file_patterns}")
    full_dataset = MultiFrequencyTumorDataset(
        args.data_dir, 
        args.file_patterns, 
        args.csv_file, 
        transform=transform_train,
        min_patterns_required=args.min_patterns
    )    
    
    # Count class distribution
    label_counts = {0: 0, 1: 0}
    for _, label, _ in full_dataset:
        label_counts[label] += 1
    
    print(f"Class distribution - Not Malignant: {label_counts[0]}, Malignant: {label_counts[1]}")
    
    # Set class weights
    class_weight = None
    if label_counts[1] > 0:
        # Calculate based on inverse class frequency
        class_weight = label_counts[0] / label_counts[1]
        print(f"Automatically calculated class weight for malignant: {class_weight:.2f}")
    
    # Calculate sample weights for weighted random sampler
    sample_weights = []
    for _, label, _ in full_dataset:
        # Higher weight for minority class
        weight = 1.0 if label == 0 else class_weight
        sample_weights.append(weight)
    
    # Create weighted sampler to address class imbalance
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    # Split dataset FIRST
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    trainset, valset, testset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator)
    
    # Calculate sample weights for the TRAINING set only
    train_sample_weights = []
    for idx in trainset.indices:
        _, label, _ = full_dataset[idx]  
        # Higher weight for minority class
        weight = 1.0 if label == 0 else class_weight
        train_sample_weights.append(weight)
    
    # Create weighted sampler for the training set
    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights, 
        num_samples=len(train_sample_weights), 
        replacement=True
    )
    
    # Create data loaders - using weighted sampler for training
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2
    )
    
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    # Check dataset sizes
    print(f"Training set size: {len(trainset)}")
    print(f"Validation set size: {len(valset)}")
    print(f"Test set size: {len(testset)}")
    
    # Count the actual number of input channels (with complex data creating 2 channels)
    # Count how many complex patterns we have
    complex_patterns = []
    for pattern in args.file_patterns:
        if 'complex' in pattern.lower():
            complex_patterns.append(pattern)

    # Calculate total number of channels
    # Count the actual number of input channels
    # We need to check the actual data type, not just the filename
    sample_data, _, _ = full_dataset[0]  # Get a sample to check actual channels
    num_channels = sample_data.shape[0]  # Use actual channels from data

    print(f"Detected actual input channels: {num_channels}")

    print(f"Total input channels: {num_channels} ({len(complex_patterns)} complex patterns, {len(args.file_patterns) - len(complex_patterns)} real patterns)")

    # Create the network with the correct number of input channels
    net = ImprovedTumorClassifier(in_channels=num_channels, dropout_rate=args.dropout_rate)
    net.to(device)
    
    # Define loss function - use Focal Loss for better handling of class imbalance
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    
    # Define optimizer with higher weight decay for regularization
    optimizer = optim.AdamW(
        net.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        amsgrad=True
    )
    
    # Learning rate scheduler with patience
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, 
        verbose=True, min_lr=1e-6
    )
    
    # Lists to store metrics
    train_metrics_history = []
    val_metrics_history = []
    
    # Training loop
    best_val_metric = 0  # Using a combined metric of sensitivity and specificity
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = 10  # Extended early stopping patience
    
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch+1}/{args.num_epochs}')
        
        # Train
        train_acc, train_sens, train_spec = train_epoch(net, trainloader, optimizer, criterion, device)
        train_metrics_history.append((train_acc, train_sens, train_spec))
        
        # Validate
        val_loss, val_acc, val_sens, val_spec, val_auroc, val_auprc = validate(net, valloader, criterion, device)
        val_metrics_history.append((val_acc, val_sens, val_spec, val_auroc, val_auprc))
        
        # Combined validation metric - balanced accuracy (average of sensitivity and specificity)
        val_metric = (val_sens + val_spec) / 2
        
        print(f'Train Acc: {train_acc:.2f}%, Sens: {train_sens:.2f}%, Spec: {train_spec:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%, Sens: {val_sens:.2f}%, Spec: {val_spec:.2f}%')
        print(f'Val AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}, Loss: {val_loss:.4f}')
        
        # Learning rate adjustment
        scheduler.step(val_metric)
        
        # Save best model
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'tumor_best_model.pth'))
            print(f'New best model saved with validation metric: {val_metric:.2f}%')
            epochs_without_improvement = 0
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'tumor_best_loss_model.pth'))
            print(f'New best model saved with validation loss: {val_loss:.4f}')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs without improvement')
            break
    
    print('Finished Training')
    
    # Load best model
    net.load_state_dict(torch.load(os.path.join(args.output_dir, 'tumor_best_model.pth')))
    
    # Test the model
    test_acc, all_labels, all_predictions, (test_sens, test_spec, test_auroc, test_auprc) = test(testloader, device)
    
    print(f'Final test metrics:')
    print(f'Accuracy: {test_acc:.2f}%')
    print(f'Sensitivity: {test_sens*100:.2f}%')
    print(f'Specificity: {test_spec*100:.2f}%')
    print(f'AUROC: {test_auroc:.4f}')
    print(f'AUPRC: {test_auprc:.4f}')
    
    # Plot training history and metrics
    plt.figure(figsize=(20, 15))
    
    # Accuracy plot
    plt.subplot(3, 2, 1)
    train_acc_history = [m[0] for m in train_metrics_history]
    val_acc_history = [m[0] for m in val_metrics_history]
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy vs. Epoch')
    
    # Sensitivity plot
    plt.subplot(3, 2, 2)
    train_sens_history = [m[1] for m in train_metrics_history]
    val_sens_history = [m[1] for m in val_metrics_history]
    plt.plot(train_sens_history, label='Train Sensitivity')
    plt.plot(val_sens_history, label='Validation Sensitivity')
    plt.xlabel('Epoch')
    plt.ylabel('Sensitivity (%)')
    plt.legend()
    plt.title('Sensitivity vs. Epoch')
    
    # Specificity plot
    plt.subplot(3, 2, 3)
    train_spec_history = [m[2] for m in train_metrics_history]
    val_spec_history = [m[2] for m in val_metrics_history]
    plt.plot(train_spec_history, label='Train Specificity')
    plt.plot(val_spec_history, label='Validation Specificity')
    plt.xlabel('Epoch')
    plt.ylabel('Specificity (%)')
    plt.legend()
    plt.title('Specificity vs. Epoch')
    
    # AUROC history
    plt.subplot(3, 2, 4)
    val_auroc_history = [m[3] for m in val_metrics_history]
    plt.plot(val_auroc_history)
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title('Validation AUROC vs. Epoch')
    
    # AUPRC history
    plt.subplot(3, 2, 5)
    val_auprc_history = [m[4] for m in val_metrics_history]
    plt.plot(val_auprc_history)
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.title('Validation AUPRC vs. Epoch')
    
    # Plot ROC curve
    plt.subplot(3, 2, 6)
    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_results.png'))
    plt.show()


if __name__ == "__main__":
    main()