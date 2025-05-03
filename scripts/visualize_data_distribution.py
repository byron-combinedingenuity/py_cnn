# visualize_data_distribution.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def visualize_dataset_samples(dataset, num_samples=10, output_dir='./data_visualization'):
    """Visualize samples from the dataset to understand data distribution."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get labels for all samples
    all_labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        all_labels.append(label)
    
    all_labels = np.array(all_labels)
    
    # Plot label distribution
    plt.figure(figsize=(10, 6))
    unique, counts = np.unique(all_labels, return_counts=True)
    plt.bar(['Non-Malignant', 'Malignant'], counts)
    plt.title('Class Distribution in Dataset')
    plt.ylabel('Count')
    for i, count in enumerate(counts):
        plt.text(i, count + 0.1, str(count), ha='center')
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    # Find indices for each class
    negative_indices = np.where(all_labels == 0)[0]
    positive_indices = np.where(all_labels == 1)[0]
    
    # Sample some indices
    num_samples_per_class = min(num_samples, len(positive_indices), len(negative_indices))
    neg_samples = np.random.choice(negative_indices, num_samples_per_class, replace=False)
    pos_samples = np.random.choice(positive_indices, num_samples_per_class, replace=False)
    
    # Visualize samples
    fig, axes = plt.subplots(2, num_samples_per_class, figsize=(4*num_samples_per_class, 8))
    
    for i, idx in enumerate(neg_samples):
        img, label, (exam, slice_num) = dataset[idx]
        
        # If multi-channel, show first channel
        if img.shape[0] > 3:
            img_show = img[0]  # Take first channel
        else:
            img_show = img.mean(dim=0) if img.shape[0] > 1 else img[0]
        
        img_np = img_show.numpy()
        
        axes[0, i].imshow(img_np, cmap='gray')
        axes[0, i].set_title(f'Non-Malignant\nExam {exam}, Slice {slice_num}')
        axes[0, i].axis('off')
    
    for i, idx in enumerate(pos_samples):
        img, label, (exam, slice_num) = dataset[idx]
        
        # If multi-channel, show first channel
        if img.shape[0] > 3:
            img_show = img[0]  # Take first channel
        else:
            img_show = img.mean(dim=0) if img.shape[0] > 1 else img[0]
        
        img_np = img_show.numpy()
        
        axes[1, i].imshow(img_np, cmap='gray')
        axes[1, i].set_title(f'Malignant\nExam {exam}, Slice {slice_num}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_images.png'), dpi=150)
    plt.close()
    
    # Analyze pixel value distributions
    plt.figure(figsize=(12, 8))
    
    # Get pixel values for some samples from each class
    neg_pixels = []
    pos_pixels = []
    
    for i in range(min(20, len(negative_indices))):
        img, _, _ = dataset[negative_indices[i]]
        neg_pixels.extend(img.flatten().numpy())
    
    for i in range(min(20, len(positive_indices))):
        img, _, _ = dataset[positive_indices[i]]
        pos_pixels.extend(img.flatten().numpy())
    
    plt.hist(neg_pixels, bins=50, alpha=0.7, label='Non-Malignant', density=True)
    plt.hist(pos_pixels, bins=50, alpha=0.7, label='Malignant', density=True)
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.title('Pixel Value Distribution by Class')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pixel_distributions.png'))
    plt.close()
    
    print(f"Visualization saved to {output_dir}")
    print(f"Total samples: {len(dataset)}")
    print(f"Non-malignant: {counts[0]}")
    print(f"Malignant: {counts[1]}")
    print(f"Class ratio: {counts[0]/counts[1]:.2f}:1")

# If run as script
if __name__ == "__main__":
    import argparse
    from tumor_cnn_multif import MultiFrequencyTumorDataset
    import torchvision.transforms as transforms
    
    parser = argparse.ArgumentParser(description='Visualize dataset distribution')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--file_patterns', type=str, nargs='+', required=True)
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./data_visualization')
    parser.add_argument('--num_samples', type=int, default=5)
    
    args = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = MultiFrequencyTumorDataset(
        args.data_dir,
        args.file_patterns,
        args.csv_file,
        transform=transform
    )
    
    visualize_dataset_samples(dataset, args.num_samples, args.output_dir)