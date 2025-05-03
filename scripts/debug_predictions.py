# debug_predictions.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def debug_model_predictions(net, dataloader, device, output_dir='./debug_output', num_samples=100):
    """Debug model predictions and output distributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    net.eval()
    
    all_outputs = []
    all_labels = []
    all_probs = []
    all_predictions = []
    
    print("Debugging model predictions...")
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i * dataloader.batch_size >= num_samples:
                break
                
            images, labels, _ = data
            images, labels = images.to(device), labels.to(device).float()
            
            outputs = net(images)
            probs = torch.sigmoid(outputs)
            predictions = (outputs > 0.0).float()
            
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)
    
    # Print statistics
    print(f"Number of samples analyzed: {len(all_labels)}")
    print(f"Label distribution: {np.unique(all_labels, return_counts=True)}")
    print(f"Prediction distribution: {np.unique(all_predictions, return_counts=True)}")
    print(f"Output (logit) statistics:")
    print(f"  Min: {all_outputs.min():.4f}")
    print(f"  Max: {all_outputs.max():.4f}")
    print(f"  Mean: {all_outputs.mean():.4f}")
    print(f"  Std: {all_outputs.std():.4f}")
    print(f"Probability statistics:")
    print(f"  Min: {all_probs.min():.4f}")
    print(f"  Max: {all_probs.max():.4f}")
    print(f"  Mean: {all_probs.mean():.4f}")
    print(f"  Std: {all_probs.std():.4f}")
    
    # Plot distributions
    plt.figure(figsize=(15, 10))
    
    # Logit distribution
    plt.subplot(2, 2, 1)
    plt.hist(all_outputs[all_labels == 0], bins=30, alpha=0.7, label='Negative class', density=True)
    plt.hist(all_outputs[all_labels == 1], bins=30, alpha=0.7, label='Positive class', density=True)
    plt.xlabel('Output (logits)')
    plt.ylabel('Density')
    plt.title('Output Distribution by Class')
    plt.legend()
    
    # Probability distribution
    plt.subplot(2, 2, 2)
    plt.hist(all_probs[all_labels == 0], bins=30, alpha=0.7, label='Negative class', density=True)
    plt.hist(all_probs[all_labels == 1], bins=30, alpha=0.7, label='Positive class', density=True)
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title('Probability Distribution by Class')
    plt.legend()
    
    # Scatter plot
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(range(len(all_outputs)), all_outputs, c=all_labels, cmap='coolwarm', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Output (logit)')
    plt.title('Outputs by Sample')
    plt.colorbar(scatter, label='True Label')
    
    # Box plot
    plt.subplot(2, 2, 4)
    data_to_plot = [all_outputs[all_labels == 0], all_outputs[all_labels == 1]]
    plt.boxplot(data_to_plot, labels=['Negative', 'Positive'])
    plt.ylabel('Output (logit)')
    plt.title('Output Distribution by Class (Box Plot)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_debug_plots.png'))
    plt.close()
    
    # Check gradient flow
    plt.figure(figsize=(12, 8))
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in net.named_parameters():
        if p.requires_grad and p.grad is not None and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())
    
    if layers:  # Only plot if we have gradients
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=max(max_grads) * 1.1)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("Average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                   plt.Line2D([0], [0], color="b", lw=4),
                   plt.Line2D([0], [0], color="k", lw=4)], 
                  ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_flow.png'))
    else:
        print("No gradients available for plotting.")
    
    plt.close()
    
    return all_outputs, all_labels, all_probs, all_predictions

def check_weight_distribution(net, output_dir='./debug_output'):
    """Check weight distributions across layers."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    layer_names = []
    weight_stats = []
    
    for name, param in net.named_parameters():
        if 'weight' in name:
            layer_names.append(name)
            weights = param.data.cpu().numpy().flatten()
            weight_stats.append({
                'mean': weights.mean(),
                'std': weights.std(),
                'min': weights.min(),
                'max': weights.max()
            })
    
    # Plot weight distributions for first 4 layers
    for i, (name, param) in enumerate(net.named_parameters()):
        if i >= 4:
            break
        if 'weight' in name:
            weights = param.data.cpu().numpy().flatten()
            axes[i].hist(weights, bins=50, alpha=0.7)
            axes[i].set_title(f'{name}\nMean: {weights.mean():.4f}, Std: {weights.std():.4f}')
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_distributions.png'))
    plt.close()
    
    # Print weight statistics
    print("\nWeight Statistics:")
    for name, stats in zip(layer_names, weight_stats):
        print(f"{name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")

def analyze_batch_predictions(net, dataloader, device, output_dir='./debug_output', num_batches=5):
    """Analyze predictions batch by batch."""
    os.makedirs(output_dir, exist_ok=True)
    
    net.eval()
    
    print("\nBatch-by-batch analysis:")
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= num_batches:
                break
                
            images, labels, _ = data
            images, labels = images.to(device), labels.to(device).float()
            
            outputs = net(images)
            probs = torch.sigmoid(outputs)
            predictions = (outputs > 0.0).float()
            
            print(f"\nBatch {i+1}:")
            print(f"Labels: {labels.cpu().numpy()}")
            print(f"Outputs (logits): {outputs.cpu().numpy()}")
            print(f"Probabilities: {probs.cpu().numpy()}")
            print(f"Predictions: {predictions.cpu().numpy()}")
            
            # Check if all predictions are the same
            if torch.all(predictions == predictions[0]):
                print(f"WARNING: All predictions in batch are {predictions[0].item()}")