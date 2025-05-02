#!/usr/bin/env python3
# analyze_complex_data.py - Tool to analyze complex values in RF imaging data
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import argparse
from PIL import Image


def analyze_npy_file(file_path, save_plots=True, output_dir='./analysis_results'):
    """Analyze a numpy file containing RF imaging data."""
    
    print(f"Analyzing: {file_path}")
    
    # Create output dir if saving plots
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the data
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Get file basename for plot titles
    basename = os.path.basename(file_path)
    
    # Print basic information
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Is complex: {np.iscomplexobj(data)}")
    
    if np.iscomplexobj(data):
        # Complex data statistics
        real_part = np.real(data)
        imag_part = np.imag(data)
        magnitude = np.abs(data)
        phase = np.angle(data)
        
        print("\nComplex data statistics:")
        print(f"Real part - Min: {real_part.min()}, Max: {real_part.max()}, Mean: {real_part.mean()}")
        print(f"Imaginary part - Min: {imag_part.min()}, Max: {imag_part.max()}, Mean: {imag_part.mean()}")
        print(f"Magnitude - Min: {magnitude.min()}, Max: {magnitude.max()}, Mean: {magnitude.mean()}")
        print(f"Phase (radians) - Min: {phase.min()}, Max: {phase.max()}, Mean: {phase.mean()}")
        
        # Create figure for visualization
        if save_plots:
            plt.figure(figsize=(16, 12))
            
            plt.subplot(2, 2, 1)
            plt.imshow(real_part, cmap='gray')
            plt.colorbar()
            plt.title('Real Part')
            
            plt.subplot(2, 2, 2)
            plt.imshow(imag_part, cmap='gray')
            plt.colorbar()
            plt.title('Imaginary Part')
            
            plt.subplot(2, 2, 3)
            plt.imshow(magnitude, cmap='viridis')
            plt.colorbar()
            plt.title('Magnitude')
            
            plt.subplot(2, 2, 4)
            plt.imshow(phase, cmap='hsv')
            plt.colorbar()
            plt.title('Phase')
            
            plt.suptitle(f"Complex Data Analysis: {basename}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{os.path.splitext(basename)[0]}_analysis.png"))
            
            # Save the four components as separate images
            norm_real = normalize_for_image(real_part)
            norm_imag = normalize_for_image(imag_part)
            norm_mag = normalize_for_image(magnitude)
            norm_phase = normalize_for_image(phase, is_phase=True)
            
            Image.fromarray(norm_real).save(os.path.join(output_dir, f"{os.path.splitext(basename)[0]}_real.png"))
            Image.fromarray(norm_imag).save(os.path.join(output_dir, f"{os.path.splitext(basename)[0]}_imag.png"))
            Image.fromarray(norm_mag).save(os.path.join(output_dir, f"{os.path.splitext(basename)[0]}_mag.png"))
            Image.fromarray(norm_phase).save(os.path.join(output_dir, f"{os.path.splitext(basename)[0]}_phase.png"))
            
            plt.close()
    else:
        # Real data statistics
        print("\nReal data statistics:")
        print(f"Min: {data.min()}, Max: {data.max()}, Mean: {data.mean()}")
        
        # Visualize the real data
        if save_plots:
            plt.figure(figsize=(10, 8))
            plt.imshow(data, cmap='gray')
            plt.colorbar()
            plt.title(f"Real Data: {basename}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{os.path.splitext(basename)[0]}_analysis.png"))
            plt.close()
            
            # Save normalized image
            norm_data = normalize_for_image(data)
            Image.fromarray(norm_data).save(os.path.join(output_dir, f"{os.path.splitext(basename)[0]}_normalized.png"))


def normalize_for_image(data, is_phase=False):
    """Normalize data to 0-255 range for saving as an image."""
    if is_phase:
        # Phase is in -π to π range, normalize to 0-255
        normalized = ((data + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    else:
        # Standard min-max normalization
        if data.max() != data.min():
            normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(data, dtype=np.uint8)
    
    return normalized


def batch_analyze_files(data_dir, file_pattern, output_dir='./analysis_results', max_files=10):
    """Analyze multiple files matching a pattern."""
    
    # Find files matching the pattern
    if '*' in file_pattern:
        files = glob.glob(os.path.join(data_dir, file_pattern))
    else:
        files = glob.glob(os.path.join(data_dir, f"*{file_pattern}*"))
    
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    print(f"Found {len(files)} files matching pattern. Analyzing up to {max_files}...")
    
    # Analyze a subset of files
    for i, file_path in enumerate(files[:max_files]):
        print(f"\nFile {i+1}/{min(max_files, len(files))}")
        analyze_npy_file(file_path, save_plots=True, output_dir=output_dir)
    
    print("\nBatch analysis complete.")


def main():
    parser = argparse.ArgumentParser(description='Analyze complex RF imaging data.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing data files')
    parser.add_argument('--file_pattern', type=str, required=True, help='Pattern to match files (e.g., *_1.00GHz*.npy)')
    parser.add_argument('--output_dir', type=str, default='./analysis_results', help='Directory to save analysis results')
    parser.add_argument('--max_files', type=int, default=10, help='Maximum number of files to analyze')
    
    args = parser.parse_args()
    
    batch_analyze_files(args.data_dir, args.file_pattern, args.output_dir, args.max_files)


if __name__ == "__main__":
    main()