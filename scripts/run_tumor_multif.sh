#!/bin/bash

# Script to run the improved multi-frequency tumor CNN with complex data support

# Directory containing the images
DATA_DIR="/mnt/d/work/datasets/breast_models_repository/processed_set_v2/output/masked_images/"

# Create output directories
OUTPUT_DIR="./multi_freq_results"
ANALYSIS_DIR="./complex_data_analysis"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$ANALYSIS_DIR"

# 1. First analyze the complex data in npy files
echo "Analyzing complex data files..."

# Analyze 1.00GHz data
python3 scripts/analyze_complex_data.py \
  --data_dir "$DATA_DIR" \
  --file_pattern "*1.00GHz*.npy" \
  --output_dir "$ANALYSIS_DIR/1.00GHz" \
  --max_files 5

# Analyze 2.00GHz data
python3 scripts/analyze_complex_data.py \
  --data_dir "$DATA_DIR" \
  --file_pattern "*2.00GHz*.npy" \
  --output_dir "$ANALYSIS_DIR/2.00GHz" \
  --max_files 5

echo "Complex data analysis completed. Results saved to $ANALYSIS_DIR"

# Define the different frequency patterns to use
# This includes both .npy files and .png files with various components
FILE_PATTERNS=(
    # 1.00GHz Patterns
    "*_1.00GHz_ccd_complex_masked_complex.npy"
    "*_1.00GHz_ccd_complex_real_masked.npy"
    "*_1.00GHz_ccd_imaginary_masked.npy"
    "*_1.00GHz_ccd_real_masked.npy"
    

    # 1.50GHz Patterns
    "*_1.50GHz_ccd_complex_real_masked.npy"
    "*_1.50GHz_ccd_imaginary_masked.npy"
    "*_1.50GHz_ccd_real_masked.npy"
   

    # 2.00GHz Patterns  
    "*_2.00GHz_ccd_complex_real_masked.npy"
    "*_2.00GHz_ccd_imaginary_masked.npy"
    "*_2.00GHz_ccd_real_masked.npy"
   
    # averaged
    "*_multifreq_avg_complex_real_masked.npy"
    "*_multifreq_avg_complex_real_mask.npy"
    "*_multifreq_avg_imaginary_masked.npy"
    "*_multifreq_avg_real_masked.npy"
)

# CSV file with labels
CSV_FILE="/mnt/d/work/datasets/breast_models_repository/exam_analysis_results.csv"

echo "Starting model training with all frequency data..."

# Run the improved CNN with all available frequency data
python3 scripts/tumor_cnn_multif.py \
  --data_dir "$DATA_DIR" \
  --file_patterns "${FILE_PATTERNS[@]}" \
  --csv_file "$CSV_FILE" \
  --batch_size 8 \
  --num_epochs 100 \
  --learning_rate 0.00005 \
  --image_size 256 \
  --weight_decay 5e-5 \
  --dropout_rate 0.5 \
  --output_dir "$OUTPUT_DIR"

echo "Training completed. Results saved to $OUTPUT_DIR"