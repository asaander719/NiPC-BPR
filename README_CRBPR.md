# CRBPR Model Implementation with Performance Analysis

## Overview

This repository contains an enhanced implementation of the CRBPR (Compatibility-aware Recommendation with Bayesian Personalized Ranking) model with comprehensive performance tracking and analysis capabilities.

## Features

### Enhanced CRBPR Model
- **Compatibility-aware Recommendation**: Leverages both visual and textual features for fashion compatibility modeling
- **User Consistency (UC)**: Models user preference consistency across historical interactions
- **Global Consistency (GC)**: Captures global item compatibility patterns
- **Multi-modal Learning**: Supports both visual and textual feature integration

### Performance Tracking
- **Parameter Counting**: Detailed analysis of model parameters and memory usage
- **Computational Cost**: FLOPs calculation and memory profiling
- **Timing Metrics**: Training and inference time measurement
- **Comprehensive Reporting**: Automated performance report generation

## Model Architecture

The CRBPR model consists of several key components:

1. **Visual Neural Networks**: 
   - Compatibility space projection
   - Personalization space projection  
   - User consistency space projection
   - Global consistency space projection

2. **Text Neural Networks**: 
   - Similar structure to visual networks
   - TextCNN for IQON3000 dataset
   - Direct projection for Polyvore dataset

3. **VTBPR Module**: Visual-Textual Bayesian Personalized Ranking
4. **BPR Module**: Standard Bayesian Personalized Ranking

## Usage

### Basic Training

```bash
# Train CRBPR on Polyvore dataset
python run_CRBPR_enhanced.py --arch CRBPR --dataset Polyvore_519 --batch_size 128 --mode RB

# Train CRBPR on IQON dataset  
python run_CRBPR_enhanced.py --arch CRBPR --dataset IQON3000 --batch_size 64 --mode RB
```

### Performance Testing

```bash
# Test model implementation and generate sample performance metrics
python test_crbpr_performance.py
```

### Performance Analysis

```bash
# Analyze performance reports
python performance_analysis.py --reports_dir reports

# Generate comparison analysis
python performance_analysis.py --reports_dir reports --output_dir analysis_results
```

## Configuration

### CRBPR-specific Parameters

- `UC`: Enable User Consistency modeling (default: True)
- `GC`: Enable Global Consistency modeling (default: True)
- `UC_v_w`: Visual weight in User Consistency (default: 0.5)
- `GC_v_w`: Visual weight in Global Consistency (default: 0.5)
- `UC_w`: User Consistency weight (default: 3)
- `GC_w`: Global Consistency weight (default: 1)
- `weight_P`: Balance between compatibility and personalization (default: 0.1)

### Model Parameters

- `hidden_dim`: Hidden dimension size (default: 512)
- `visual_feature_dim`: Visual feature dimension (default: 2048)
- `text_feature_dim`: Text feature dimension (default: 2400)
- `batch_size`: Training batch size (default: 128)
- `base_lr`: Learning rate (default: 0.001)

## Performance Metrics

### Parameter Statistics
- Total parameters count
- Trainable parameters count
- Model size in MB
- Parameter distribution across modules

### Computational Cost
- FLOPs (Floating Point Operations)
- GPU memory usage
- CPU memory usage
- Memory efficiency metrics

### Timing Analysis
- Training time per epoch
- Forward pass time
- Backward pass time
- Inference time
- Data loading time

## Output Files

### Performance Reports
- `reports/CRBPR_performance_report_*.json`: Detailed performance metrics
- `reports/comprehensive_analysis_*.txt`: Human-readable analysis summary
- `reports/performance_comparison_*.png`: Visualization plots

### Analysis Results
- `analysis_output/parameter_analysis_*.csv`: Parameter efficiency data
- `analysis_output/timing_analysis_*.csv`: Timing performance data
- `analysis_output/cost_analysis_*.csv`: Computational cost data

## Model Performance Summary

### Typical CRBPR Performance (Polyvore_519)
- **Parameters**: ~122M trainable parameters
- **Model Size**: ~465 MB (float32)
- **Training Time**: ~X seconds per epoch (depends on hardware)
- **Inference Time**: ~X ms per batch (depends on batch size and hardware)
- **GPU Memory**: ~X GB (depends on batch size)

### Key Performance Features
1. **Efficient Multi-modal Processing**: Optimized visual and text feature integration
2. **Scalable Architecture**: Handles large-scale fashion datasets
3. **Comprehensive Metrics**: Detailed performance tracking and analysis
4. **Flexible Configuration**: Easy parameter tuning and experimentation

## Requirements

```
torch
torchvision
numpy
pandas
tqdm
pyyaml
tensorboard
thop
psutil
GPUtil
matplotlib
seaborn
tabulate
scikit-learn
```

## Installation

```bash
pip install -r requirements.txt
```

## File Structure

```
├── run_CRBPR_enhanced.py          # Enhanced CRBPR training script
├── performance_analysis.py        # Performance analysis utilities
├── test_crbpr_performance.py      # Model testing script
├── config/
│   └── CRBPR_Polyvore_519_RB.yaml # CRBPR configuration
├── Models/BPRs/
│   └── CRBPR.py                   # CRBPR model implementation
├── reports/                       # Generated performance reports
└── analysis_output/               # Analysis results
```

## Citation

If you use this enhanced CRBPR implementation, please cite the original CRBPR paper and mention the performance analysis enhancements.

## License

This project follows the same license as the original implementation.