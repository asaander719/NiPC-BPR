# CRBPR Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: `logsigmoid` Function Error

**Error Message:**
```
Traceback (most recent call last):
  File "run_CRBPR.py", line 510, in <module>
    main()
  File "run_CRBPR.py", line 484, in main
    train_metrics = training(args.device, model, train_loader, optimizer, epoch)
  File "run_CRBPR.py", line 134, in training
    loss = (-logsigmoid(output)).sum() 
```

**Cause:** 
- Global variable scope issues with `logger` and `args`
- Import path issues with `logsigmoid`

**Solutions:**

1. **Use the Fixed Version:**
   ```bash
   # Use the corrected script instead
   python run_CRBPR_enhanced_fixed.py --arch CRBPR --dataset Polyvore_519 --batch_size 128 --mode RB
   ```

2. **Manual Fix for Original Script:**
   Replace `logsigmoid(output)` with `F.logsigmoid(output)` in the training function:
   ```python
   # Change this line:
   loss = (-logsigmoid(output)).sum() 
   
   # To this:
   loss = (-F.logsigmoid(output)).sum()
   ```

3. **Add Global Variable Access:**
   Add global variable declarations in functions:
   ```python
   def training(device, model, train_data_loader, optimizer, epoch):
       global logger, args  # Add this line
       # ... rest of function
   ```

### Issue 2: Missing Dependencies

**Error Messages:**
```
ModuleNotFoundError: No module named 'thop'
ModuleNotFoundError: No module named 'psutil'
```

**Solution:**
Install the required dependencies:
```bash
pip install thop psutil GPUtil matplotlib seaborn tabulate scikit-learn
```

Or use the complete requirements file:
```bash
pip install -r requirements.txt
```

### Issue 3: CUDA/GPU Issues

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce Batch Size:**
   ```bash
   python run_CRBPR_enhanced_fixed.py --arch CRBPR --dataset Polyvore_519 --batch_size 32 --mode RB
   ```

2. **Use CPU Mode:**
   ```bash
   python run_CRBPR_enhanced_fixed.py --arch CRBPR --dataset Polyvore_519 --device cpu --mode RB
   ```

3. **Clear GPU Cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Issue 4: Configuration File Not Found

**Error Message:**
```
FileNotFoundError: config/CRBPR_Polyvore_519_RB.yaml not found
```

**Solutions:**

1. **Use Existing Config:**
   ```bash
   # Copy from existing config
   cp config/APCL_Polyvore_519_RB.yaml config/CRBPR_Polyvore_519_RB.yaml
   ```

2. **Modify Architecture in Config:**
   Edit the config file and change:
   ```yaml
   TRAIN:
     arch: CRBPR  # Change from APCL to CRBPR
   ```

### Issue 5: Data Loading Issues

**Error Message:**
```
FileNotFoundError: dataset/Polyvore_519/... not found
```

**Solutions:**

1. **Use Mock Data for Testing:**
   ```bash
   python test_crbpr_performance.py  # This creates mock data
   ```

2. **Check Data Paths:**
   Verify that data files exist in the expected locations:
   ```
   dataset/Polyvore_519/polyvore_U_519_subset_data/
   ├── train_sub_data.csv
   ├── valid_sub_data.csv
   └── test_sub_data.csv
   ```

### Issue 6: Performance Monitoring Libraries Missing

**Warning Messages:**
```
Warning: thop not available, FLOPs calculation disabled
Warning: psutil not available, CPU memory monitoring disabled
```

**Solution:**
This is not critical - the script will run without these libraries, but with reduced performance monitoring capabilities. To get full monitoring:

```bash
pip install thop psutil GPUtil
```

## Recommended Usage Patterns

### 1. Basic Training (Safe Mode)
```bash
# Use the fixed version with smaller batch size
python run_CRBPR_enhanced_fixed.py --arch CRBPR --dataset Polyvore_519 --batch_size 64 --mode RB --epochs 10
```

### 2. Full Performance Monitoring
```bash
# Install all dependencies first
pip install -r requirements.txt

# Run with full monitoring
python run_CRBPR_enhanced_fixed.py --arch CRBPR --dataset Polyvore_519 --batch_size 128 --mode RB
```

### 3. Testing and Validation
```bash
# Test model implementation first
python test_crbpr_performance.py

# Run demo mode
python run_crbpr_demo.py --demo_mode --epochs 5
```

### 4. Performance Analysis
```bash
# After training, analyze results
python performance_analysis.py --reports_dir reports
```

## Key Differences Between Versions

### `run_CRBPR_enhanced.py` (Original)
- Uses global variables that may cause scope issues
- Direct import of `logsigmoid`
- May fail with certain Python/PyTorch versions

### `run_CRBPR_enhanced_fixed.py` (Recommended)
- Robust error handling
- Graceful degradation when optional libraries are missing
- Proper parameter passing to avoid global variable issues
- Uses `F.logsigmoid` for better compatibility
- PerformanceTracker class for better organization

## Environment Setup

### Recommended Environment
```bash
# Create virtual environment
python -m venv crbpr_env
source crbpr_env/bin/activate  # Linux/Mac
# or
crbpr_env\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision
pip install -r requirements.txt
```

### Minimum Requirements
- Python 3.7+
- PyTorch 1.8+
- NumPy
- Pandas
- PyYAML

### Optional (for full monitoring)
- thop (FLOPs calculation)
- psutil (CPU monitoring)
- GPUtil (GPU monitoring)
- matplotlib, seaborn (visualization)

## Performance Optimization Tips

1. **Batch Size Tuning:**
   - Start with smaller batch sizes (32-64)
   - Increase gradually based on GPU memory

2. **Memory Management:**
   - Use `torch.cuda.empty_cache()` between epochs
   - Enable gradient checkpointing for large models

3. **Data Loading:**
   - Use multiple workers: `num_workers=4`
   - Pin memory: `pin_memory=True`

4. **Mixed Precision:**
   - Use `torch.cuda.amp` for faster training
   - Requires PyTorch 1.6+

## Getting Help

If you encounter issues not covered in this guide:

1. Check the generated log files
2. Verify your environment matches the requirements
3. Try the demo mode first: `python run_crbpr_demo.py --demo_mode`
4. Use the fixed version: `run_CRBPR_enhanced_fixed.py`

## Quick Start Commands

```bash
# 1. Test everything works
python test_crbpr_performance.py

# 2. Run demo
python run_crbpr_demo.py --demo_mode

# 3. Run actual training
python run_CRBPR_enhanced_fixed.py --arch CRBPR --dataset Polyvore_519 --batch_size 64 --mode RB

# 4. Analyze results
python performance_analysis.py --reports_dir reports
```