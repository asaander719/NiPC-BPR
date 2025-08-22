#!/bin/bash

echo "CRBPR Model Complete Pipeline Demo"
echo "=================================="

# Install dependencies
echo "1. Installing dependencies..."
pip install -r requirements.txt

# Test model implementation
echo "2. Testing CRBPR model implementation..."
python test_crbpr_performance.py

# Run enhanced training (demo mode with 5 epochs)
echo "3. Running enhanced CRBPR training..."
python run_CRBPR_enhanced.py --arch CRBPR --dataset Polyvore_519 --batch_size 64 --mode RB --epochs 5

# Analyze performance
echo "4. Analyzing performance reports..."
python performance_analysis.py --reports_dir reports

# Run comprehensive demo
echo "5. Running comprehensive demo..."
python run_crbpr_demo.py --demo_mode --epochs 3

echo "=================================="
echo "CRBPR Pipeline Demo Completed!"
echo "Check the following directories:"
echo "- reports/ : Performance reports"
echo "- analysis_output/ : Analysis results"
echo "=================================="