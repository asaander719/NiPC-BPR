#!/usr/bin/env python3
"""
CRBPR Model Performance Demonstration Script

This script demonstrates the enhanced CRBPR implementation with:
1. Comprehensive parameter tracking
2. Computational cost measurement
3. Training and inference timing
4. Performance report generation

Usage:
    python run_crbpr_demo.py --dataset Polyvore_519 --epochs 5 --demo_mode
"""

import os
import sys
import time
import torch
import argparse
import json
import numpy as np
from pathlib import Path

# Add workspace to path
sys.path.append('/workspace')

def create_demo_config():
    """Create a demo configuration for testing"""
    config = {
        'arch': 'CRBPR',
        'dataset': 'Polyvore_519',
        'mode': 'RB',
        'epochs': 5,
        'batch_size': 64,
        'test_batch_size': 64,
        'hidden_dim': 512,
        'base_lr': 0.001,
        'wd': 0.0001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'with_visual': True,
        'with_text': True,
        'UC': True,
        'GC': True,
        'UC_w': 3,
        'GC_w': 1,
        'UC_v_w': 0.5,
        'GC_v_w': 0.5,
        'weight_P': 0.1,
        'early_stop': False,  # Disable for demo
        'evaluate': True,
        'patience': 5
    }
    return config

def demonstrate_crbpr_performance():
    """Demonstrate CRBPR model with performance tracking"""
    print("CRBPR Model Performance Demonstration")
    print("="*60)
    
    # Test model implementation first
    print("\n1. Testing CRBPR Model Implementation...")
    from test_crbpr_performance import test_crbpr_model, create_mock_data
    
    # Create mock data if needed
    if not os.path.exists('dataset/Polyvore_519'):
        print("   Creating mock data...")
        create_mock_data()
    
    # Test basic model functionality
    success = test_crbpr_model()
    if not success:
        print("❌ Model test failed! Exiting...")
        return False
    
    print("✅ Model test passed!")
    
    # Run enhanced training (short demo)
    print("\n2. Running Enhanced CRBPR Training (Demo Mode)...")
    print("   Note: This is a short demo run with 5 epochs")
    
    try:
        # Import and run enhanced training
        os.system('python run_CRBPR_enhanced.py --arch CRBPR --dataset Polyvore_519 --batch_size 64 --mode RB --epochs 5')
        print("✅ Enhanced training completed!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    # Analyze performance reports
    print("\n3. Analyzing Performance Reports...")
    
    try:
        from performance_analysis import ModelPerformanceAnalyzer
        
        analyzer = ModelPerformanceAnalyzer('reports')
        reports = analyzer.load_performance_reports()
        
        if reports:
            print(f"   Found {len(reports)} performance reports")
            param_df, cost_df, timing_df = analyzer.generate_comparison_report(reports)
            print("✅ Performance analysis completed!")
            
            # Print key metrics
            if len(param_df) > 0:
                latest_report = reports[-1]  # Most recent report
                param_info = latest_report['parameter_info']
                comp_cost = latest_report['computational_cost']
                
                print("\n4. CRBPR Model Performance Summary:")
                print("-"*50)
                print(f"   Architecture: CRBPR")
                print(f"   Dataset: {latest_report['model_info']['dataset']}")
                print(f"   Total Parameters: {param_info['total_params']:,}")
                print(f"   Trainable Parameters: {param_info['trainable_params']:,}")
                print(f"   Model Size: {param_info['trainable_params'] * 4 / 1024**2:.2f} MB")
                print(f"   FLOPs: {comp_cost['flops']}")
                print(f"   GPU Memory: {comp_cost['gpu_memory_used_gb']:.3f} GB")
                print(f"   CPU Memory: {comp_cost['cpu_memory_used_gb']:.3f} GB")
                
                if latest_report.get('training_metrics'):
                    avg_epoch_time = np.mean([m['epoch_time'] for m in latest_report['training_metrics']])
                    avg_forward_time = np.mean([m['avg_forward_time'] for m in latest_report['training_metrics']])
                    avg_backward_time = np.mean([m['avg_backward_time'] for m in latest_report['training_metrics']])
                    
                    print(f"   Avg Training Time/Epoch: {avg_epoch_time:.3f}s")
                    print(f"   Avg Forward Pass Time: {avg_forward_time:.4f}s") 
                    print(f"   Avg Backward Pass Time: {avg_backward_time:.4f}s")
                
                if latest_report.get('validation_metrics'):
                    avg_inference_time = np.mean([m['avg_inference_time'] for m in latest_report['validation_metrics']])
                    print(f"   Avg Inference Time: {avg_inference_time:.4f}s")
        else:
            print("   No performance reports found")
            
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n5. Generated Files:")
    print("-"*30)
    
    # List generated files
    if os.path.exists('reports'):
        report_files = list(Path('reports').glob('*'))
        for f in report_files:
            print(f"   📊 {f}")
    
    if os.path.exists('analysis_output'):
        analysis_files = list(Path('analysis_output').glob('*'))
        for f in analysis_files:
            print(f"   📈 {f}")
    
    print("\n✅ CRBPR Performance Demonstration Completed Successfully!")
    print("\nNext Steps:")
    print("- Review the generated performance reports in the 'reports/' directory")
    print("- Analyze the CSV files in 'analysis_output/' for detailed metrics")
    print("- Use the enhanced training script for full experiments")
    print("- Compare performance with other models using the analysis tools")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='CRBPR Performance Demo')
    parser.add_argument('--dataset', type=str, default='Polyvore_519', help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for demo')
    parser.add_argument('--demo_mode', action='store_true', help='Run in demo mode with mock data')
    
    args = parser.parse_args()
    
    if args.demo_mode:
        print("Running CRBPR in demo mode with mock data...")
    
    success = demonstrate_crbpr_performance()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Demo failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()