import os
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tabulate import tabulate
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('performance_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class ModelPerformanceAnalyzer:
    def __init__(self, reports_dir='reports'):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        
    def load_performance_reports(self, pattern="*performance_report*.json"):
        """Load all performance reports matching the pattern"""
        reports = []
        for report_file in self.reports_dir.glob(pattern):
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    report['filename'] = report_file.name
                    reports.append(report)
                logger.info(f"Loaded report: {report_file.name}")
            except Exception as e:
                logger.error(f"Error loading {report_file}: {e}")
        return reports
    
    def analyze_parameter_efficiency(self, reports):
        """Analyze parameter efficiency across models"""
        analysis = []
        
        for report in reports:
            model_info = report['model_info']
            param_info = report['parameter_info']
            
            # Calculate parameter efficiency metrics
            params_per_mb = param_info['trainable_params'] / (1024 * 1024)
            
            analysis.append({
                'model': model_info['architecture'],
                'dataset': model_info['dataset'],
                'mode': model_info['mode'],
                'total_params': param_info['total_params'],
                'trainable_params': param_info['trainable_params'],
                'params_mb': param_info['trainable_params'] * 4 / (1024**2),  # float32
                'hidden_dim': model_info['hidden_dim'],
                'batch_size': model_info['batch_size'],
                'learning_rate': model_info['learning_rate']
            })
        
        return pd.DataFrame(analysis)
    
    def analyze_computational_cost(self, reports):
        """Analyze computational cost metrics"""
        analysis = []
        
        for report in reports:
            model_info = report['model_info']
            comp_cost = report['computational_cost']
            
            analysis.append({
                'model': model_info['architecture'],
                'dataset': model_info['dataset'],
                'mode': model_info['mode'],
                'flops': comp_cost['flops'],
                'gpu_memory_gb': comp_cost['gpu_memory_used_gb'],
                'cpu_memory_gb': comp_cost['cpu_memory_used_gb'],
                'batch_size': model_info['batch_size']
            })
        
        return pd.DataFrame(analysis)
    
    def analyze_timing_performance(self, reports):
        """Analyze timing performance metrics"""
        analysis = []
        
        for report in reports:
            model_info = report['model_info']
            training_metrics = report.get('training_metrics', [])
            validation_metrics = report.get('validation_metrics', [])
            
            if training_metrics:
                avg_epoch_time = np.mean([m['epoch_time'] for m in training_metrics])
                avg_forward_time = np.mean([m['avg_forward_time'] for m in training_metrics])
                avg_backward_time = np.mean([m['avg_backward_time'] for m in training_metrics])
                total_training_time = sum([m['epoch_time'] for m in training_metrics])
            else:
                avg_epoch_time = avg_forward_time = avg_backward_time = total_training_time = 0
            
            if validation_metrics:
                avg_validation_time = np.mean([m['total_validation_time'] for m in validation_metrics])
                avg_inference_time = np.mean([m['avg_inference_time'] for m in validation_metrics])
            else:
                avg_validation_time = avg_inference_time = 0
            
            analysis.append({
                'model': model_info['architecture'],
                'dataset': model_info['dataset'],
                'mode': model_info['mode'],
                'avg_epoch_time': avg_epoch_time,
                'avg_forward_time': avg_forward_time,
                'avg_backward_time': avg_backward_time,
                'total_training_time': total_training_time,
                'avg_validation_time': avg_validation_time,
                'avg_inference_time': avg_inference_time,
                'batch_size': model_info['batch_size']
            })
        
        return pd.DataFrame(analysis)
    
    def generate_comparison_report(self, reports):
        """Generate comprehensive comparison report"""
        if not reports:
            logger.warning("No reports found for analysis")
            return
        
        # Analyze different aspects
        param_df = self.analyze_parameter_efficiency(reports)
        cost_df = self.analyze_computational_cost(reports)
        timing_df = self.analyze_timing_performance(reports)
        
        # Create comprehensive report
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_file = self.reports_dir / f'comprehensive_analysis_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1. PARAMETER EFFICIENCY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(tabulate(param_df, headers='keys', tablefmt='grid', floatfmt='.2f'))
            f.write("\n\n")
            
            f.write("2. COMPUTATIONAL COST ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(tabulate(cost_df, headers='keys', tablefmt='grid', floatfmt='.3f'))
            f.write("\n\n")
            
            f.write("3. TIMING PERFORMANCE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(tabulate(timing_df, headers='keys', tablefmt='grid', floatfmt='.4f'))
            f.write("\n\n")
            
            # Summary statistics
            f.write("4. SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            
            if len(param_df) > 0:
                f.write(f"Average Trainable Parameters: {param_df['trainable_params'].mean():,.0f}\n")
                f.write(f"Max Trainable Parameters: {param_df['trainable_params'].max():,.0f}\n")
                f.write(f"Min Trainable Parameters: {param_df['trainable_params'].min():,.0f}\n")
                f.write(f"Average Model Size: {param_df['params_mb'].mean():.2f} MB\n\n")
            
            if len(timing_df) > 0:
                f.write(f"Average Training Time per Epoch: {timing_df['avg_epoch_time'].mean():.3f}s\n")
                f.write(f"Average Inference Time: {timing_df['avg_inference_time'].mean():.4f}s\n")
                f.write(f"Average Forward Pass Time: {timing_df['avg_forward_time'].mean():.4f}s\n")
                f.write(f"Average Backward Pass Time: {timing_df['avg_backward_time'].mean():.4f}s\n\n")
            
            if len(cost_df) > 0:
                f.write(f"Average GPU Memory Usage: {cost_df['gpu_memory_gb'].mean():.3f} GB\n")
                f.write(f"Average CPU Memory Usage: {cost_df['cpu_memory_gb'].mean():.3f} GB\n")
        
        logger.info(f"Comprehensive analysis saved to: {report_file}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("CRBPR MODEL PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        if len(param_df) > 0:
            print("\nPARAMETER EFFICIENCY:")
            print(tabulate(param_df[['model', 'dataset', 'trainable_params', 'params_mb']], 
                         headers=['Model', 'Dataset', 'Trainable Params', 'Size (MB)'], 
                         tablefmt='grid', floatfmt='.2f'))
        
        if len(timing_df) > 0:
            print("\nTIMING PERFORMANCE:")
            print(tabulate(timing_df[['model', 'dataset', 'avg_epoch_time', 'avg_inference_time']], 
                         headers=['Model', 'Dataset', 'Epoch Time (s)', 'Inference Time (s)'], 
                         tablefmt='grid', floatfmt='.4f'))
        
        if len(cost_df) > 0:
            print("\nCOMPUTATIONAL COST:")
            print(tabulate(cost_df[['model', 'dataset', 'flops', 'gpu_memory_gb']], 
                         headers=['Model', 'Dataset', 'FLOPs', 'GPU Memory (GB)'], 
                         tablefmt='grid', floatfmt='.3f'))
        
        return param_df, cost_df, timing_df
    
    def plot_performance_metrics(self, param_df, timing_df, cost_df):
        """Generate performance visualization plots"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Parameter count comparison
        if len(param_df) > 0:
            axes[0, 0].bar(range(len(param_df)), param_df['trainable_params'] / 1e6)
            axes[0, 0].set_title('Trainable Parameters (Millions)')
            axes[0, 0].set_xticks(range(len(param_df)))
            axes[0, 0].set_xticklabels([f"{row['model']}\n{row['dataset']}" for _, row in param_df.iterrows()], rotation=45)
            
        # Training time comparison
        if len(timing_df) > 0:
            axes[0, 1].bar(range(len(timing_df)), timing_df['avg_epoch_time'])
            axes[0, 1].set_title('Average Training Time per Epoch (seconds)')
            axes[0, 1].set_xticks(range(len(timing_df)))
            axes[0, 1].set_xticklabels([f"{row['model']}\n{row['dataset']}" for _, row in timing_df.iterrows()], rotation=45)
            
        # Inference time comparison
        if len(timing_df) > 0:
            axes[1, 0].bar(range(len(timing_df)), timing_df['avg_inference_time'] * 1000)  # Convert to ms
            axes[1, 0].set_title('Average Inference Time (milliseconds)')
            axes[1, 0].set_xticks(range(len(timing_df)))
            axes[1, 0].set_xticklabels([f"{row['model']}\n{row['dataset']}" for _, row in timing_df.iterrows()], rotation=45)
            
        # Memory usage comparison
        if len(cost_df) > 0:
            axes[1, 1].bar(range(len(cost_df)), cost_df['gpu_memory_gb'])
            axes[1, 1].set_title('GPU Memory Usage (GB)')
            axes[1, 1].set_xticks(range(len(cost_df)))
            axes[1, 1].set_xticklabels([f"{row['model']}\n{row['dataset']}" for _, row in cost_df.iterrows()], rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.reports_dir / f'performance_comparison_{time.strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Performance plots saved to: {plot_file}")
        
        return plot_file

def main():
    parser = argparse.ArgumentParser(description='Model Performance Analysis')
    parser.add_argument('--reports_dir', type=str, default='reports', help='Directory containing performance reports')
    parser.add_argument('--pattern', type=str, default='*performance_report*.json', help='Pattern to match report files')
    parser.add_argument('--output_dir', type=str, default='analysis_output', help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ModelPerformanceAnalyzer(args.reports_dir)
    
    # Load reports
    reports = analyzer.load_performance_reports(args.pattern)
    
    if not reports:
        logger.warning("No performance reports found!")
        return
    
    logger.info(f"Found {len(reports)} performance reports")
    
    # Generate analysis
    param_df, cost_df, timing_df = analyzer.generate_comparison_report(reports)
    
    # Generate plots if we have data
    if len(param_df) > 0 or len(timing_df) > 0 or len(cost_df) > 0:
        try:
            plot_file = analyzer.plot_performance_metrics(param_df, timing_df, cost_df)
            logger.info(f"Visualization plots generated: {plot_file}")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    # Save detailed analysis to CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    if len(param_df) > 0:
        param_file = output_dir / f'parameter_analysis_{timestamp}.csv'
        param_df.to_csv(param_file, index=False)
        logger.info(f"Parameter analysis saved to: {param_file}")
    
    if len(timing_df) > 0:
        timing_file = output_dir / f'timing_analysis_{timestamp}.csv'
        timing_df.to_csv(timing_file, index=False)
        logger.info(f"Timing analysis saved to: {timing_file}")
    
    if len(cost_df) > 0:
        cost_file = output_dir / f'cost_analysis_{timestamp}.csv'
        cost_df.to_csv(cost_file, index=False)
        logger.info(f"Cost analysis saved to: {cost_file}")

if __name__ == '__main__':
    main()