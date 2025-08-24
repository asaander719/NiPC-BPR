#!/usr/bin/env python3
"""
Test script for CRBPR model performance tracking
This script validates the enhanced CRBPR implementation and generates performance reports
"""

import os
import sys
import torch
import numpy as np
import json
import time
from pathlib import Path

# Add the workspace to Python path
sys.path.append('/workspace')

def create_mock_data():
    """Create mock data for testing CRBPR model"""
    # Create mock directories
    os.makedirs('dataset/Polyvore_519/polyvore_U_519_subset_data', exist_ok=True)
    os.makedirs('dataset/Polyvore_519/polyvore_U_519_data', exist_ok=True)
    
    # Create mock training data (user_id, top_id, pos_bottom_id, neg_bottom_id)
    train_data = []
    for i in range(1000):  # 1000 samples
        user_id = np.random.randint(0, 100)  # 100 users
        top_id = np.random.randint(0, 500)   # 500 tops
        pos_bottom_id = np.random.randint(500, 1000)  # 500 bottoms
        neg_bottom_id = np.random.randint(500, 1000)
        train_data.append([user_id, top_id, pos_bottom_id, neg_bottom_id])
    
    # Save training data
    np.savetxt('dataset/Polyvore_519/polyvore_U_519_subset_data/train_sub_data.csv', 
               train_data, delimiter=',', fmt='%d')
    
    # Create smaller validation and test sets
    valid_data = train_data[:200]
    test_data = train_data[:300]
    
    np.savetxt('dataset/Polyvore_519/polyvore_U_519_subset_data/valid_sub_data.csv', 
               valid_data, delimiter=',', fmt='%d')
    np.savetxt('dataset/Polyvore_519/polyvore_U_519_subset_data/test_sub_data.csv', 
               test_data, delimiter=',', fmt='%d')
    
    # Create mock user and item maps
    user_map = {str(i): i for i in range(100)}
    item_map = {str(i): i for i in range(1000)}
    
    with open('dataset/Polyvore_519/polyvore_U_519_data/user_map', 'w') as f:
        json.dump(user_map, f)
    
    with open('dataset/Polyvore_519/polyvore_U_519_data/item_map', 'w') as f:
        json.dump(item_map, f)
    
    # Create mock visual features (1000 items, 2048 features)
    visual_features = torch.randn(1000, 2048)
    torch.save(visual_features, 'dataset/Polyvore_519/polyvore_U_519_data/tb_visual_fea_tensor_new')
    
    # Create mock text features (1000 items, 2400 features)
    text_features = torch.randn(1000, 2400)
    torch.save(text_features, 'dataset/Polyvore_519/polyvore_U_519_data/tb_text_vector')
    
    print("Mock data created successfully!")


def test_crbpr_model():
    """Test CRBPR model with mock data"""
    try:
        # Import necessary modules
        from Models.BPRs.CRBPR import CRBPR
        from config.configurator import parse_configure
        
        # Create a simple args object for testing
        class Args:
            def __init__(self):
                self.user_num = 100
                self.item_num = 1000
                self.hidden_dim = 512
                self.visual_feature_dim = 2048
                self.text_feature_dim = 2400
                self.with_visual = True
                self.with_text = True
                self.with_Nor = True
                self.b_PC = True
                self.cos = True
                self.UC = True
                self.GC = True
                self.UC_v_w = 0.5
                self.GC_v_w = 0.5
                self.UC_w = 3
                self.GC_w = 1
                self.weight_P = 0.1
                self.dataset = 'Polyvore_519'
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.num_his = 3
                self.max_sentence = 83
                self.textcnn_layer = 4
                self.wide_evaluate = True
        
        args = Args()
        
        # Create mock features
        visual_features = torch.randn(1001, 2048)  # +1 for padding
        text_features = torch.randn(1001, 2400)    # +1 for padding
        embedding_weight = None
        
        # Initialize model
        model = CRBPR(args, embedding_weight, visual_features, text_features)
        model.to(args.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"CRBPR Model Test Results:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Model Size: {trainable_params * 4 / 1024**2:.2f} MB")
        print(f"  Device: {args.device}")
        
        # Test forward pass
        batch_size = 32
        sample_batch = [
            torch.randint(0, 100, (batch_size,)).to(args.device),    # users
            torch.randint(0, 500, (batch_size,)).to(args.device),    # tops (items)
            torch.randint(500, 1000, (batch_size,)).to(args.device), # positive bottoms
            torch.randint(500, 1000, (batch_size,)).to(args.device), # negative bottoms
            torch.randint(500, 1000, (batch_size, 3)).to(args.device), # user history bottoms
            torch.randint(0, 500, (batch_size, 3)).to(args.device),    # top history
            torch.randint(500, 1000, (batch_size, 3)).to(args.device)  # top-bottom history
        ]
        
        # Test training forward pass
        start_time = time.time()
        model.train()
        output = model.forward(sample_batch, train=True)
        forward_time = time.time() - start_time
        
        print(f"  Forward Pass Time: {forward_time:.4f}s")
        print(f"  Output Shape: {output.shape}")
        print(f"  Output Range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # Test inference
        start_time = time.time()
        model.eval()
        with torch.no_grad():
            # Modify batch for inference (add candidate items)
            inference_batch = sample_batch.copy()
            inference_batch[3] = torch.randint(500, 1000, (1000,)).to(args.device)  # All candidate items
            
            scores = model.inference(inference_batch, train=False)
        inference_time = time.time() - start_time
        
        print(f"  Inference Time: {inference_time:.4f}s")
        print(f"  Inference Output Shape: {scores.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error testing CRBPR model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("CRBPR Model Performance Test")
    print("="*50)
    
    # Create mock data if needed
    if not os.path.exists('dataset/Polyvore_519'):
        print("Creating mock data for testing...")
        create_mock_data()
    
    # Test model
    print("\nTesting CRBPR model...")
    success = test_crbpr_model()
    
    if success:
        print("\n✅ CRBPR model test completed successfully!")
        print("\nTo run the enhanced CRBPR training with performance tracking:")
        print("python run_CRBPR_enhanced.py --arch CRBPR --dataset Polyvore_519 --batch_size 128 --mode RB")
        print("\nTo analyze performance reports:")
        print("python performance_analysis.py --reports_dir reports")
    else:
        print("\n❌ CRBPR model test failed!")
    
    return success


if __name__ == '__main__':
    main()