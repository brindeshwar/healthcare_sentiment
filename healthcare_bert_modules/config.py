#!/usr/bin/env python3
"""
Healthcare BERT - Configuration Settings
=======================================
Centralized configuration for all healthcare sentiment analysis components
"""

import torch
import os

class HealthcareBERTConfig:
    """Configuration class for Healthcare BERT sentiment analysis"""
    
    def __init__(self):
        # Project paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(self.project_root, "healthcare_sentiment", "drug_reviews", "drugsComTrain_raw.csv")
        self.model_save_path = os.path.join(self.project_root, "healthcare_bert_final")
        self.visualizations_path = os.path.join(self.project_root, "visualizations")
        
        # Model configuration
        self.model_name = "distilbert-base-uncased"
        self.num_labels = 5
        self.max_length = 512
        
        # Training hyperparameters
        self.learning_rate = 2e-5
        self.batch_size = 16
        self.num_epochs = 3
        self.warmup_steps = 0
        self.weight_decay = 0.01
        
        # Data processing
        self.train_split = 0.8
        self.val_split = 0.1
        self.test_split = 0.1
        self.min_review_length = 20
        self.random_seed = 42
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Visualization settings
        self.figure_size = (12, 8)
        self.color_palette = "viridis"
        self.save_dpi = 300
        
        # Model performance tracking
        self.expected_accuracy = 65.4
        self.target_classes = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
        
    def get_training_args(self):
        """Get training arguments for HuggingFace Trainer"""
        return {
            'output_dir': './results',
            'num_train_epochs': self.num_epochs,
            'per_device_train_batch_size': self.batch_size,
            'per_device_eval_batch_size': self.batch_size,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'logging_dir': './logs',
            'logging_steps': 100,
            'eval_strategy': "epoch",
            'save_strategy': "epoch",
            'load_best_model_at_end': True,
            'metric_for_best_model': "eval_accuracy",
            'greater_is_better': True,
            'save_total_limit': 2,
            'dataloader_pin_memory': True if self.device.type == 'cuda' else False,
        }
    
    def print_config(self):
        """Print current configuration"""
        print("🔧 HEALTHCARE BERT CONFIGURATION")
        print("=" * 50)
        print(f"📊 Dataset: {self.data_path}")
        print(f"🤖 Model: {self.model_name}")
        print(f"💾 Save Path: {self.model_save_path}")
        print(f"🎯 Device: {self.device}")
        print(f"📈 Expected Accuracy: {self.expected_accuracy}%")
        print(f"🔢 Classes: {len(self.target_classes)}")
        print(f"⚙️  Batch Size: {self.batch_size}")
        print(f"🔄 Epochs: {self.num_epochs}")
        print(f"📏 Max Length: {self.max_length}")
        print("=" * 50)

# Global config instance
config = HealthcareBERTConfig()

if __name__ == "__main__":
    config.print_config()
