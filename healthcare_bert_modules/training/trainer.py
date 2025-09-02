#!/usr/bin/env python3
"""
Healthcare BERT - Training Module
================================
Training utilities and metrics computation for healthcare sentiment analysis
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HealthcareBERTTrainer:
    """Training utilities for Healthcare BERT model"""
    
    def __init__(self, config):
        self.config = config
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_f1': []
        }
        self.start_time = None
        self.end_time = None
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def start_training_timer(self):
        """Start training timer"""
        self.start_time = time.time()
        print(f"🚀 Training started at: {datetime.now().strftime('%H:%M:%S')}")
    
    def end_training_timer(self):
        """End training timer and calculate duration"""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        print(f"⏱️  Training completed at: {datetime.now().strftime('%H:%M:%S')}")
        if hours > 0:
            print(f"⏱️  Total training time: {hours}h {minutes}m {seconds}s")
        else:
            print(f"⏱️  Total training time: {minutes}m {seconds}s")
        
        return duration
    
    def train_model(self, trainer):
        """Train the model with timing and logging"""
        print("\n🏋️  STARTING MODEL TRAINING")
        print("=" * 50)
        
        self.start_training_timer()
        
        # Train model
        train_result = trainer.train()
        
        # End timing
        duration = self.end_training_timer()
        
        # Log training results
        print(f"✅ Training completed successfully!")
        print(f"📊 Final training loss: {train_result.training_loss:.4f}")
        
        return train_result, duration
    
    def evaluate_model(self, trainer, test_dataset, dataset_name="Test"):
        """Comprehensive model evaluation"""
        print(f"\n📊 EVALUATING MODEL ON {dataset_name.upper()} SET")
        print("=" * 50)
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        
        # Extract predictions and labels
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        weighted_f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]
        
        # Print results
        print(f"🎯 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"📈 Weighted F1-Score: {weighted_f1:.3f}")
        
        # Per-class results
        print(f"\n📊 Per-Class Performance:")
        print("-" * 70)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 70)
        
        for i in range(len(self.config.target_classes)):
            class_name = self.config.target_classes[i]
            print(f"{class_name:<15} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {support[i]:<10}")
        
        print("-" * 70)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'precision': precision,
            'recall': recall,
            'f1_scores': f1,
            'support': support,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_true,
            'classification_report': classification_report(y_true, y_pred, target_names=self.config.target_classes)
        }
        
        return results
    
    def save_training_log(self, results, duration, filepath=None):
        """Save training log to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"training_log_{timestamp}.txt"
        
        with open(filepath, 'w') as f:
            f.write("HEALTHCARE BERT TRAINING LOG\n")
            f.write("=" * 50 + "\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.config.model_name}\n")
            f.write(f"Device: {self.config.device}\n")
            f.write(f"Training Duration: {duration:.2f} seconds\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Learning Rate: {self.config.learning_rate}\n")
            f.write(f"Batch Size: {self.config.batch_size}\n")
            f.write(f"Epochs: {self.config.num_epochs}\n")
            f.write(f"Max Length: {self.config.max_length}\n\n")
            
            f.write("RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.3f}\n")
            f.write(f"Weighted F1: {results['weighted_f1']:.3f}\n\n")
            
            f.write("CLASSIFICATION REPORT:\n")
            f.write("-" * 30 + "\n")
            f.write(results['classification_report'])
        
        print(f"📝 Training log saved to: {filepath}")
    
    def compare_with_baseline(self, accuracy):
        """Compare results with baseline performance"""
        baseline = self.config.expected_accuracy / 100
        improvement = ((accuracy - baseline) / baseline) * 100
        
        print(f"\n🎯 PERFORMANCE COMPARISON")
        print("=" * 40)
        print(f"Expected (Baseline): {baseline:.1%}")
        print(f"Achieved: {accuracy:.1%}")
        
        if improvement > 0:
            print(f"✅ Improvement: +{improvement:.1f}%")
        elif improvement < -5:
            print(f"❌ Performance drop: {improvement:.1f}%")
        else:
            print(f"⚖️  Within expected range: {improvement:.1f}%")
        
        print("=" * 40)
    
    def print_gpu_memory_usage(self):
        """Print GPU memory usage if available"""
        if torch.cuda.is_available():
            print(f"\n🖥️  GPU MEMORY USAGE")
            print("=" * 30)
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            print("=" * 30)

class TrainingCallbacks:
    """Custom training callbacks"""
    
    def __init__(self, config):
        self.config = config
    
    def on_epoch_end(self, trainer, epoch):
        """Callback for end of epoch"""
        print(f"📈 Epoch {epoch + 1}/{self.config.num_epochs} completed")
        
        # Print memory usage if GPU
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"🖥️  GPU Memory: {memory_used:.2f} GB")
    
    def on_training_end(self, trainer):
        """Callback for end of training"""
        print("🎉 Training completed successfully!")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")
