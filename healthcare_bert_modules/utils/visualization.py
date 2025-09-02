#!/usr/bin/env python3
"""
Healthcare BERT - Visualization Module
=====================================
Comprehensive visualization utilities for healthcare sentiment analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HealthcareBERTVisualizer:
    """Visualization utilities for Healthcare BERT analysis"""
    
    def __init__(self, config):
        self.config = config
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create visualizations directory if it doesn't exist
        os.makedirs(self.config.visualizations_path, exist_ok=True)
    
    def plot_class_distribution(self, df, save_name="class_distribution.png"):
        """Plot class distribution of the dataset"""
        plt.figure(figsize=self.config.figure_size)
        
        # Count sentiments
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        labels = [self.config.target_classes[i] for i in sentiment_counts.index]
        
        # Create bar plot
        bars = plt.bar(range(len(sentiment_counts)), sentiment_counts.values, 
                      color=['#ff6b6b', '#ffa726', '#ffeb3b', '#66bb6a', '#42a5f5'])
        
        # Customize plot
        plt.title('Healthcare Drug Reviews - Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Sentiment Class', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, sentiment_counts.values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 500,
                    f'{count:,}\n({count/len(df)*100:.1f}%)', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add statistics
        total = len(df)
        imbalance_ratio = sentiment_counts.max() / sentiment_counts.min()
        plt.figtext(0.02, 0.02, f'Total Samples: {total:,} | Imbalance Ratio: {imbalance_ratio:.1f}:1', 
                   fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.config.visualizations_path, save_name)
        plt.savefig(save_path, dpi=self.config.save_dpi, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Class distribution plot saved to: {save_path}")
    
    def plot_confusion_matrix(self, y_true, y_pred, accuracy, save_name="confusion_matrix.png"):
        """Plot confusion matrix with healthcare-specific styling"""
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.config.target_classes,
                   yticklabels=self.config.target_classes,
                   cbar_kws={'label': 'Number of Samples'})
        
        plt.title(f'Healthcare Sentiment Analysis - BERT\nAccuracy: {accuracy:.1f}%', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.config.visualizations_path, save_name)
        plt.savefig(save_path, dpi=self.config.save_dpi, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    def plot_performance_metrics(self, results, save_name="performance_metrics.png"):
        """Plot detailed performance metrics per class"""
        # Extract metrics
        classes = self.config.target_classes
        precision = results['precision']
        recall = results['recall']
        f1_scores = results['f1_scores']
        
        # Create subplot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
        
        x = range(len(classes))
        
        # Precision
        bars1 = ax1.bar(x, precision, color='#ff9999', alpha=0.8)
        ax1.set_title('Precision by Class', fontweight='bold')
        ax1.set_ylabel('Precision Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars1, precision):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Recall
        bars2 = ax2.bar(x, recall, color='#99ccff', alpha=0.8)
        ax2.set_title('Recall by Class', fontweight='bold')
        ax2.set_ylabel('Recall Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars2, recall):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # F1-Score
        bars3 = ax3.bar(x, f1_scores, color='#99ff99', alpha=0.8)
        ax3.set_title('F1-Score by Class', fontweight='bold')
        ax3.set_ylabel('F1-Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(classes, rotation=45, ha='right')
        ax3.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars3, f1_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add overall metrics
        overall_accuracy = results['accuracy']
        overall_f1 = results['weighted_f1']
        
        fig.suptitle(f'Healthcare BERT Performance Metrics\nOverall Accuracy: {overall_accuracy:.3f} | Weighted F1: {overall_f1:.3f}', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.config.visualizations_path, save_name)
        plt.savefig(save_path, dpi=self.config.save_dpi, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Performance metrics plot saved to: {save_path}")
    
    def plot_dataset_statistics(self, df, save_name="dataset_statistics.png"):
        """Plot comprehensive dataset statistics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Review length distribution
        review_lengths = df['review'].str.len()
        ax1.hist(review_lengths, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_title('Review Length Distribution', fontweight='bold')
        ax1.set_xlabel('Review Length (characters)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(review_lengths.mean(), color='red', linestyle='--', 
                   label=f'Mean: {review_lengths.mean():.0f}')
        ax1.legend()
        
        # 2. Rating distribution (original 1-10 scale)
        if 'rating' in df.columns:
            rating_counts = df['rating'].value_counts().sort_index()
            ax2.bar(rating_counts.index, rating_counts.values, color='lightcoral', alpha=0.7)
            ax2.set_title('Original Rating Distribution (1-10)', fontweight='bold')
            ax2.set_xlabel('Rating')
            ax2.set_ylabel('Count')
        
        # 3. Sentiment vs Rating mapping
        if 'rating' in df.columns:
            sentiment_rating_df = df.groupby(['sentiment', 'rating']).size().unstack(fill_value=0)
            sns.heatmap(sentiment_rating_df, annot=True, fmt='d', cmap='YlOrRd', ax=ax3)
            ax3.set_title('Sentiment-Rating Mapping Heatmap', fontweight='bold')
            ax3.set_xlabel('Original Rating')
            ax3.set_ylabel('Sentiment Class')
            ax3.set_yticklabels(self.config.target_classes, rotation=0)
        
        # 4. Drug frequency (top 10)
        if 'drugName' in df.columns:
            top_drugs = df['drugName'].value_counts().head(10)
            ax4.barh(range(len(top_drugs)), top_drugs.values, color='lightgreen', alpha=0.7)
            ax4.set_title('Top 10 Most Reviewed Drugs', fontweight='bold')
            ax4.set_xlabel('Number of Reviews')
            ax4.set_yticks(range(len(top_drugs)))
            ax4.set_yticklabels(top_drugs.index)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.config.visualizations_path, save_name)
        plt.savefig(save_path, dpi=self.config.save_dpi, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Dataset statistics plot saved to: {save_path}")
    
    def plot_training_progress(self, training_history, save_name="training_progress.png"):
        """Plot training progress over epochs"""
        if not training_history:
            print("⚠️  No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(training_history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, training_history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
        if 'eval_loss' in training_history:
            ax1.plot(epochs, training_history['eval_loss'], 'r-o', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'eval_accuracy' in training_history:
            ax2.plot(epochs, training_history['eval_accuracy'], 'g-o', label='Validation Accuracy', linewidth=2)
        if 'eval_f1' in training_history:
            ax2.plot(epochs, training_history['eval_f1'], 'm-o', label='Validation F1', linewidth=2)
        ax2.set_title('Validation Metrics Progress', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.config.visualizations_path, save_name)
        plt.savefig(save_path, dpi=self.config.save_dpi, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Training progress plot saved to: {save_path}")
    
    def create_model_comparison_plot(self, model_results, save_name="model_comparison.png"):
        """Create comparison plot between different approaches"""
        plt.figure(figsize=(12, 8))
        
        # Example data structure for model_results:
        # {'BERT': {'accuracy': 0.654, 'f1': 0.642}, 'Baseline': {'accuracy': 0.55, 'f1': 0.52}}
        
        models = list(model_results.keys())
        accuracies = [model_results[model]['accuracy'] for model in models]
        f1_scores = [model_results[model]['f1'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, accuracies, width, label='Accuracy', color='#4CAF50', alpha=0.8)
        bars2 = plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='#FF9800', alpha=0.8)
        
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(x, models)
        plt.legend()
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.config.visualizations_path, save_name)
        plt.savefig(save_path, dpi=self.config.save_dpi, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Model comparison plot saved to: {save_path}")
    
    def generate_comprehensive_report(self, df, results, training_duration):
        """Generate a comprehensive visual report"""
        print("\n📊 GENERATING COMPREHENSIVE VISUAL REPORT")
        print("=" * 60)
        
        # 1. Dataset analysis
        self.plot_class_distribution(df, "01_class_distribution.png")
        self.plot_dataset_statistics(df, "02_dataset_statistics.png")
        
        # 2. Model performance
        accuracy_percent = results['accuracy'] * 100
        self.plot_confusion_matrix(results['true_labels'], results['predictions'], 
                                 accuracy_percent, "03_confusion_matrix.png")
        self.plot_performance_metrics(results, "04_performance_metrics.png")
        
        # 3. Create summary report
        self._create_text_summary(df, results, training_duration)
        
        print("✅ Comprehensive visual report generated!")
    
    def _create_text_summary(self, df, results, training_duration):
        """Create text summary of the analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.config.visualizations_path, f"analysis_summary_{timestamp}.txt")
        
        with open(summary_path, 'w') as f:
            f.write("HEALTHCARE BERT SENTIMENT ANALYSIS - SUMMARY REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total samples: {len(df):,}\n")
            f.write(f"Number of classes: {len(self.config.target_classes)}\n")
            f.write(f"Average review length: {df['review'].str.len().mean():.1f} characters\n")
            if 'drugName' in df.columns:
                f.write(f"Unique drugs: {df['drugName'].nunique():,}\n")
            f.write("\n")
            
            f.write("MODEL PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)\n")
            f.write(f"Weighted F1-score: {results['weighted_f1']:.3f}\n")
            f.write(f"Training time: {training_duration/60:.1f} minutes\n")
            f.write(f"Device used: {self.config.device}\n\n")
            
            f.write("CLASS-WISE PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            for i, class_name in enumerate(self.config.target_classes):
                f.write(f"{class_name}: P={results['precision'][i]:.3f}, ")
                f.write(f"R={results['recall'][i]:.3f}, F1={results['f1_scores'][i]:.3f}\n")
        
        print(f"📄 Text summary saved to: {summary_path}")
