#!/usr/bin/env python3
"""
Healthcare BERT - Data Processing Module
=======================================
Handles data loading, preprocessing, and dataset creation for healthcare sentiment analysis
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import re
import warnings
warnings.filterwarnings('ignore')

class HealthcareDataProcessor:
    """Handles all data processing for healthcare sentiment analysis"""
    
    def __init__(self, config):
        self.config = config
        
    def load_raw_data(self):
        """Load raw healthcare drug reviews dataset"""
        print("📊 Loading healthcare drug reviews dataset...")
        
        try:
            df = pd.read_csv(self.config.data_path)
            print(f"✅ Loaded {len(df):,} raw reviews")
            return df
        except FileNotFoundError:
            print(f"❌ Dataset not found at: {self.config.data_path}")
            print("💡 Please ensure the UCI ML Drug Review dataset is available")
            raise
    
    def clean_data(self, df):
        """Clean and preprocess the raw data"""
        print("🧹 Cleaning data...")
        
        # Remove null values
        initial_size = len(df)
        df = df.dropna(subset=['review', 'rating'])
        print(f"📝 Removed {initial_size - len(df):,} rows with missing values")
        
        # Filter short reviews
        df = df[df['review'].str.len() >= self.config.min_review_length]
        print(f"📏 Removed reviews shorter than {self.config.min_review_length} characters")
        
        # Clean review text
        df['review'] = df['review'].apply(self._clean_text)
        
        print(f"✅ Final dataset size: {len(df):,} reviews")
        return df
    
    def _clean_text(self, text):
        """Clean individual review text"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Limit length
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text
    
    def create_sentiment_labels(self, df):
        """Convert drug ratings to sentiment labels (0-4)"""
        print("🏷️  Creating sentiment labels...")
        
        def rating_to_sentiment(rating):
            """Convert 1-10 rating to 0-4 sentiment scale"""
            try:
                rating = float(rating)
                if rating <= 2:
                    return 0  # Very Negative
                elif rating <= 4:
                    return 1  # Negative
                elif rating <= 6:
                    return 2  # Neutral
                elif rating <= 8:
                    return 3  # Positive
                else:
                    return 4  # Very Positive
            except (ValueError, TypeError):
                return 2  # Default to neutral for invalid ratings
        
        df['sentiment'] = df['rating'].apply(rating_to_sentiment)
        
        # Show distribution
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        print("📊 Sentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            label = self.config.target_classes[sentiment]
            percentage = (count / len(df)) * 100
            print(f"  {sentiment} ({label}): {count:,} ({percentage:.1f}%)")
        
        return df
    
    def split_data(self, df):
        """Split data into train/validation/test sets"""
        print("✂️  Splitting data...")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(self.config.val_split + self.config.test_split),
            stratify=df['sentiment'],
            random_state=self.config.random_seed
        )
        
        # Second split: val vs test
        val_size = self.config.val_split / (self.config.val_split + self.config.test_split)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['sentiment'],
            random_state=self.config.random_seed
        )
        
        print(f"📈 Training set: {len(train_df):,} samples")
        print(f"🔍 Validation set: {len(val_df):,} samples") 
        print(f"🧪 Test set: {len(test_df):,} samples")
        
        return train_df, val_df, test_df
    
    def get_dataset_statistics(self, df):
        """Get comprehensive dataset statistics"""
        stats = {
            'total_samples': len(df),
            'avg_review_length': df['review'].str.len().mean(),
            'max_review_length': df['review'].str.len().max(),
            'min_review_length': df['review'].str.len().min(),
            'unique_drugs': df['drugName'].nunique() if 'drugName' in df.columns else 'N/A',
            'class_distribution': df['sentiment'].value_counts().to_dict(),
            'class_imbalance_ratio': df['sentiment'].value_counts().max() / df['sentiment'].value_counts().min()
        }
        return stats
    
    def print_dataset_info(self, df):
        """Print comprehensive dataset information"""
        print("\n📊 DATASET INFORMATION")
        print("=" * 50)
        
        stats = self.get_dataset_statistics(df)
        
        print(f"Total Samples: {stats['total_samples']:,}")
        print(f"Average Review Length: {stats['avg_review_length']:.1f} characters")
        print(f"Review Length Range: {stats['min_review_length']}-{stats['max_review_length']} characters")
        print(f"Unique Drugs: {stats['unique_drugs']}")
        print(f"Class Imbalance Ratio: {stats['class_imbalance_ratio']:.1f}:1")
        
        print("\nClass Distribution:")
        for sentiment, count in sorted(stats['class_distribution'].items()):
            label = self.config.target_classes[sentiment]
            percentage = (count / stats['total_samples']) * 100
            print(f"  {sentiment} ({label}): {count:,} ({percentage:.1f}%)")
        
        print("=" * 50)

class HealthcareDataset(Dataset):
    """PyTorch Dataset for healthcare sentiment analysis"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
