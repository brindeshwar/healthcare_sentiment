#!/usr/bin/env python3
"""
Healthcare BERT - Model Module
=============================
BERT model configuration and setup for healthcare sentiment analysis
"""

import torch
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import warnings
warnings.filterwarnings('ignore')

class HealthcareBERTModel:
    """Healthcare-specific BERT model wrapper"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_tokenizer(self):
        """Load and configure the DistilBERT tokenizer"""
        print(f"🔤 Loading tokenizer: {self.config.model_name}")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.config.model_name)
        print("✅ Tokenizer loaded successfully")
        return self.tokenizer
    
    def create_model(self):
        """Create and configure the DistilBERT model"""
        print(f"🤖 Creating model: {self.config.model_name}")
        print(f"🎯 Target classes: {self.config.num_labels}")
        print(f"💾 Device: {self.config.device}")
        
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )
        
        # Move model to device
        self.model.to(self.config.device)
        
        print("✅ Model created and moved to device")
        return self.model
    
    def setup_trainer(self, train_dataset, val_dataset, compute_metrics_fn):
        """Setup HuggingFace Trainer"""
        print("🏋️  Setting up trainer...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            save_total_limit=2,
            dataloader_pin_memory=True if self.config.device.type == 'cuda' else False,
            fp16=True if self.config.device.type == 'cuda' else False,  # Mixed precision for GPU
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
        )
        
        print("✅ Trainer configured successfully")
        return self.trainer
    
    def save_model(self, path=None):
        """Save the trained model"""
        save_path = path or self.config.model_save_path
        
        print(f"💾 Saving model to: {save_path}")
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print("✅ Model saved successfully")
    
    def load_model(self, path=None):
        """Load a pre-trained model"""
        load_path = path or self.config.model_save_path
        
        print(f"📥 Loading model from: {load_path}")
        
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(load_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(load_path)
            self.model.to(self.config.device)
            
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_single(self, text):
        """Predict sentiment for a single text"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call create_model() or load_model() first.")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': self.config.target_classes[predicted_class],
            'confidence': confidence,
            'all_probabilities': predictions[0].tolist()
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for a batch of texts"""
        predictions = []
        for text in texts:
            pred = self.predict_single(text)
            predictions.append(pred)
        return predictions
    
    def get_model_info(self):
        """Get model information"""
        if not self.model:
            return {"status": "Model not loaded"}
        
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name,
            "num_labels": self.config.num_labels,
            "total_parameters": param_count,
            "trainable_parameters": trainable_params,
            "device": str(self.config.device),
            "max_length": self.config.max_length
        }
    
    def print_model_info(self):
        """Print model information"""
        info = self.get_model_info()
        
        print("\n🤖 MODEL INFORMATION")
        print("=" * 50)
        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value:,}" if isinstance(value, int) else f"{key.replace('_', ' ').title()}: {value}")
        print("=" * 50)
