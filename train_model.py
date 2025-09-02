#!/usr/bin/env python3
"""
Healthcare BERT - Professional Model Training
============================================
Complete healthcare sentiment analysis using professional modular BERT architecture
"""

import sys
import os

# Add the modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'healthcare_bert_modules'))

from config import config
from data.preprocessing import HealthcareDataProcessor, HealthcareDataset
from models.bert_model import HealthcareBERTModel
from training.trainer import HealthcareBERTTrainer
from utils.visualization import HealthcareBERTVisualizer

def check_gpu():
    """Check GPU availability and configuration"""
    print("🔍 GPU CONFIGURATION CHECK")
    print("=" * 40)
    
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"✅ GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"✅ Current Device: {torch.cuda.current_device()}")
        print("🚀 Training will use GPU acceleration!")
    else:
        print("❌ CUDA not available - CPU training will be VERY slow!")
        print("💡 Please ensure you have CUDA-enabled PyTorch installed:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        response = input("\nContinue with CPU training? (y/N): ")
        if response.lower() != 'y':
            exit("Training cancelled. Please install CUDA-enabled PyTorch.")
    
    print(f"🎯 Using device: {config.device}")
    print("=" * 40)

def main():
    """Main training pipeline"""
    print("🏥 HEALTHCARE SENTIMENT ANALYSIS - MODULAR BERT TRAINING")
    print("=" * 80)
    print("📅 Using modularized architecture for better maintainability")
    print("🎯 Target: 65.4% accuracy with GPU acceleration")
    print("=" * 80)
    
    # Check GPU
    check_gpu()
    
    # Print configuration
    config.print_config()
    
    # Initialize components
    print("\n🔧 INITIALIZING COMPONENTS")
    print("=" * 40)
    
    data_processor = HealthcareDataProcessor(config)
    bert_model = HealthcareBERTModel(config)
    trainer_utils = HealthcareBERTTrainer(config)
    visualizer = HealthcareBERTVisualizer(config)
    
    print("✅ All components initialized")
    
    # Step 1: Load and process data
    print("\n📊 STEP 1: DATA PROCESSING")
    print("=" * 40)
    
    # Load raw data
    df = data_processor.load_raw_data()
    
    # Clean data
    df = data_processor.clean_data(df)
    
    # Create sentiment labels
    df = data_processor.create_sentiment_labels(df)
    
    # Print dataset info
    data_processor.print_dataset_info(df)
    
    # Split data
    train_df, val_df, test_df = data_processor.split_data(df)
    
    # Step 2: Create visualizations for dataset
    print("\n📈 STEP 2: DATASET VISUALIZATION")
    print("=" * 40)
    
    # Dataset visualizations
    visualizer.plot_class_distribution(df, "dataset_class_distribution.png")
    visualizer.plot_dataset_statistics(df, "dataset_comprehensive_stats.png")
    
    # Step 3: Prepare model
    print("\n🤖 STEP 3: MODEL PREPARATION")
    print("=" * 40)
    
    # Load tokenizer and create model
    tokenizer = bert_model.load_tokenizer()
    model = bert_model.create_model()
    
    # Print model info
    bert_model.print_model_info()
    
    # Step 4: Prepare datasets
    print("\n📋 STEP 4: DATASET PREPARATION")
    print("=" * 40)
    
    train_dataset = HealthcareDataset(
        train_df['review'], train_df['sentiment'], tokenizer, config.max_length
    )
    val_dataset = HealthcareDataset(
        val_df['review'], val_df['sentiment'], tokenizer, config.max_length
    )
    test_dataset = HealthcareDataset(
        test_df['review'], test_df['sentiment'], tokenizer, config.max_length
    )
    
    print(f"✅ Training dataset: {len(train_dataset):,} samples")
    print(f"✅ Validation dataset: {len(val_dataset):,} samples")
    print(f"✅ Test dataset: {len(test_dataset):,} samples")
    
    # Step 5: Setup trainer
    print("\n🏋️  STEP 5: TRAINER SETUP")
    print("=" * 40)
    
    trainer = bert_model.setup_trainer(
        train_dataset, val_dataset, trainer_utils.compute_metrics
    )
    
    # Step 6: Train model
    print("\n🚀 STEP 6: MODEL TRAINING")
    print("=" * 40)
    
    train_result, duration = trainer_utils.train_model(trainer)
    
    # Print memory usage
    trainer_utils.print_gpu_memory_usage()
    
    # Step 7: Evaluate model
    print("\n📊 STEP 7: MODEL EVALUATION")
    print("=" * 40)
    
    # Evaluate on test set
    results = trainer_utils.evaluate_model(trainer, test_dataset, "Test")
    
    # Compare with baseline
    trainer_utils.compare_with_baseline(results['accuracy'])
    
    # Step 8: Save model
    print("\n💾 STEP 8: SAVE MODEL")
    print("=" * 40)
    
    bert_model.save_model()
    
    # Step 9: Create visualizations
    print("\n📈 STEP 9: RESULT VISUALIZATION")
    print("=" * 40)
    
    # Model performance visualizations
    accuracy_percent = results['accuracy'] * 100
    visualizer.plot_confusion_matrix(
        results['true_labels'], results['predictions'], 
        accuracy_percent, "bert_confusion_matrix.png"
    )
    
    visualizer.plot_performance_metrics(results, "bert_performance_metrics.png")
    
    # Generate comprehensive report
    visualizer.generate_comprehensive_report(df, results, duration)
    
    # Step 10: Save training log
    print("\n📝 STEP 10: SAVE TRAINING LOG")
    print("=" * 40)
    
    trainer_utils.save_training_log(results, duration)
    
    # Final summary
    print("\n🎉 TRAINING COMPLETE!")
    print("=" * 50)
    print(f"🎯 Final Accuracy: {results['accuracy']:.3f} ({accuracy_percent:.1f}%)")
    print(f"📈 Weighted F1-Score: {results['weighted_f1']:.3f}")
    print(f"⏱️  Training Time: {duration/60:.1f} minutes")
    print(f"💾 Model saved to: {config.model_save_path}")
    print(f"📊 Visualizations saved to: {config.visualizations_path}")
    print("=" * 50)
    
    # Test single prediction
    print("\n🧪 TESTING SINGLE PREDICTION")
    print("=" * 40)
    
    test_text = "This medication helped with my pain but caused some side effects."
    prediction = bert_model.predict_single(test_text)
    
    print(f"📝 Test review: '{test_text}'")
    print(f"🎯 Predicted sentiment: {prediction['predicted_label']} (class {prediction['predicted_class']})")
    print(f"📊 Confidence: {prediction['confidence']:.3f}")
    
    print("\n✅ Healthcare BERT training completed successfully!")
    return results

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
