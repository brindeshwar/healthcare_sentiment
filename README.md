# Healthcare Sentiment Analysis with BERT

A professional sentiment analysis system for healthcare drug reviews using DistilBERT with GPU acceleration and modular architecture.

## 🎯 Project Overview

This project performs sentiment analysis on healthcare drug reviews from the UCI ML Drug Review dataset. **DistilBERT achieves 65.4% accuracy** with GPU-optimized training in just 8 minutes using enterprise-grade modular architecture.

## 🏆 Model Performance: 65.4% Accuracy

**Production-Ready BERT Model** with professional modular codebase for healthcare sentiment classification (5 classes: Very Negative → Very Positive)

## 🚀 Quick Start - Professional Training

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: RTX 4060 or better)  
- 8GB+ RAM

### Installation & Training
```bash
# Setup and train with modular architecture
cd minor_project
pip install -r requirements.txt
python train_model.py
```

**Professional modular training completes in ~8 minutes with GPU acceleration!**

## 📁 Project Structure (Professional Modular Architecture)

```
minor_project/
├── 📚 Documentation
│   ├── README.md                     # Comprehensive project overview
│   └── PROJECT_REPORT.md            # Complete technical analysis
│
├── 🤖 BERT Training (Modular Architecture)
│   ├── train_model.py               # 🆕 Professional modular training
│   └── healthcare_bert_final/       # Trained model (65.4% accuracy)
│
├── 🏗️ Professional Module Structure
│   └── healthcare_bert_modules/     # Enterprise-ready architecture
│       ├── config.py               # Centralized configuration
│       ├── data/preprocessing.py    # Dataset handling & cleaning
│       ├── models/bert_model.py     # BERT model wrapper
│       ├── training/trainer.py      # Training & evaluation utilities
│       └── utils/visualization.py   # Comprehensive visualizations
│
├── 📊 Data & Visualizations
│   ├── healthcare_sentiment/        # Dataset folder (160K+ reviews)
│   ├── requirements.txt            # Dependencies
│   ├── create_visualizations.py    # Generate comprehensive charts
│   ├── visualizations/             # Professional analysis charts
│   │   ├── healthcare_class_distribution.png
│   │   ├── healthcare_dataset_dashboard.png
│   │   ├── bert_results_confusion_matrix.png
│   │   ├── rating_sentiment_mapping.png
│   │   └── review_length_analysis.png
│   └── bert_results_confusion_matrix.png # Original result
│
└── 📁 Archive
    └── backup/                      # Original project files
```

## 🚀 Training & Usage

### Professional Modular Training 🏗️
```bash
# Enterprise-ready modular approach (recommended)
python train_model.py
```

### Generate Visualizations 📊
```bash
# Comprehensive BERT and dataset analysis
python create_visualizations.py
```

**Modular architecture provides 65.4% accuracy with professional code organization!**

## 🎯 Usage Example

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the trained model
tokenizer = DistilBertTokenizer.from_pretrained('./healthcare_bert_final')
model = DistilBertForSequenceClassification.from_pretrained('./healthcare_bert_final')

# Predict sentiment
text = "This medication helped with my pain but caused some side effects."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
prediction = outputs.logits.argmax().item()  # 0-4 sentiment scale
```

## 💡 Why This Model Works

### ✅ Proven Performance
- **65.4% accuracy** on healthcare drug reviews
- **GPU-optimized**: 8-minute training vs 45 minutes CPU-only
- **Production-ready**: Stable, reliable, and well-tested

### 🎯 Technical Specifications
- **Model**: DistilBERT-base-uncased (66M parameters)
- **Dataset**: 160K+ drug reviews, 5 sentiment classes (0-4)
- **Hardware**: Optimized for NVIDIA GPUs (tested on RTX 4060)
- **Framework**: PyTorch 2.5.1 with CUDA 12.1

### 🤔 Why BERT Instead of GloVe Embeddings?
**Short Answer**: BERT achieves 65.4% vs GloVe's expected 50-55% accuracy.

**Key Differences**:
- **BERT**: Context-aware, pre-trained on massive text, handles medical terms
- **GloVe**: Static embeddings, same vector per word regardless of context
- **Result**: BERT understands "This drug helped my depression but caused nausea" while GloVe gets confused by mixed sentiment

### 🗂️ Why Professional Modular Architecture? 
**Answer**: Enterprise-grade organization with maintainability and scalability!

**Benefits of `healthcare_bert_modules/` Structure**:
- **Team Development**: Multiple developers can work on different modules simultaneously
- **Easy Testing**: Each component can be tested independently with unit tests
- **Flexible Usage**: Mix and match components as needed for different projects
- **Clear Separation**: Data processing, model handling, training, and visualization are cleanly separated
- **Scalability**: Easy to add new models, preprocessing methods, or evaluation metrics
- **Production Ready**: Professional structure suitable for enterprise deployment
- **Maintainable**: Changes in one module don't affect others

**Enterprise Benefits**:
- **`train_model.py`**: Production deployment, team development, enterprise use
- **Modular Components**: Independent development, testing, and deployment
- **Professional Standards**: Follows software engineering best practices

*See `PROJECT_REPORT.md` for detailed technical comparisons and architecture decisions.*

## 📊 Performance Details

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **65.4%** |
| Training Time (GPU) | 8 minutes |
| Training Time (CPU) | 45 minutes |
| Model Size | 256 MB |
| Classes | 5 (Very Negative → Very Positive) |

## 📚 Dependencies

```bash
torch>=2.5.1
transformers>=4.21.0
sklearn>=1.1.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

## 📞 Support & Next Steps

- **Training Issues?** Check CUDA installation and GPU memory (needs 6GB+ VRAM)
- **Want to improve?** See `PROJECT_REPORT.md` for research findings
- **Production deployment?** Model is saved in `healthcare_bert_final/` directory

---

**🏆 Result: Production-ready healthcare sentiment analysis in 8 minutes with GPU acceleration**
