# Indic-TunedLens: Interpreting Multilingual Models in Indian Languages

[![Paper](https://img.shields.io/badge/Paper-ACL%202025-blue)](https://github.com/AnonymousAccountACL/IndicTunedLens)
[![Demo](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/AnonymousAccountACL/IndicTunedLens)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

A novel interpretability framework that extends the Tuned Lens approach to Indian languages by learning language specific affine transformations for better multilingual model understanding using the Sarvam-1 model.

## 🌟 Overview

Most interpretability tools for large language models (LLMs) are designed for English, leaving a significant gap for understanding multilingual models in linguistically diverse regions like India. **Indic-TunedLens** addresses this critical limitation by:

- **Language Specific Analysis**: Separate analysis scripts for Hindi, Marathi, and Bengali
- **Morphological Awareness**: Handling rich morphology and complex linguistic structures of Indian languages  
- **Cross lingual Transfer**: Evaluating interpretability on both seen (Hindi, Marathi) and unseen (Bengali) languages
- **Layer wise Analysis**: Providing insights into semantic encoding across transformer layers using Sarvam-1

## 🔍 Key Findings

- **Training Languages** (Hindi, Marathi): Show early and stable interpretability with peak performance in layers 1-2
- **Unseen Languages** (Bengali): Demonstrate delayed processing with concentrated improvements in final layers
- **Morphological Rich Languages**: Require specialized transformations for effective interpretability
- **Cross lingual Transfer**: Standard English centric methods fail to generalize to Indian languages

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/AnonymousAccountACL/IndicTunedLens.git
cd IndicTunedLens
pip install -r requirements.txt
```

### Prerequisites

1. **Sarvam-1 Model**: Download and place the Sarvam-1 model in your local directory
2. **Trained Lens**: The pre-trained lens should be in `final_sarvam_lens/sarvamai/sarvam-1/`
3. **Data**: MMLU datasets for Hindi, Marathi, and Bengali

### Training the Lens

Train the Indic-TunedLens using the provided training command:

```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=5 \
    -m tuned_lens train \
    sarvamai/sarvam-1 \
    final_combined_data.jsonl \
    --per-gpu-batch-size 1 \
    -o final_sarvam_lens/sarvamai/sarvam-1 \
    --fsdp \
    >> final_sarvam_output.log 2>&1
```

### Data Preparation

Convert Parquet files to JSONL format for training:

```bash
python parquet_to_jsonl.py
```

This combines Hindi, Marathi, and Punjabi training data from the Sangraha dataset.

### Running Analysis

#### Hindi Analysis
```bash
python tuned_lens_hi.py > logs/tuned_lens_hi.log 2>&1
```

#### Marathi Analysis  
```bash
python tuned_lens_mr.py > logs/tuned_lens_mr.log 2>&1
```

#### Bengali Analysis
```bash
python tuned_lens_bn.py > logs/tuned_lens_bn.log 2>&1
```

#### Combined Analysis (Tuned vs Logit Lens)
```bash
python tuned_lens_main.py > logs/tuned_lens_main.log 2>&1
```

## 📁 Project Structure

```
├── README.md                   # This file
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── commands.txt              # Training and execution commands
├── download_dataset.py       # MMLU dataset downloader
├── parquet_to_jsonl.py      # Data format converter
├── snapshot.py              # Model snapshot utility
├── tuned_lens_hi.py         # Hindi analysis script
├── tuned_lens_mr.py         # Marathi analysis script  
├── tuned_lens_bn.py         # Bengali analysis script
├── tuned_lens_main.py       # Combined analysis script
└── final_sarvam_lens/       # Pre-trained lens directory
    └── sarvamai/
        └── sarvam-1/
            └── config.json   # Lens configuration
```

## 🔧 Configuration

### Model Configuration
- **Base Model**: `sarvamai/sarvam-1`
- **Hidden Size**: 2048
- **Layers**: 28
- **Vocabulary Size**: 68,096
- **Context Length**: 8,192 tokens

### Training Configuration
- **Distributed Training**: FSDP enabled
- **Batch Size**: 1 per GPU
- **Nodes**: 1
- **Processes per Node**: 5

### GPU Assignment
- **Hindi**: CUDA:0
- **Marathi**: CUDA:1  
- **Bengali**: CUDA:2
- **Main**: CUDA:0

## 📊 Analysis Output

Each analysis script generates detailed CSV files with:

- **Layer-wise predictions**: Token predictions at each transformer layer
- **Probability rankings**: Top-k token probabilities and rankings
- **Language detection**: Automatic language identification for predicted tokens
- **Position analysis**: Token position effects on interpretability

### Output Structure
```
results/
├── hi/                           # Hindi results
│   └── tunedlens_combined_logits_probabilities.csv
├── mr/                           # Marathi results  
│   └── tunedlens_combined_logits_probabilities.csv
├── bn/                           # Bengali results
│   └── tunedlens_combined_logits_probabilities.csv
```

## 🗂️ Dataset Information

### Training Data [Sangraha Dataset](https://huggingface.co/datasets/ai4bharat/sangraha)
The training combines multiple Parquet files:
- **Hindi**: `hindi0.parquet` to `hindi4.parquet`
- **Marathi**: `marathi0.parquet` to `marathi4.parquet`  
- **Punjabi**: `panjabi0.parquet` to `panjabi4.parquet`

### Evaluation Data [Multilingual MMLU](https://huggingface.co/datasets/alexandrainst/m_mmlu)
- **Hindi**: `m_mmlu_hi.csv`
- **Marathi**: `m_mmlu_mr.csv`
- **Bengali**: `m_mmlu_bn.csv` (unseen language)

<!--- 

## 📝 Citation

If you use Indic-TunedLens in your research, please cite:

```bibtex

```


--->

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🔗 Links**: [Paper](https://github.com/AnonymousAccountACL/IndicTunedLens) | [Demo](https://huggingface.co/spaces/AnonymousAccountACL/IndicTunedLens) | [Sarvam-1 Model](https://huggingface.co/sarvamai/sarvam-1)
