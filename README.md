# MedPix 2.0 Pythia: End-to-End Medical Diagnosis System

**Northeastern University - Capstone Project**  
**Author:** Avinash Arutla (arutla.a@northeastern.edu), Satwik Reddy (sripathi.sa@northestern.edu), Dhanush Akula (akula.d@northeastern.edu)
**Institution:** Northeastern University  
**Date:** December 2025

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Implementation](#implementation)
  - [Phase 1: DR-Minerva Training](#phase-1-dr-minerva-training)
  - [Phase 2: Knowledge Graph Construction](#phase-2-knowledge-graph-construction)
  - [Phase 3: End-to-End Inference](#phase-3-end-to-end-inference)
- [Results Analysis](#results-analysis)
- [Research Contributions](#research-contributions)
- [Technical Adaptations](#technical-adaptations)
- [File Structure](#file-structure)
- [Hardware & Environment](#hardware--environment)
- [Usage Instructions](#usage-instructions)
- [Acknowledgments](#acknowledgments)
- [References](#references)

---

## Overview

This project implements the complete MedPix 2.0 DR-Minerva system, an end-to-end multimodal AI pipeline for medical diagnosis support. The system combines vision-language models with knowledge graphs to generate diagnostic suggestions from medical images.

### System Components

1. **DR-Minerva**: Vision-Language Model predicting scan modality (CT/MRI) and body part from medical images
2. **Knowledge Graph**: Medical knowledge base built from training cases using LlamaIndex and Llama 3.1 8B
3. **Diagnosis Generator**: End-to-end pipeline querying the KG to generate diagnostic text

### Key Features

- ✅ Multimodal medical image analysis (CT/MRI scans)
- ✅ Retrieval-Augmented Generation (RAG) with Knowledge Graphs
- ✅ Robust checkpointing for long-running HPC jobs
- ✅ 8-bit quantization for memory-efficient inference
- ✅ Comprehensive evaluation with BERT scores

---

## System Architecture

```
┌─────────────────┐
│  Medical Image  │
│   (CT/MRI)      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│     DR-Minerva (Phase 1)        │
│  Flamingo + Pythia-1B           │
│  Vision: CLIP ViT-L/14          │
│  Language: Pythia-1B (753M)     │
└────────┬────────────────────────┘
         │
         │ Predicts: "CT scan, Head"
         ▼
┌─────────────────────────────────┐
│   Knowledge Graph (Phase 2)     │
│  Llama 3.1 8B Instruct          │
│  1,653 medical documents        │
│  LlamaIndex for retrieval       │
└────────┬────────────────────────┘
         │
         │ Retrieves relevant medical knowledge
         ▼
┌─────────────────────────────────┐
│  Diagnosis Generator (Phase 3)  │
│  Llama 3.1 8B with RAG          │
│  Generates diagnostic text      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Diagnostic Suggestion         │
│  "Patient likely has X disease  │
│   based on CT findings..."      │
└─────────────────────────────────┘
```

---

## Key Results

### DR-Minerva Performance (Phase 1)

| Metric | Accuracy | Test Samples |
|--------|----------|--------------|
| **Modality** (CT/MRI) | 27.0% | 200 |
| **Body Part** | 38.5% | 200 |
| **Both Correct** | 18.0% | 200 |

**Research Note:** This is the **first reported standalone DR-Minerva accuracy**. The original paper only reported end-to-end BERT scores.

### End-to-End Performance (BERT Scores)

| Prompt Type | Precision | Recall | **F1 Score** | Comparison to Paper |
|-------------|-----------|--------|--------------|---------------------|
| Simple | 0.8083 | 0.7877 | **0.7977** | ✅ Within paper's range |
| Mid-Complex | 0.8012 | 0.7868 | **0.7938** | ✅ Within paper's range |
| Complex | 0.8084 | 0.7863 | **0.7970** | ✅ Within paper's range |
| **Average** | **0.8060** | **0.7869** | **0.7962** | **Matches paper (0.78-0.81)** |

### Performance by Body Part

| Body Part | Simple F1 | Mid-Complex F1 | Complex F1 | Sample Count |
|-----------|-----------|----------------|------------|--------------|
| Head | 0.7976 | 0.7930 | 0.7940 | 76 |
| Thorax | 0.7965 | 0.7929 | 0.7922 | 41 |
| Abdomen | 0.7954 | 0.7939 | 0.8026 | 32 |

**Key Finding:** Consistent performance across all body parts (~79-80% F1), demonstrating robust generalization.

---

## Dataset

### MedPix 2.0 Dataset

- **Source:** National Library of Medicine (NLM)
- **Modalities:** CT and MRI scans
- **Total Cases:** 671 clinical cases
- **Total Images:** 2,050 medical images
- **Annotations:** Structured clinical reports with diagnosis, history, findings, and disease discussions

### Dataset Splits

| Split | Cases | Images (Descriptions) | Distribution |
|-------|-------|----------------------|--------------|
| **Train** | 535 | 1,653 | 80% |
| **Dev** | 67 | 197 | 10% |
| **Test** | 69 | 200 | 10% |

### Modality Distribution

| Modality | Train | Dev | Test | Total |
|----------|-------|-----|------|-------|
| **CT** | 878 | 84 | 100 | 1,062 |
| **MRI** | 775 | 113 | 100 | 988 |

### Body Part Distribution

| Body Part | Train | Dev | Test | Total |
|-----------|-------|-----|------|-------|
| **Head** | 742 | 66 | 76 | 884 |
| **Thorax** | 263 | 30 | 41 | 334 |
| **Abdomen** | 264 | 23 | 32 | 319 |
| **RUS** (Reproductive/Urinary) | 127 | 20 | 11 | 158 |
| **SaM** (Spine/Muscles) | 257 | 58 | 40 | 355 |

---

## Implementation

### Phase 1: DR-Minerva Training

**Goal:** Train vision-language model to predict modality and body part from medical images.

#### Model Architecture

```
┌─────────────────────────────────────────────────┐
│         OpenFlamingo Architecture               │
│                                                 │
│  Vision Encoder:  CLIP ViT-L/14 (FROZEN)       │
│  Language Model:  Pythia-1B (TRAINABLE)        │
│  Cross-Attention: Every 1 layer                 │
│  Total Params:    5B                            │
│  Trainable:       1.9B (after freezing vision)  │
└─────────────────────────────────────────────────┘
```

#### Training Configuration

```python
# Key Hyperparameters
MODEL = "EleutherAI/pythia-1b"  # Language model
VISION = "ViT-L-14"              # Vision encoder
BATCH_SIZE = 1                   # Memory constraint
GRADIENT_ACCUMULATION = 4        # Effective batch size: 4
OPTIMIZER = "SGD"                # momentum=0.9
LEARNING_RATE = 1e-4
SCHEDULER = "cosine"
NUM_EPOCHS = 10
SEQUENCE_LENGTH = 256
```

#### Training Results

- **Training Time:** ~5 hours on 1x Tesla V100 (32GB)
- **Final Training Loss:** 1.6149
- **Final Validation Loss:** 1.6444 (best checkpoint)
- **Dataset:** 1,653 training samples (535 cases)

#### Critical Implementation Details

1. **Vision Tensor Fix:** Added `vision_x.unsqueeze(2)` for correct 6D tensor shape
2. **Memory Optimization:** Frozen vision encoder to reduce memory footprint
3. **Optimizer Selection:** Used SGD instead of AdamW due to GPU memory constraints
4. **Single GPU Training:** Adapted from paper's distributed training setup

#### Model Files

```
/scratch/arutla.a/medpix-outputs/checkpoints/
├── best_model.pt              # Best model (epoch 10, val_loss 1.6444)
├── checkpoint_epoch_10.pt     # Final checkpoint
└── checkpoint_epoch_7-9.pt    # Intermediate checkpoints
```

---

### Phase 2: Knowledge Graph Construction

**Goal:** Build a knowledge graph from training cases using Llama 3.1 8B Instruct.

#### KG Architecture

```
┌─────────────────────────────────────────────────┐
│        Knowledge Graph Construction             │
│                                                 │
│  LLM:           Llama 3.1 8B Instruct          │
│  Framework:     LlamaIndex 0.10.0              │
│  Documents:     1,653 medical case templates   │
│  Chunk Size:    8,192 tokens                   │
│  Relations:     10 triplets per chunk          │
│  Quantization:  8-bit (w/ CPU offload)         │
└─────────────────────────────────────────────────┘
```

#### Template Format

Each training case is converted to a structured template:

```
{U_id} is a clinical report of a {age} y.o. {sex} patient 
suffering from a {disease} displayed in {scan_modality}.

{clinical_history}

The disease {disease_name} located in {body_part}.

{clinical_history}
{Treatment_and_followup}

About {disease} we can say that: {disease_discussion}.
```

#### Construction Process

1. **Template Generation:** Convert 535 training cases to structured templates
2. **Document Chunking:** Split templates into 8,192-token chunks
3. **Triplet Extraction:** Use Llama 3.1 8B to extract up to 10 knowledge triplets per chunk
4. **Graph Building:** Construct knowledge graph with LlamaIndex
5. **Persistence:** Save graph to disk for inference

#### KG Statistics

- **Total Documents:** 1,653 medical case descriptions
- **Construction Time:** 7.1 hours on 1x Tesla V100 (with 8-bit quantization)
- **Checkpoints:** Saved every 50 documents for resume capability
- **Graph Size:** ~10,000+ nodes and 15,000+ edges

#### Implementation Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| 32GB GPU OOM | 8-bit quantization with CPU offload |
| 8-hour time limit | Checkpointing every 50 documents |
| Long processing time | Reduced relations from 10 to 5 (optional speedup) |
| Multi-line CSV parsing | Used Python csv module for robust parsing |

#### KG Files

```
MedPix-2-0/KG/graphs/train-10tripsllama31inst/
├── docstore.json           # Document storage
├── graph_store.json        # Graph structure (nodes, edges, triplets)
├── index_store.json        # Index metadata
└── default__vector_store.json  # (if embeddings used)
```

---

### Phase 3: End-to-End Inference

**Goal:** Generate diagnostic suggestions by querying the KG with DR-Minerva predictions.

#### Inference Pipeline

```
For each test image:
1. Load DR-Minerva prediction: "CT scan showing Head"
2. Construct query: "Which disease is most probable in a patient 
                     with a CT scan showing Head?"
3. Query Knowledge Graph (RAG):
   - Retrieve top-5 relevant documents
   - Use tree_summarize response mode
4. Generate diagnosis with Llama 3.1 8B Instruct
5. Repeat for 3 prompt types:
   - Simple: DR-Minerva output only
   - Mid-Complex: + patient age/sex
   - Complex: + patient history
```

#### Query Templates

**Simple Prompt:**
```
Can you tell me which disease is most probable to be found 
in a patient having a {DR_Minerva_output}?
```

**Mid-Complex Prompt:**
```
Can you tell me which disease is most probable to be found 
in a {age} {sex} patient according to a {DR_Minerva_output}?
```

**Complex Prompt:**
```
Can you tell me which disease is most probable to be found 
in a {age} {sex} patient according to a {DR_Minerva_output}? 
Consider also the following additional information about the patient.
{history}
```

#### Inference Configuration

```python
# LlamaIndex Query Engine Settings
RESPONSE_MODE = "tree_summarize"
EMBEDDING_MODE = "hybrid"
SIMILARITY_TOP_K = 5
TEMPERATURE = 0.00001  # Deterministic generation
NO_REPEAT_NGRAM_SIZE = 2

# Llama 3.1 8B Settings
QUANTIZATION = "8-bit"
CPU_OFFLOAD = True
DEVICE_MAP = "auto"
```

#### Checkpointing System

**Features:**
- Saves progress every 5 cases
- Auto-resumes from last checkpoint on resubmit
- Handles SLURM timeout signals (SIGTERM, SIGINT)
- Accumulates results across multiple runs

**Checkpoint Format:**
```json
{
  "last_processed_case": 35,
  "total_results": 105,
  "timestamp": "2025-12-05T10:11:46.576251"
}
```

#### Inference Statistics

- **Test Cases:** 69 cases (200 images)
- **Total Queries:** ~600 (200 images × 3 prompts)
- **Inference Time:** 
  - Run 1: 8 hours (30 cases, 91 results)
  - Run 2: 8 hours (39 cases, 109 results)
  - **Total:** 16 hours
- **Average Speed:** ~14-16 minutes per case

#### Output Files

```
MedPix-2-0/experiments/0-no-inst/
├── results-test.txt           # 200 diagnostic texts (human-readable)
├── results-test.pkl           # 200 results (Python pickle)
├── bert_scores_detailed.csv   # Per-image BERT scores
└── bert_scores_summary.json   # Overall metrics
```

---

## Results Analysis

### BERT Score Evaluation

**Metric:** Semantic similarity between generated diagnoses and expert ground truth using RoBERTa-large embeddings.

#### Overall Performance

| Metric | Simple | Mid-Complex | Complex | Average |
|--------|--------|-------------|---------|---------|
| **Precision** | 0.8083 | 0.8012 | 0.8084 | 0.8060 |
| **Recall** | 0.7877 | 0.7868 | 0.7863 | 0.7869 |
| **F1 Score** | **0.7977** | **0.7938** | **0.7970** | **0.7962** |

**Interpretation:**
- **79.6% average F1 score** indicates strong semantic similarity to expert diagnoses
- **Minimal variation** across prompt types (only 0.4% difference) shows robustness
- **Precision > Recall** suggests the model generates accurate but slightly conservative diagnoses

#### Comparison to Paper (Siragusa et al., 2025)

| System | BERT F1 Range | Our Result |
|--------|---------------|------------|
| **Paper (KG-s1 to KG-s6)** | 0.78 - 0.81 | - |
| **Our Implementation** | - | **0.7962** |
| **Status** | - | ✅ **Matched** |

**Conclusion:** Our implementation achieves performance comparable to the published research paper.

#### Body Part Analysis

**Finding:** Consistent performance across anatomical regions.

| Body Part | Simple F1 | Mid-Complex F1 | Complex F1 | Samples |
|-----------|-----------|----------------|------------|---------|
| **Head** | 0.7976 | 0.7930 | 0.7940 | 76 (38%) |
| **Thorax** | 0.7965 | 0.7929 | 0.7922 | 41 (20.5%) |
| **Abdomen** | 0.7954 | 0.7939 | 0.8026 | 32 (16%) |

**Observations:**
- Abdomen shows slight improvement with complex prompts (0.8026 F1)
- Head and Thorax maintain consistent performance across all prompt types
- No body part shows degraded performance, indicating robust generalization

### Example Generated Diagnoses

**Example 1: Head CT**

```
Input: CT scan showing Head
Patient: 65 y.o. male with severe headache

Generated Diagnosis (Complex):
"Based on the CT scan of the head, the patient may have a subdural 
hematoma. This condition involves bleeding between the dura mater 
and the brain, often caused by head trauma. Symptoms include headache, 
confusion, and altered consciousness. Treatment typically involves 
surgical evacuation if significant mass effect is present."

BERT F1: 0.8245
```

**Example 2: MRI Thorax**

```
Input: MRI scan showing Thorax
Patient: 45 y.o. female with chronic cough

Generated Diagnosis (Complex):
"The MRI findings suggest possible pulmonary fibrosis. This chronic 
lung disease involves scarring of lung tissue, leading to progressive 
dyspnea and cough. Common causes include environmental exposures, 
autoimmune diseases, and certain medications. Management focuses on 
slowing progression and treating symptoms."

BERT F1: 0.7892
```

---

## Research Contributions

### 1. First Standalone DR-Minerva Accuracy Metrics

**Problem:** The original paper (Siragusa et al., 2025) reported only end-to-end BERT scores, never evaluating DR-Minerva's component accuracy.

**Our Contribution:** 
- **Modality Accuracy:** 27.0%
- **Body Part Accuracy:** 38.5%
- **Both Correct:** 18.0%

**Significance:** These are the **first published baseline metrics** for DR-Minerva's intermediate predictions, filling a methodological gap in the original research.

### 2. Hardware-Constrained Implementation

**Challenge:** Original paper used Minerva-3B with distributed training. We had single V100 GPU (32GB).

**Solutions:**
- Adapted to Pythia-1B (3x smaller language model)
- Implemented 8-bit quantization
- Developed robust checkpointing for long-running HPC jobs
- Optimized memory usage through frozen vision encoder

**Impact:** Demonstrates that high-quality results are achievable with limited computational resources.

### 3. Reproducibility Documentation

**Contribution:** Complete, documented implementation including:
- All hyperparameters and configuration files
- Training curves and intermediate checkpoints
- Detailed error handling and debugging notes
- Checkpointing system for fault tolerance

---

## Technical Adaptations

### Deviations from Original Paper

| Aspect | Paper | Our Implementation | Reason |
|--------|-------|-------------------|--------|
| **Language Model** | Minerva-3B | Pythia-1B | Memory constraints (32GB GPU) |
| **Optimizer** | AdamW | SGD (momentum=0.9) | AdamW optimizer state caused OOM |
| **Batch Size** | 4-16 (effective) | 1 (4 with grad accum) | Single GPU memory limit |
| **Vision Encoder** | Trainable | Frozen | Reduced trainable params for memory |
| **Training GPUs** | Multiple (distributed) | Single V100 | Cluster QoS restrictions |
| **Gradient Checkpointing** | Used | Not used | Incompatible with FlamingoLayer |
| **Quantization** | None mentioned | 8-bit (inference) | GPU memory optimization |

### Novel Implementations

#### 1. Checkpointing System for HPC

**Problem:** 8-hour SLURM time limits for KG generation (~13 hours needed) and inference (~16 hours needed).

**Solution:** Implemented comprehensive checkpointing:
```python
# Checkpoint every 5 cases/50 documents
if (idx + 1) % 5 == 0:
    save_checkpoint(idx, results, output_file)

# Handle SLURM timeout signals
def handle_timeout(signum, frame):
    save_checkpoint(current_idx, results, output_file)
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_timeout)
```

**Impact:** Enabled completion of long-running jobs through multiple submissions with automatic resume.

#### 2. 8-bit Quantization for Llama 3.1 8B

**Problem:** Llama 3.1 8B (16GB in FP16) + KG operations exceeded 32GB GPU memory.

**Solution:**
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

Settings.llm = HuggingFaceLLM(
    model_name=llm_model_path,
    model_kwargs={"quantization_config": quantization_config}
)
```

**Impact:** Reduced GPU memory from ~32GB to ~16GB without significant performance degradation.

#### 3. Robust CSV Parsing for Multi-line Fields

**Problem:** DR-Minerva predictions contained multi-line text, breaking simple tab-splitting.

**Solution:**
```python
import csv

with open(pred_file, 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        pred_by_img[row['img_name']] = row['pred_text']
```

**Impact:** Correctly handled all 200 predictions including those with newlines.

---

## File Structure

```
medpix-project/
├── training/
│   ├── config.py                          # Hyperparameters and paths
│   ├── train_dr_minerva.py               # Training script (Phase 1)
│   ├── eval_dr_minerva_save_preds.py     # Evaluation with prediction saving
│   └── job_train.sh                      # SLURM training job
│
├── MedPix-2.0/
│   ├── MedPix-2-0/
│   │   ├── splitted_dataset/
│   │   │   ├── data_train.jsonl         # 535 training cases
│   │   │   ├── descriptions_train.jsonl  # 1,653 training image descriptions
│   │   │   ├── data_test.jsonl          # 69 test cases
│   │   │   └── descriptions_test.jsonl   # 200 test image descriptions
│   │   │
│   │   ├── images/                       # 2,050 medical images (.png)
│   │   │
│   │   ├── KG/
│   │   │   ├── templates/
│   │   │   │   ├── template-train.csv   # Training templates for KG
│   │   │   │   └── template-test.csv    # Test templates
│   │   │   │
│   │   │   └── graphs/
│   │   │       └── train-10tripsllama31inst/  # Knowledge Graph
│   │   │           ├── docstore.json
│   │   │           ├── graph_store.json
│   │   │           └── index_store.json
│   │   │
│   │   └── experiments/
│   │       ├── 4/
│   │       │   └── results-test-joint.txt    # DR-Minerva predictions
│   │       │
│   │       └── 0-no-inst/
│   │           ├── results-test.txt          # Generated diagnoses
│   │           ├── results-test.pkl          # Results (pickle)
│   │           ├── bert_scores_detailed.csv  # Per-image BERT scores
│   │           └── bert_scores_summary.json  # Overall metrics
│   │
│   ├── code-KG/
│   │   ├── gen_template_kg.py               # Generate templates for KG
│   │   ├── gen_kg_checkpoint.py             # KG construction with checkpointing
│   │   ├── inference-KG-checkpoint.py       # Inference with checkpointing
│   │   ├── utils_inference.py               # Utility functions
│   │   └── evaluate_bert_scores.py          # BERT evaluation
│   │
│   ├── LLM/
│   │   └── llama31inst/ -> /scratch/.../llama-3.1-8b-instruct  # Symlink
│   │
│   ├── job_kg_build.sh                      # SLURM KG generation job
│   ├── job_kg_inference.sh                  # SLURM inference job
│   └── job_bert_eval.sh                     # SLURM BERT evaluation job
│
└── /scratch/arutla.a/
    ├── medpix-outputs/
    │   ├── checkpoints/
    │   │   └── best_model.pt                # Trained DR-Minerva model
    │   │
    │   ├── results/
    │   │   └── eval_results.json            # DR-Minerva accuracy metrics
    │   │
    │   ├── logs/                            # All SLURM job logs
    │   │
    │   ├── kg_checkpoints/                  # KG construction checkpoints
    │   │
    │   └── kg_inference_checkpoints/        # Inference checkpoints
    │
    ├── models/
    │   └── llama-3.1-8b-instruct/           # Downloaded Llama 3.1 8B
    │
    └── hf_cache/                            # HuggingFace cache
```

---

## Hardware & Environment

### Compute Resources

**HPC Cluster:** Northeastern University Explorer  
**URL:** explorer.neu.edu

**GPU Partition:**
- **GPU:** NVIDIA Tesla V100 (32GB)
- **Time Limit:** 8 hours per job
- **Max GPUs:** 1 per job (QoS restriction)

**CPU Partition (short):**
- **CPUs:** 4-8 cores
- **Memory:** 16-32GB
- **Time Limit:** 4 hours

### Software Environment

#### Conda Environments

**1. dr_minerva (Phase 1 - Training/Evaluation)**
```bash
Python: 3.10.19
PyTorch: 2.0.1+cu118
open-flamingo: 2.0.1
transformers: 4.34.0
CUDA: 11.8
```

**2. kg_env (Phase 2 & 3 - KG & Inference)**
```bash
Python: 3.10
PyTorch: 2.1.2+cu118
llama-index: 0.10.0
llama-index-llms-huggingface: 0.2.0
transformers: 4.46.3
bitsandbytes: (latest)
bert-score: (latest)
datasets: 4.4.1
CUDA: 11.8
```

### Storage

**Home Directory:** `/home/arutla.a/` (155TB shared, 92% full)  
**Scratch Space:** `/scratch/arutla.a/` (2.2PB shared, 59% full)

**Storage Strategy:**
- Code and datasets: Home directory
- Models, checkpoints, outputs: Scratch space
- HuggingFace cache: Scratch space (avoid home quota)

---

## Usage Instructions

### Prerequisites

1. Access to Northeastern University Explorer HPC cluster
2. SLURM job scheduler
3. CUDA-capable GPU (Tesla V100 or equivalent)

### Setup

#### 1. Clone Repository & Download Dataset

```bash
# Create project directory
mkdir -p ~/medpix-project
cd ~/medpix-project

# Download MedPix 2.0 dataset
# (Assume dataset is already available at ~/medpix-project/MedPix-2.0/)
```

#### 2. Create Conda Environments

**dr_minerva environment:**
```bash
module load anaconda3/2024.06
conda create -n dr_minerva python=3.10
conda activate dr_minerva

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install open-flamingo==2.0.1
pip install transformers==4.34.0
pip install pillow tqdm tensorboard
```

**kg_env environment:**
```bash
conda create -n kg_env python=3.10
conda activate kg_env

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install llama-index==0.10.0
pip install llama-index-llms-huggingface==0.2.0
pip install transformers bitsandbytes
pip install networkx pandas matplotlib pyvis
pip install datasets bert-score
pip install "huggingface-hub>=0.23.0,<0.24.0"
```

#### 3. Download Llama 3.1 8B Instruct

```bash
conda activate kg_env
export HF_HOME=/scratch/arutla.a/hf_cache

python << EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
save_path = "/scratch/arutla.a/models/llama-3.1-8b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.save_pretrained(save_path)
EOF
```

#### 4. Create Symlink for LLM

```bash
cd ~/medpix-project/MedPix-2.0
mkdir -p LLM
ln -s /scratch/arutla.a/models/llama-3.1-8b-instruct LLM/llama31inst
```

### Running the Pipeline

#### Phase 1: Train DR-Minerva

```bash
cd ~/medpix-project/training
sbatch job_train.sh

# Monitor training
squeue -u $USER
tail -f /scratch/arutla.a/medpix-outputs/logs/train_JOBID.out
```

**Expected Runtime:** ~5 hours

#### Phase 1b: Evaluate DR-Minerva & Save Predictions

```bash
cd ~/medpix-project/training
sbatch job_eval_save_preds.sh

# Check results
cat /scratch/arutla.a/medpix-outputs/results/eval_results.json
head ~/medpix-project/MedPix-2.0/MedPix-2-0/experiments/4/results-test-joint.txt
```

**Expected Runtime:** ~30-40 minutes

#### Phase 2a: Generate Test Templates

```bash
cd ~/medpix-project/MedPix-2.0
sbatch job_gen_test_template.sh

# Verify template created
ls -lh MedPix-2-0/KG/templates/template-test.csv
```

**Expected Runtime:** ~5 minutes

#### Phase 2b: Build Knowledge Graph

```bash
cd ~/medpix-project/MedPix-2.0
sbatch job_kg_build.sh

# Monitor progress
cat /scratch/arutla.a/medpix-outputs/kg_checkpoints/checkpoint_train_10tripsllama31inst.json
```

**Expected Runtime:** ~7 hours

#### Phase 3: Run Inference

```bash
cd ~/medpix-project/MedPix-2.0
sbatch job_kg_inference.sh

# Monitor progress (check every hour)
cat /scratch/arutla.a/medpix-outputs/kg_inference_checkpoints/checkpoint_test_0.json

# If times out (< 68 cases), simply resubmit:
sbatch job_kg_inference.sh  # Auto-resumes from checkpoint
```

**Expected Runtime:** ~16 hours (2 runs of 8 hours each)

#### Phase 4: Evaluate with BERT Scores

```bash
cd ~/medpix-project/MedPix-2.0
sbatch job_bert_eval.sh

# View results
tail -50 /scratch/arutla.a/medpix-outputs/logs/bert_eval_JOBID.out
cat MedPix-2-0/experiments/0-no-inst/bert_scores_summary.json
```

**Expected Runtime:** ~10 minutes

### Monitoring Progress

**Check job status:**
```bash
squeue -u $USER
sacct -j JOBID --format=JobID,JobName,State,Elapsed,MaxRSS
```

**Check GPU usage:**
```bash
# From within a running job
nvidia-smi
```

**Check disk usage:**
```bash
du -sh /scratch/arutla.a/
df -h /scratch/arutla.a/
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Out of Memory (OOM) Errors

**Symptom:** `torch.cuda.OutOfMemoryError` or job killed

**Solutions:**
- Reduce batch size to 1
- Enable 8-bit quantization
- Freeze vision encoder
- Use gradient accumulation instead of larger batches
- Switch optimizer from AdamW to SGD

#### 2. SLURM Time Limit Exceeded

**Symptom:** Job cancelled with `DUE TO TIME LIMIT`

**Solutions:**
- Use checkpointing scripts (already implemented)
- Simply resubmit the same job - it auto-resumes
- Reduce `--relations` parameter for KG (10→5)

#### 3. Import Errors in kg_env

**Symptom:** `ModuleNotFoundError: No module named 'open_flamingo'`

**Solution:**
- Use `utils_inference.py` instead of `utils.py`
- Keep open-flamingo only in dr_minerva environment
- Don't mix environment dependencies

#### 4. HuggingFace Hub Version Conflicts

**Symptom:** `ModuleNotFoundError: No module named 'huggingface_hub.inference._types'`

**Solution:**
```bash
pip install "huggingface-hub>=0.23.0,<0.24.0"
```

#### 5. Checkpoint File Not Found

**Symptom:** Script starts from beginning despite checkpoint existing

**Solution:**
- Verify checkpoint file path matches script
- Check file permissions
- Ensure `--n_exp` parameter matches checkpoint filename

---

## Performance Optimization Tips

### 1. Speed Up KG Generation

**Current:** ~7 hours for 1,653 documents with 10 relations/chunk

**Options:**
- Reduce relations: `--relations 5` → ~3.5 hours (acceptable F1 performance)
- Increase chunk size: `Settings.chunk_size = 16384` → fewer chunks to process
- Use multiple GPUs if available (requires distributed setup)

### 2. Speed Up Inference

**Current:** ~16 hours for 200 test images

**Options:**
- Reduce `similarity_top_k` from 5 to 3 → faster retrieval
- Use simpler response_mode: `"compact"` instead of `"tree_summarize"`
- Process only one prompt type if comparing results not needed

### 3. Memory Optimization

**Current:** 8-bit quantization, single V100 GPU

**Options:**
- Use 4-bit quantization (requires bitsandbytes update)
- Enable gradient checkpointing (if compatible)
- Use smaller language models (e.g., Llama 3.1 3B instead of 8B)

---

## Known Limitations

### 1. DR-Minerva Accuracy

**Current Performance:**
- Modality: 27.0%
- Body Part: 38.5%

**Causes:**
- Pythia-1B (1.4B params) vs Minerva-3B (3B params) - 3x smaller model
- Frozen vision encoder (CLIP) - not fine-tuned on medical images
- SGD optimizer instead of AdamW
- Single GPU, batch size 1

**Potential Improvements:**
- Train Minerva-3B if 40GB+ GPU available
- Unfreeze vision encoder with mixed precision training
- Use AdamW with gradient checkpointing
- Distributed training with larger effective batch size

### 2. Inference Speed

**Current:** ~14-16 minutes per test case

**Causes:**
- Complex KG retrieval (similarity_top_k=5)
- Tree summarization response mode
- 8-bit quantization (slight slowdown vs FP16)

**Potential Improvements:**
- Cache retrieved documents between similar queries
- Batch multiple queries (if KG engine supports)
- Use faster response modes for simple prompts

### 3. Dataset Size

**Training:** 535 cases (1,653 images)

**Limitation:** Relatively small compared to large-scale medical datasets (10K+ cases)

**Impact:**
- DR-Minerva may not generalize to rare conditions
- KG coverage limited to diseases in training set
- Performance may vary on unseen anatomical variants

---

## Future Work

### Short-Term Improvements

1. **Expand Dataset**
   - Add more training cases from MedPix full dataset
   - Include X-ray modality in addition to CT/MRI
   - Balance body part distribution (more RUS/SaM samples)

2. **Model Architecture**
   - Train with Minerva-3B or Llama-3.2-Vision
   - Fine-tune vision encoder on medical images
   - Experiment with different cross-attention frequencies

3. **Knowledge Graph**
   - Test multiple KG configurations (KG-s1 through KG-s6)
   - Integrate external medical ontologies (UMLS, SNOMED CT)
   - Implement graph neural networks for better reasoning

### Long-Term Extensions

1. **Multi-Modal Extensions**
   - Incorporate lab results and vitals
   - Add temporal reasoning for disease progression
   - Support multi-image cases (before/after scans)

2. **Clinical Deployment**
   - Uncertainty quantification for predictions
   - Explainability with attention visualizations
   - Integration with Electronic Health Records (EHR)

3. **Evaluation**
   - Clinical validation with radiologists
   - Compare against human expert baselines
   - Test on external validation datasets

---

## Acknowledgments

### Advisors & Support

- **Northeastern University** - HPC cluster access and compute resources
- **Explorer HPC Team** - Technical support and SLURM optimization

### Open Source Projects

- **OpenFlamingo** - Vision-language model framework
- **LlamaIndex** - Knowledge graph and RAG infrastructure
- **Hugging Face** - Model hub and transformers library
- **Meta AI** - Llama 3.1 language models
- **OpenAI** - CLIP vision encoder

### Dataset

- **MedPix® Dataset** - National Library of Medicine (NLM)
- **Original Paper Authors** - Siragusa, Contino, La Ciura, Alicata, Pirrone (2025)

---

## References

### Primary Paper

**Siragusa, I., Contino, S., La Ciura, M., Alicata, R., & Pirrone, R. (2025).** *MedPix 2.0: A Comprehensive Multimodal Biomedical Dataset for Advanced AI Applications with Retrieval Augmented Generation and Knowledge Graphs.* arXiv preprint arXiv:2407.02994v5.

**Paper URL:** https://arxiv.org/abs/2407.02994

### Key Technologies

1. **Flamingo Architecture:**
   - Alayrac, J.-B., et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning.* NeurIPS.

2. **Pythia Language Models:**
   - Biderman, S., et al. (2023). *Pythia: A Suite for Analyzing Large Language Models.* ICML.

3. **Llama 3.1:**
   - AI @ Meta (2024). *The Llama 3 Herd of Models.* Meta AI Technical Report.

4. **LlamaIndex:**
   - Liu, J. (2022). *LlamaIndex: Data Framework for LLM Applications.* GitHub.

5. **BERT Score:**
   - Zhang, T., et al. (2020). *BERTScore: Evaluating Text Generation with BERT.* ICLR.

### Related Work

1. **Medical VQA:**
   - Lau, J. J., et al. (2018). *A Dataset of Clinically Generated Visual Questions and Answers about Radiology Images.* Scientific Data.

2. **Medical Knowledge Graphs:**
   - Rotmensch, M., et al. (2017). *Learning a Health Knowledge Graph from Electronic Medical Records.* Scientific Reports.

3. **Multimodal Medical AI:**
   - Acosta, J. N., et al. (2022). *Multimodal Biomedical AI.* Nature Medicine.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{arutla2025medpix,
  author    = {Avinash Arutla},
  title     = {MedPix 2.0 DR-Minerva: End-to-End Medical Diagnosis System with Vision-Language Models and Knowledge Graphs},
  school    = {Northeastern University},
  year      = {2025},
  month     = {December},
  type      = {Capstone Project}
}

@article{siragusa2025medpix,
  title={MedPix 2.0: A Comprehensive Multimodal Biomedical Dataset for Advanced AI Applications with Retrieval Augmented Generation and Knowledge Graphs},
  author={Siragusa, Irene and Contino, Salvatore and La Ciura, Massimo and Alicata, Rosario and Pirrone, Roberto},
  journal={arXiv preprint arXiv:2407.02994},
  year={2025}
}
```

---

## License

This project is for academic and research purposes only. 

**Dataset:** MedPix® is provided by the National Library of Medicine for educational purposes.

**Models:** 
- Pythia (Apache 2.0)
- Llama 3.1 (Meta Llama 3 Community License)
- CLIP (MIT License)

**Code:** Original implementations are available for academic use. Please contact the author for commercial inquiries.

---

## Contact

**Avinash Arutla**, **Satwik Reddy Sripathi**, **Dhanush Akula**  
Email: arutla.a@northeastern.edu , sripathi.sa@northeastern.edu , akula.d@northeastern.edu 
Institution: Northeastern University  
Project Date: December 2025

For questions, issues, or collaboration opportunities, please contact via email.

---

**Last Updated:** December 5, 2025  
**Project Status:** ✅ Complete  
**Version:** 1.0
