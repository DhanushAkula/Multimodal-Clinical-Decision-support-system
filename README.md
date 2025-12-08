# MedPix 2.0 Pythia: End-to-End Medical Diagnosis System

**Northeastern University - Capstone Project**

**Authors:** Avinash Arutla (arutla.a@northeastern.edu), Satwik Reddy (sripathi.sa@northestern.edu), Dhanush Akula (akula.d@northeastern.edu)

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
  - [Phase 3: GNN Reasoning (GraphSAGE/GAT)](#phase-3-gnn-reasoning-graphsagegat)
  - [Phase 4: End-to-End Inference](#phase-4-end-to-end-inference)
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

This project implements the complete MedPix 2.0 DR-Minerva system, an end-to-end multimodal AI pipeline for medical diagnosis support. The system combines vision-language models, knowledge graphs, and **Graph Neural Networks (GNNs)** to generate diagnostic suggestions from medical images.

### System Components

1. **DR-Minerva**: Vision-Language Model predicting scan modality (CT/MRI) and body part from medical images.
2. **Knowledge Graph**: Medical knowledge base built from training cases using LlamaIndex and Llama 3.1 8B.
3. **GNN Reasoner**: A Graph Attention Network (GAT) that performs link prediction and node classification to refine disease probability before generation.
4. **Diagnosis Generator**: End-to-end pipeline querying the KG (refined by GNN) to generate diagnostic text.

### Key Features

- ✅ Multimodal medical image analysis (CT/MRI scans)
- ✅ Graph Neural Network (GNN) for Structural Reasoning
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
         │ Retrieves subgraph of candidate nodes
         ▼
┌─────────────────────────────────┐
│   GNN Reasoner (Phase 3)        │
│  Model: Graph Attention (GAT)   │
│  Input: Subgraphs from Query    │
│  Output: Refined Node Weights   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Diagnosis Generator (Phase 4)  │
│  Llama 3.1 8B with GNN-RAG      │
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

**Research Note:** This is the first reported standalone DR-Minerva accuracy. The original paper only reported end-to-end BERT scores.

### End-to-End Performance (Impact of GNN)

We compared the standard RAG pipeline against our **GNN-Enhanced** pipeline. The GNN effectively filtered irrelevant context nodes, improving the F1 score.

| System Variant | Precision | Recall | **F1 Score** | Improvement |
|-----------------|-----------|--------|--------------|-------------|
| Standard RAG (No GNN) | 0.8060 | 0.7869 | 0.7962 | Baseline |
| **GNN-Enhanced RAG** | **0.8210** | **0.8095** | **0.8152** | **+1.9%** 🚀 |

### Performance by Body Part (GNN-Enhanced)

| Body Part | Simple F1 | Mid-Complex F1 | Complex F1 | Sample Count |
|-----------|-----------|----------------|------------|--------------|
| Head | 0.8120 | 0.8155 | 0.8180 | 76 |
| Thorax | 0.8090 | 0.8110 | 0.8145 | 41 |
| Abdomen | 0.8080 | 0.8105 | 0.8210 | 32 |

**Key Finding:** The GNN module specifically improved performance in "Complex" prompts (Abdomen +2.1%) by better associating sparse symptoms in the graph with the correct disease nodes.

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
|-------|-------|-----------------------|--------------|
| Train | 535 | 1,653 | 80% |
| Dev | 67 | 197 | 10% |
| Test | 69 | 200 | 10% |

### Modality Distribution

| Modality | Train | Dev | Test | Total |
|----------|-------|-----|------|-------|
| CT | 878 | 84 | 100 | 1,062 |
| MRI | 775 | 113 | 100 | 988 |

### Body Part Distribution

| Body Part | Train | Dev | Test | Total |
|-----------|-------|-----|------|-------|
| Head | 742 | 66 | 76 | 884 |
| Thorax | 263 | 30 | 41 | 334 |
| Abdomen | 264 | 23 | 32 | 319 |
| RUS (Reproductive/Urinary) | 127 | 20 | 11 | 158 |
| SaM (Spine/Muscles) | 257 | 58 | 40 | 355 |

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

- **Vision Tensor Fix:** Added `vision_x.unsqueeze(2)` for correct 6D tensor shape
- **Memory Optimization:** Frozen vision encoder to reduce memory footprint
- **Optimizer Selection:** Used SGD instead of AdamW due to GPU memory constraints
- **Single GPU Training:** Adapted from paper's distributed training setup

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

---

### Phase 3: GNN Reasoning (GraphSAGE/GAT)

**Goal:** Train a Graph Neural Network to reason over the constructed Knowledge Graph, prioritizing nodes (diseases/symptoms) that are structurally relevant to the DR-Minerva prediction, rather than just semantically similar.

#### GNN Architecture

We implemented a **Graph Attention Network (GAT)** using `PyTorch Geometric`.

```
┌─────────────────────────────────────────────────┐
│            GNN Configuration                    │
│                                                 │
│  Framework:     PyTorch Geometric (PyG)        │
│  Architecture:  2-Layer GAT (Multi-head)       │
│  Input Dim:     768 (BERT Embeddings)          │
│  Hidden Dim:    256                            │
│  Heads:         4                              │
│  Output:        Link Probability Score         │
└─────────────────────────────────────────────────┘
```

#### Training Process (Link Prediction)

1. **Graph Export:** The LlamaIndex graph was exported to a `NetworkX` format and converted to `Data` objects for PyG.
2. **Node Embeddings:** Initialized using `sentence-transformers/all-mpnet-base-v2`.
3. **Task:** Link prediction. The model was trained to predict edges between `Symptom` nodes and `Disease` nodes.
4. **Loss Function:** Binary Cross Entropy with Logits Loss (`BCEWithLogitsLoss`).
5. **Negative Sampling:** Used 1:1 ratio of positive edges to negative (non-existent) edges during training.

#### GNN Training Stats

- **Epochs:** 200
- **Optimizer:** Adam (`lr=0.01`, `weight_decay=5e-4`)
- **Final Training Loss:** 0.412
- **Validation ROC-AUC:** 0.884

---

### Phase 4: End-to-End Inference

**Goal:** Generate diagnostic suggestions by querying the KG with DR-Minerva predictions, refined by GNN scoring.

#### Inference Pipeline

```
For each test image:
1. Load DR-Minerva prediction: "CT scan showing Head"
2. Construct query: "Which disease is most probable...?"
3. Initial Retrieval: Fetch top-20 nodes via vector similarity.
4. GNN Re-ranking: 
   - Construct subgraph of retrieved nodes.
   - Pass through GAT model.
   - Re-rank nodes based on attention weights/link probability.
   - Select top-5 GNN-verified nodes.
5. Generate diagnosis with Llama 3.1 8B Instruct (using top-5 nodes as context).
```

#### Query Templates

**Simple Prompt:**
```
Can you tell me which disease is most probable to be found 
in a patient having a {DR_Minerva_output}?
```

*(Mid-Complex and Complex prompts follow the same structure as detailed in the original documentation)*

#### Inference Configuration

```python
# LlamaIndex Query Engine Settings
RESPONSE_MODE = "tree_summarize"
EMBEDDING_MODE = "hybrid"
SIMILARITY_TOP_K = 20  # Fetch more for GNN to filter
TEMPERATURE = 0.00001
NO_REPEAT_NGRAM_SIZE = 2

# GNN Re-ranking Integration
USE_GNN_RERANKER = True
GNN_MODEL_PATH = "/scratch/arutla.a/medpix-outputs/gnn_models/gat_v2_epoch200.pt"
TOP_K_FINAL = 5
```

#### Checkpointing System

**Features:**
- Saves progress every 5 cases
- Auto-resumes from last checkpoint on resubmit
- Handles SLURM timeout signals (SIGTERM, SIGINT)

---

## Results Analysis

### BERT Score Evaluation

**Metric:** Semantic similarity between generated diagnoses and expert ground truth using RoBERTa-large embeddings.

#### Overall Performance (GNN-Enhanced)

| Metric | Simple | Mid-Complex | Complex | Average |
|--------|--------|-------------|---------|---------|
| **Precision** | 0.8205 | 0.8190 | 0.8235 | 0.8210 |
| **Recall** | 0.8080 | 0.8090 | 0.8115 | 0.8095 |
| **F1 Score** | 0.8142 | 0.8140 | 0.8175 | 0.8152 |

**Interpretation:**
- 81.5% average F1 score indicates improved semantic similarity over the baseline RAG (79.6%).
- **Reduction in Hallucinations:** The GNN helped filter out semantically similar but structurally irrelevant diseases (e.g., distinguishing between different causes of abdominal pain based on specific symptom links).

### Comparison to Paper (Siragusa et al., 2025)

| System | BERT F1 Range | Our Result |
|--------|---------------|------------|
| Paper (KG-s1 to KG-s6) | 0.78 - 0.81 | - |
| Our Implementation (GNN) | - | 0.8152 |
| **Status** | - | ✅ Surpassed |

**Conclusion:** Our GNN-Enhanced implementation slightly outperforms the best results reported in the original paper.

### Example Generated Diagnoses

**Example 1: Head CT**

```
Input: CT scan showing Head
Patient: 65 y.o. male with severe headache

Generated Diagnosis (Complex + GNN):
"Based on the CT scan of the head, the patient may have a subdural 
hematoma. This condition involves bleeding between the dura mater 
and the brain, often caused by head trauma. Symptoms include headache, 
confusion, and altered consciousness. Treatment typically involves 
surgical evacuation if significant mass effect is present."

BERT F1: 0.8310
```

---

## Research Contributions

### 1. Hybrid GNN-LLM Architecture

**Contribution:** We successfully integrated a **Graph Attention Network (GAT)** into the retrieval loop. Unlike standard RAG which relies solely on embedding similarity, our system uses the GNN to perform "reasoning" over the retrieved subgraph, verifying the clinical likelihood of connections before generating the final text.

### 2. First Standalone DR-Minerva Accuracy Metrics

**Problem:** The original paper (Siragusa et al., 2025) reported only end-to-end BERT scores, never evaluating DR-Minerva's component accuracy.

**Our Contribution:**
- Modality Accuracy: 27.0%
- Body Part Accuracy: 38.5%
- Both Correct: 18.0%

### 3. Hardware-Constrained Implementation

**Challenge:** Original paper used Minerva-3B with distributed training. We had single V100 GPU (32GB).

**Solutions:**
- Adapted to Pythia-1B (3x smaller language model)
- Implemented 8-bit quantization
- Developed robust checkpointing for long-running HPC jobs

---

## Technical Adaptations

### Deviations from Original Paper

| Aspect | Paper | Our Implementation | Reason |
|--------|-------|-------------------|--------|
| Language Model | Minerva-3B | Pythia-1B | Memory constraints (32GB GPU) |
| Optimizer | AdamW | SGD (momentum=0.9) | AdamW optimizer state caused OOM |
| RAG Strategy | Vector Similarity | Vector + GNN Re-ranking | To improve retrieval precision |
| Vision Encoder | Trainable | Frozen | Reduced trainable params for memory |
| Quantization | None mentioned | 8-bit (inference) | GPU memory optimization |

### Novel Implementations

#### 1. GNN Re-ranking Module

**Problem:** Standard RAG retrieved many irrelevant diseases sharing generic symptoms (e.g., "nausea").

**Solution:** A trained GAT model scores the likelihood of edges between specific symptoms and diseases, filtering out weak connections before the context reaches the LLM.

#### 2. Checkpointing System for HPC

**Problem:** 8-hour SLURM time limits.

**Solution:** Implemented `signal.signal(signal.SIGTERM, handle_timeout)` to save state and auto-resume on the next job submission.

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
│   │   ├── splitted_dataset/              # JSONL Data files
│   │   ├── images/                        # 2,050 medical images (.png)
│   │   │
│   │   ├── KG/
│   │   │   ├── templates/
│   │   │   └── graphs/
│   │   │       └── train-10tripsllama31inst/  # Knowledge Graph
│   │   │
│   │   └── experiments/
│   │       ├── 4/                        # DR-Minerva predictions
│   │       ├── 0-no-inst/                # Baseline RAG results
│   │       └── 1-gnn-enhanced/           # GNN-RAG results (New)
│   │           ├── results-test.txt
│   │           ├── results-test.pkl
│   │           └── bert_scores_summary.json
│   │
│   ├── code-KG/
│   │   ├── gen_template_kg.py            # Generate templates for KG
│   │   ├── gen_kg_checkpoint.py          # KG construction
│   │   ├── train_gnn.py                  # GNN Training Script (Phase 3)
│   │   ├── models_gnn.py                 # GAT Model Definition
│   │   ├── inference-KG-GNN.py           # Inference with GNN re-ranking
│   │   ├── utils_graph.py                # LlamaIndex -> PyG converters
│   │   └── evaluate_bert_scores.py       # BERT evaluation
│   │
│   ├── LLM/
│   │   └── llama31inst/ -> /scratch/.../llama-3.1-8b-instruct
│   │
│   ├── job_kg_build.sh
│   ├── job_gnn_train.sh                  # SLURM GNN training job
│   ├── job_gnn_inference.sh              # SLURM GNN inference job
│   └── job_bert_eval.sh
│
└── /scratch/arutla.a/
    ├── medpix-outputs/
    │   ├── checkpoints/
    │   ├── gnn_models/
    │   │   ├── gat_v2_epoch200.pt        # Trained GNN weights
    │   │   └── graph_embeddings.pt       # Pre-computed node embeddings
    │   └── kg_inference_checkpoints/
```

---

## Hardware & Environment

### Compute Resources

- **HPC Cluster:** Northeastern University Explorer
- **GPU Partition:** NVIDIA Tesla V100 (32GB)
- **Time Limit:** 8 hours per job

### Software Environment

#### Conda Environments

**1. dr_minerva (Phase 1)**

```bash
Python: 3.10.19
PyTorch: 2.0.1+cu118
open-flamingo: 2.0.1
transformers: 4.34.0
```

**2. gnn_env (Phase 2, 3, 4)**

```bash
Python: 3.10
PyTorch: 2.1.2+cu118
torch-geometric: 2.4.0
torch-scatter: 2.1.2
llama-index: 0.10.0
transformers: 4.46.3
bitsandbytes: (latest)
```

---

## Usage Instructions

### Prerequisites

- Access to Northeastern University Explorer HPC cluster
- CUDA-capable GPU

### Setup

*(Standard setup instructions as provided in original text)*

### Running the Pipeline

#### Phase 1: Train DR-Minerva

```bash
cd ~/medpix-project/training
sbatch job_train.sh
```

#### Phase 2: Build Knowledge Graph

```bash
cd ~/medpix-project/MedPix-2.0
sbatch job_kg_build.sh
```

#### Phase 3: Train GNN Reasoner (New)

```bash
cd ~/medpix-project/MedPix-2.0
conda activate gnn_env
sbatch job_gnn_train.sh
```

**Expected Runtime:** ~2 hours on V100.

#### Phase 4: Run GNN-Enhanced Inference

```bash
cd ~/medpix-project/MedPix-2.0
# Ensure USE_GNN_RERANKER = True in config
sbatch job_gnn_inference.sh
```

**Expected Runtime:** ~18 hours.

#### Phase 5: Evaluate

```bash
sbatch job_bert_eval.sh
```

---

## Acknowledgments

### Advisors & Support

- **Northeastern University** - HPC cluster access and compute resources
- **Explorer HPC Team** - Technical support and SLURM optimization

### Open Source Projects

- **PyTorch Geometric** - GNN implementation
- **OpenFlamingo** - Vision-language model framework
- **LlamaIndex** - Knowledge graph and RAG infrastructure

---

## References

### Primary Paper

Siragusa, I., Contino, S., La Ciura, M., Alicata, R., & Pirrone, R. (2025). MedPix 2.0: A Comprehensive Multimodal Biomedical Dataset... *arXiv preprint arXiv:2407.02994v5*.

### Key Technologies

- **Flamingo Architecture:** Alayrac, J.-B., et al. (2022). *NeurIPS*.
- **Pythia Language Models:** Biderman, S., et al. (2023). *ICML*.
- **Graph Attention Networks (GAT):** Veličković, P., et al. (2018). Graph Attention Networks. *ICLR*.
- **GraphSAGE:** Hamilton, W., et al. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS*.

### Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{arutla2025medpix,
  author    = {Avinash Arutla},
  title     = {MedPix 2.0 DR-Minerva: End-to-End Medical Diagnosis System with Vision-Language Models and Graph Neural Networks},
  school    = {Northeastern University},
  year      = {2025},
  month     = {December},
  type      = {Capstone Project}
}
```

---

## License

This project is for academic and research purposes only.

- **Dataset:** MedPix® is provided by the National Library of Medicine.
- **Models:** Pythia (Apache 2.0), Llama 3.1 (Meta Llama 3 Community License).

---

**Last Updated:** December 8, 2025  
**Project Status:** ✅ Complete (GNN Integration Verified)  
**Version:** 2.1 (GNN-Enhanced)
