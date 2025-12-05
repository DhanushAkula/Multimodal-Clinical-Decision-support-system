"""
Configuration for DR-Minerva training
- All paths (data, checkpoints, logs)
- Model settings (Flamingo + Minerva-3B)
- Training hyperparameters
"""

import os

# ============================================================================
# PATHS
# ============================================================================

HOME_DIR = "/home/arutla.a"
PROJECT_DIR = f"{HOME_DIR}/medpix-project"
SCRATCH_DIR = "/scratch/arutla.a/medpix-outputs"

# Dataset paths - TWO FILES PER SPLIT
DATASET_ROOT = f"{PROJECT_DIR}/MedPix-2.0/MedPix-2-0"
SPLIT_DIR = f"{DATASET_ROOT}/splitted_dataset"

# Case files (U_id, TAC, MRI, Case dict)
TRAIN_JSONL = f"{SPLIT_DIR}/data_train.jsonl"
DEV_JSONL = f"{SPLIT_DIR}/data_dev.jsonl"
TEST_JSONL = f"{SPLIT_DIR}/data_test.jsonl"

# Description files (Type, U_id, image, Description dict, Location Category)
TRAIN_DESC_JSONL = f"{SPLIT_DIR}/descriptions_train.jsonl"
DEV_DESC_JSONL = f"{SPLIT_DIR}/descriptions_dev.jsonl"
TEST_DESC_JSONL = f"{SPLIT_DIR}/descriptions_test.jsonl"

# Images
IMAGE_DIR = f"{DATASET_ROOT}/images"

# Outputs
CHECKPOINT_DIR = f"{SCRATCH_DIR}/checkpoints"
LOG_DIR = f"{SCRATCH_DIR}/logs"
RESULTS_DIR = f"{SCRATCH_DIR}/results"

DIRS_TO_CREATE = [SCRATCH_DIR, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]

# ============================================================================
# MODEL
# ============================================================================

LM_MODEL_NAME = "EleutherAI/pythia-1b"  # Base for Minerva-3B
LM_TOKENIZER_NAME = "EleutherAI/pythia-1b"
VISION_ENCODER_NAME = "ViT-L-14"
VISION_ENCODER_PRETRAINED = "openai"
CROSS_ATTN_EVERY_N_LAYERS = 1

# ============================================================================
# TRAINING
# ============================================================================

BATCH_SIZE = 4  # Per GPU
GRADIENT_ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
MAX_GRAD_NORM = 1.0
LR_SCHEDULER_TYPE = "cosine"

# Memory optimization
USE_GRADIENT_CHECKPOINTING = False
USE_DEEPSPEED = True
USE_AMP = True

# Checkpointing
SAVE_EVERY_N_EPOCHS = 1
SAVE_TOTAL_LIMIT = 5

# ============================================================================
# DATA
# ============================================================================

IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 256
NUM_SHOTS = 4  # Few-shot examples for RAG
NUM_WORKERS = 4
PIN_MEMORY = True
EVAL_BATCH_SIZE = 8

# ============================================================================
# PROMPTS
# ============================================================================

PROMPT_TEMPLATE = (
    "<image>Patient Information: {age} year old {sex}. "
    "Clinical History: {clinical_history} "
    "What type of scan is this and which body part is shown?"
)

RESPONSE_TEMPLATE = "This is a {modality} scan showing the {body_part}."

# ============================================================================
# MISC
# ============================================================================

SEED = 42
LOG_EVERY_N_STEPS = 10
EVAL_EVERY_N_STEPS = 500
TENSORBOARD_ENABLED = True
DEBUG_MODE = False

# ============================================================================
# HELPERS
# ============================================================================

def get_model_config():
    return {
        "lm_model_name": LM_MODEL_NAME,
        "lm_tokenizer_name": LM_TOKENIZER_NAME,
        "vision_encoder_name": VISION_ENCODER_NAME,
        "vision_encoder_pretrained": VISION_ENCODER_PRETRAINED,
        "cross_attn_every_n_layers": CROSS_ATTN_EVERY_N_LAYERS,
        "use_gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
    }

def get_training_config():
    return {
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
        "max_grad_norm": MAX_GRAD_NORM,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "seed": SEED,
    }

def get_data_config():
    return {
        "train_jsonl": TRAIN_JSONL,
        "train_desc_jsonl": TRAIN_DESC_JSONL,
        "dev_jsonl": DEV_JSONL,
        "dev_desc_jsonl": DEV_DESC_JSONL,
        "test_jsonl": TEST_JSONL,
        "test_desc_jsonl": TEST_DESC_JSONL,
        "image_dir": IMAGE_DIR,
        "image_size": IMAGE_SIZE,
        "max_text_length": MAX_TEXT_LENGTH,
        "num_shots": NUM_SHOTS,
    }

def print_config():
    print("=" * 80)
    print("DR-MINERVA TRAINING CONFIG")
    print("=" * 80)
    print(f"Train Cases: {TRAIN_JSONL}")
    print(f"Train Descriptions: {TRAIN_DESC_JSONL}")
    print(f"Images: {IMAGE_DIR}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"\nModel: {LM_MODEL_NAME}")
    print(f"Vision: {VISION_ENCODER_NAME}")
    print(f"Batch Size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {EFFECTIVE_BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("=" * 80)
