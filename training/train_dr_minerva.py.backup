"""
DR-Minerva Training Script - DISTRIBUTED VERSION
- Full DistributedDataParallel (DDP) support for multi-GPU training
- Loads MedPix 2.0 two-file structure (cases + descriptions)
- Trains Flamingo model to predict modality + body part
- Saves checkpoints every epoch (rank 0 only)
"""

import os
import sys
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image

from open_flamingo import create_model_and_transforms
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, "/home/arutla.a/medpix-project/training")
import config

# ============================================================================
# DISTRIBUTED SETUP
# ============================================================================

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU fallback
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        print(f"Initialized distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    return rank, world_size, local_rank

def cleanup_distributed(world_size):
    """Cleanup distributed training"""
    if world_size > 1:
        torch.distributed.destroy_process_group()

# ============================================================================
# SETUP
# ============================================================================

def setup_logging(log_dir: str, rank: int = 0) -> logging.Logger:
    """Setup logging (only on rank 0)"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}_rank{rank}.log")
    
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ] if rank == 0 else [logging.FileHandler(log_file)]
    )
    return logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================================
# DATASET - HANDLES TWO-FILE STRUCTURE
# ============================================================================

class MedPixFlamingoDataset(Dataset):
    """
    Loads MedPix 2.0 two-file structure:
    1. Cases: U_id, TAC (CT list), MRI (MRI list), Case dict
    2. Descriptions: Type, U_id, image, Description dict, Location Category
    
    Creates one sample per image (535 cases → ~1653 samples)
    """
    
    def __init__(
        self,
        cases_jsonl: str,
        descriptions_jsonl: str,
        images_dir: str,
        image_processor,
        tokenizer,
        max_length: int = 512,
        split: str = 'train',
        rank: int = 0
    ):
        self.images_dir = Path(images_dir)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Load cases (only log on rank 0)
        if rank == 0:
            print(f"Loading cases from {cases_jsonl}...")
        self.cases = []
        self.cases_by_uid = {}
        with open(cases_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    self.cases.append(case)
                    self.cases_by_uid[case['U_id']] = case
        if rank == 0:
            print(f"✓ Loaded {len(self.cases)} cases")
        
        # Load descriptions - index by (U_id, image_name)
        if rank == 0:
            print(f"Loading descriptions from {descriptions_jsonl}...")
        self.descriptions = {}
        with open(descriptions_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    desc = json.loads(line)
                    key = (desc['U_id'], desc['image'])
                    self.descriptions[key] = desc
        if rank == 0:
            print(f"✓ Loaded {len(self.descriptions)} descriptions")
        
        # Flatten: one sample per image
        self.samples = []
        for case in self.cases:
            u_id = case['U_id']
            
            # Process CT images
            for img_name in case.get('TAC', []):
                key = (u_id, img_name)
                if key in self.descriptions:
                    self.samples.append({
                        'case': case,
                        'description': self.descriptions[key],
                        'image_name': img_name,
                        'u_id': u_id
                    })
            
            # Process MRI images
            for img_name in case.get('MRI', []):
                key = (u_id, img_name)
                if key in self.descriptions:
                    self.samples.append({
                        'case': case,
                        'description': self.descriptions[key],
                        'image_name': img_name,
                        'u_id': u_id
                    })
        
        if rank == 0:
            print(f"✓ Created {len(self.samples)} samples ({split})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get one training sample"""
        sample = self.samples[idx]
        case = sample['case']
        desc = sample['description']
        img_name = sample['image_name']
        
        # Load image (try multiple extensions)
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            potential_path = self.images_dir / f"{img_name}{ext}"
            if potential_path.exists():
                img_path = potential_path
                break
        
        if img_path and img_path.exists():
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                img = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE))
        else:
            img = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE))
        
        vision_x = self.image_processor(img).unsqueeze(0)  # [1, C, H, W]
        
        # Extract fields
        desc_dict = desc.get('Description', {})
        age = desc_dict.get('Age', 'N/A')
        sex = desc_dict.get('Sex', 'N/A')
        
        case_dict = case.get('Case', {})
        history = case_dict.get('History', 'No history available.')
        
        modality = desc.get('Type', 'CT')  # "CT" or "MRI"
        body_part = desc.get('Location Category', 'Unknown')
        
        # Create prompt and response
        prompt = config.PROMPT_TEMPLATE.format(
            age=age,
            sex=sex,
            clinical_history=history
        )
        
        response = config.RESPONSE_TEMPLATE.format(
            modality=modality,
            body_part=body_part
        )
        
        # Tokenize
        full_text = f"{prompt}<answer>{response}"
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create labels (mask prompt, predict response only)
        labels = input_ids.clone()
        prompt_encoding = self.tokenizer(
            f"{prompt}<answer>",
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        labels[:prompt_length] = -100  # Ignore in loss
        
        return {
            'vision_x': vision_x,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'modality': modality,
            'body_part': body_part,
            'u_id': sample['u_id'],
            'image_name': img_name
        }

# ============================================================================
# MODEL
# ============================================================================

def create_flamingo_model(local_rank=0):
    """Create Flamingo model with Minerva-3B"""
    logger = logging.getLogger(__name__)
    
    if local_rank == 0:
        logger.info("Creating Flamingo model...")
    
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=config.VISION_ENCODER_NAME,
        clip_vision_encoder_pretrained=config.VISION_ENCODER_PRETRAINED,
        lang_encoder_path=config.LM_MODEL_NAME,
        tokenizer_path=config.LM_TOKENIZER_NAME,
        cross_attn_every_n_layers=config.CROSS_ATTN_EVERY_N_LAYERS,
    )
    
    if config.USE_GRADIENT_CHECKPOINTING:
        if local_rank == 0:
            logger.info("Enabling gradient checkpointing...")
        model.lang_encoder.gradient_checkpointing_enable()
    
    # Move to correct GPU
    device = torch.device(f'cuda:{local_rank}')
    model = model.to(device)
    
    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")
    
    return model, image_processor, tokenizer

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, logger, writer=None, rank=0, world_size=1):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    # Only show progress bar on rank 0
    if rank == 0:
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        progress_bar = train_loader
    
    for step, batch in enumerate(progress_bar):
        vision_x = batch['vision_x'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            vision_x=vision_x,
            lang_x=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        
        if rank == 0:
            if hasattr(progress_bar, 'set_postfix'):
                progress_bar.set_postfix({'loss': f'{loss.item() * config.GRADIENT_ACCUMULATION_STEPS:.4f}'})
        
        if writer and rank == 0 and step % config.LOG_EVERY_N_STEPS == 0:
            global_step = epoch * len(train_loader) + step
            writer.add_scalar('train/loss', loss.item() * config.GRADIENT_ACCUMULATION_STEPS, global_step)
    
    avg_loss = total_loss / len(train_loader)
    if rank == 0:
        logger.info(f"Epoch {epoch} - Avg train loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, val_loader, device, epoch, logger, writer=None, rank=0):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        iterator = tqdm(val_loader, desc=f"Validation {epoch}") if rank == 0 else val_loader
        for batch in iterator:
            vision_x = batch['vision_x'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                vision_x=vision_x,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(val_loader)
    if rank == 0:
        logger.info(f"Epoch {epoch} - Val loss: {avg_loss:.4f}")
        if writer:
            writer.add_scalar('val/loss', avg_loss, epoch)
    
    return avg_loss

def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir, is_best=False, rank=0):
    """Save checkpoint (only on rank 0)"""
    if rank != 0:
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Extract state dict from DDP wrapper if needed
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")
    
    # Keep only last N checkpoints
    if config.SAVE_TOTAL_LIMIT:
        checkpoints = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        if len(checkpoints) > config.SAVE_TOTAL_LIMIT:
            for old_ckpt in checkpoints[:-config.SAVE_TOTAL_LIMIT]:
                os.remove(os.path.join(checkpoint_dir, old_ckpt))

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training loop with distributed support"""
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    # Only setup logging and directories on rank 0
    if rank == 0:
        set_seed(config.SEED)
        for dir_path in config.DIRS_TO_CREATE:
            os.makedirs(dir_path, exist_ok=True)
        logger = setup_logging(config.LOG_DIR, rank)
        logger.info("Starting DR-Minerva training...")
        config.print_config()
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)  # Reduce logging on other ranks
    
    # Set seed with rank offset for data augmentation diversity
    set_seed(config.SEED + rank)
    
    device = torch.device(f'cuda:{local_rank}')
    if rank == 0:
        logger.info(f"World size: {world_size}")
        logger.info(f"Rank: {rank}/{world_size}")
        logger.info(f"Local rank: {local_rank}")
        logger.info(f"Device: {device}")
    
    # Create model
    model, image_processor, tokenizer = create_flamingo_model(local_rank)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True
        )
        if rank == 0:
            logger.info(f"Model wrapped with DDP")
    
    # Create datasets (only log on rank 0)
    if rank == 0:
        logger.info("Loading datasets...")
    
    train_dataset = MedPixFlamingoDataset(
        cases_jsonl=config.TRAIN_JSONL,
        descriptions_jsonl=config.TRAIN_DESC_JSONL,
        images_dir=config.IMAGE_DIR,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=config.MAX_TEXT_LENGTH,
        split='train',
        rank=rank
    )
    
    val_dataset = MedPixFlamingoDataset(
        cases_jsonl=config.DEV_JSONL,
        descriptions_jsonl=config.DEV_DESC_JSONL,
        images_dir=config.IMAGE_DIR,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=config.MAX_TEXT_LENGTH,
        split='dev',
        rank=rank
    )
    
    if rank == 0:
        logger.info(f"Train: {len(train_dataset)} samples")
        logger.info(f"Val: {len(val_dataset)} samples")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.SEED
    ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True  # For DDP stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Setup scheduler
    num_training_steps = len(train_loader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=num_training_steps
    )
    
    # Setup tensorboard (only on rank 0)
    writer = None
    if config.TENSORBOARD_ENABLED and rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, 'tensorboard'))
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        if rank == 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch}/{config.NUM_EPOCHS}")
            logger.info(f"{'='*80}")
        
        # Set epoch for distributed sampler (important for shuffling)
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, logger, writer, rank, world_size
        )
        
        # Validate
        val_loss = validate(
            model, val_loader, device, epoch, logger, writer, rank
        )
        
        # Save checkpoint (only rank 0)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            if rank == 0:
                logger.info(f"New best val loss: {best_val_loss:.4f}")
        
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                config.CHECKPOINT_DIR, is_best, rank
            )
        
        # Synchronize all processes
        if world_size > 1:
            torch.distributed.barrier()
    
    if rank == 0:
        logger.info(f"\nTraining complete!")
        logger.info(f"Best val loss: {best_val_loss:.4f}")
    
    if writer:
        writer.close()
    
    # Cleanup
    cleanup_distributed(world_size)

if __name__ == "__main__":
    main()
