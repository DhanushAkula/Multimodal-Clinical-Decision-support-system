"""
DR-Minerva Evaluation Script - Pythia-1B Model
Minimal version for quick evaluation
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

from open_flamingo import create_model_and_transforms

sys.path.insert(0, "/home/arutla.a/medpix-project/training")
import config

# ============================================================================
# DATASET
# ============================================================================

class MedPixEvalDataset(Dataset):
    def __init__(self, cases_jsonl, descriptions_jsonl, images_dir, image_processor, tokenizer):
        self.images_dir = Path(images_dir)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        
        # Load cases
        print(f"Loading {cases_jsonl}...")
        self.cases = []
        with open(cases_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    self.cases.append(json.loads(line))
        
        # Load descriptions
        print(f"Loading {descriptions_jsonl}...")
        self.descriptions = {}
        with open(descriptions_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    desc = json.loads(line)
                    self.descriptions[(desc['U_id'], desc['image'])] = desc
        
        # Create samples
        self.samples = []
        for case in self.cases:
            for img_name in case.get('TAC', []) + case.get('MRI', []):
                key = (case['U_id'], img_name)
                if key in self.descriptions:
                    self.samples.append({
                        'case': case,
                        'desc': self.descriptions[key],
                        'img_name': img_name
                    })
        
        print(f"✓ {len(self.samples)} test samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        desc = sample['desc']
        case = sample['case']
        
        # Load image
        img = None
        for ext in ['.png', '.jpg', '.jpeg']:
            path = self.images_dir / f"{sample['img_name']}{ext}"
            if path.exists():
                try:
                    img = Image.open(path).convert('RGB')
                    break
                except:
                    pass
        
        if img is None:
            img = Image.new('RGB', (224, 224))
        
        vision_x = self.image_processor(img).unsqueeze(0)
        
        # Get ground truth
        gt_modality = desc.get('Type', 'Unknown')
        gt_body_part = desc.get('Location Category', 'Unknown')
        
        # Create prompt
        age = desc.get('Description', {}).get('Age', 'N/A')
        sex = desc.get('Description', {}).get('Sex', 'N/A')
        history = case.get('Case', {}).get('History', 'No history')
        
        prompt = f"<image>Patient: {age} year old {sex}. History: {history} What scan type and body part?"
        
        # Tokenize
        tokens = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        return {
            'vision_x': vision_x,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'gt_modality': gt_modality,
            'gt_body_part': gt_body_part,
        }

# ============================================================================
# PARSING
# ============================================================================

def parse_output(text):
    """Extract modality and body part from generated text"""
    text_upper = text.upper()
    
    # Parse modality
    if 'MRI' in text_upper or 'MAGNETIC' in text_upper:
        modality = 'MRI'
    elif 'CT' in text_upper or 'COMPUTED' in text_upper:
        modality = 'CT'
    else:
        modality = 'Unknown'
    
    # Parse body part
    body_parts = {
        'Head': ['HEAD', 'BRAIN', 'SKULL', 'NECK'],
        'Thorax': ['THORAX', 'CHEST', 'LUNG', 'HEART'],
        'Abdomen': ['ABDOMEN', 'LIVER', 'KIDNEY', 'SPLEEN'],
        'RUS': ['PELVIS', 'BLADDER', 'REPRODUCTIVE', 'URINARY'],
        'SaM': ['SPINE', 'MUSCLE', 'BONE', 'EXTREMIT']
    }
    
    scores = defaultdict(int)
    for part, keywords in body_parts.items():
        for kw in keywords:
            if kw in text_upper:
                scores[part] += 1
    
    body_part = max(scores, key=scores.get) if scores else 'Unknown'
    
    return modality, body_part

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model(checkpoint_path):
    """Load Pythia-1B model from checkpoint"""
    print(f"\nLoading model from {checkpoint_path}...")
    
    # Create model with Pythia-1B
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="EleutherAI/pythia-1b",
        tokenizer_path="EleutherAI/pythia-1b",
        cross_attn_every_n_layers=1,
        decoder_layers_attr_name="gpt_neox.layers",
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model.eval()
    
    print(f"✓ Model loaded (epoch {checkpoint['epoch']})")
    
    return model, image_processor, tokenizer

# ============================================================================
# EVALUATE
# ============================================================================

def evaluate(model, dataloader, tokenizer):
    """Run evaluation"""
    print("\nEvaluating...")
    
    all_preds_mod = []
    all_gt_mod = []
    all_preds_part = []
    all_gt_part = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            vision_x = batch['vision_x'].cuda()
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            
            # CRITICAL: Add unsqueeze(2) for Pythia model
            vision_x = vision_x.unsqueeze(2)
            
            # Generate
            outputs = model.generate(
                vision_x=vision_x,
                lang_x=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                num_beams=1,
                do_sample=False,
            )
            
            # Decode and parse
            for i in range(len(batch['gt_modality'])):
                text = tokenizer.decode(outputs[i], skip_special_tokens=True)
                pred_mod, pred_part = parse_output(text)
                
                all_preds_mod.append(pred_mod)
                all_gt_mod.append(batch['gt_modality'][i])
                all_preds_part.append(pred_part)
                all_gt_part.append(batch['gt_body_part'][i])
    
    return all_preds_mod, all_gt_mod, all_preds_part, all_gt_part

# ============================================================================
# MAIN
# ============================================================================

def main():
    checkpoint = "/scratch/arutla.a/medpix-outputs/checkpoints/best_model.pt"
    
    # Load model
    model, image_processor, tokenizer = load_model(checkpoint)
    
    # Create dataset
    dataset = MedPixEvalDataset(
        config.TEST_JSONL,
        config.TEST_DESC_JSONL,
        config.IMAGE_DIR,
        image_processor,
        tokenizer
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Evaluate
    pred_mod, gt_mod, pred_part, gt_part = evaluate(model, dataloader, tokenizer)
    
    # Calculate metrics
    mod_correct = sum(p == g for p, g in zip(pred_mod, gt_mod))
    part_correct = sum(p == g for p, g in zip(pred_part, gt_part))
    both_correct = sum((pm == gm and pp == gp) for pm, gm, pp, gp in zip(pred_mod, gt_mod, pred_part, gt_part))
    
    total = len(pred_mod)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Modality Accuracy:  {mod_correct/total:.1%} ({mod_correct}/{total})")
    print(f"Body Part Accuracy: {part_correct/total:.1%} ({part_correct}/{total})")
    print(f"Both Correct:       {both_correct/total:.1%} ({both_correct}/{total})")
    print(f"{'='*60}\n")
    
    # Save results
    results = {
        'modality_accuracy': mod_correct / total,
        'body_part_accuracy': part_correct / total,
        'both_correct': both_correct / total,
        'total_samples': total
    }
    
    output_path = "/scratch/arutla.a/medpix-outputs/results/eval_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
