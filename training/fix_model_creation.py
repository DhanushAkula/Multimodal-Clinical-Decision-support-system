import sys

# Read the file
with open('train_dr_minerva.py', 'r') as f:
    content = f.read()

# Find and replace the create_model_and_transforms call
old_call = '''    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=config.VISION_ENCODER_NAME,
        clip_vision_encoder_pretrained=config.VISION_ENCODER_PRETRAINED,
        lang_encoder_path=config.LM_MODEL_NAME,
        tokenizer_path=config.LM_TOKENIZER_NAME,
        cross_attn_every_n_layers=config.CROSS_ATTN_EVERY_N_LAYERS,
    )'''

new_call = '''    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=config.VISION_ENCODER_NAME,
        clip_vision_encoder_pretrained=config.VISION_ENCODER_PRETRAINED,
        lang_encoder_path=config.LM_MODEL_NAME,
        tokenizer_path=config.LM_TOKENIZER_NAME,
        cross_attn_every_n_layers=config.CROSS_ATTN_EVERY_N_LAYERS,
        decoder_layers_attr_name="model.layers",  # For Minerva/Mistral architecture
    )'''

content = content.replace(old_call, new_call)

# Write back
with open('train_dr_minerva.py', 'w') as f:
    f.write(content)

print("✓ Fixed model creation call")
