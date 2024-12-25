import torch
import torch.nn as nn
from transformers import LlavaNextForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from config import bnb_config

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    for exclude in ['lm_head', 'up_proj', 'down_proj', 'gate_proj']:
        if exclude in lora_module_names:
            lora_module_names.remove(exclude)

    return list(lora_module_names)

def create_lora_model(model_id="llava-hf/llava-v1.6-mistral-7b-hf", bnb_config=bnb_config):
    # Load base model
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        image_token_index=1,
        torch_dtype=torch.float16,
        device_map=None
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    return model
