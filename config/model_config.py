import torch
from transformers import BitsAndBytesConfig

# Hyperparameters and model configuration
MAX_LENGTH = 256
MAX_TOKEN_GENERATION = 128
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
GRADIENT_CLIP_VAL = 1.0
ACCUMULATE_GRAD_BATCHES = 16

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

config = {
    "max_epochs": 1,
    "lr": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "gradient_clip_val": GRADIENT_CLIP_VAL,
    "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
    "check_val_every_n_epoch": 16,
}
