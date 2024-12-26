from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
import torch
from config.model_config import MODEL_ID
from data.dataset import DATASET_ID
from datasets import load_dataset

REPO_ID = "chandanreddy/LLaVA-ORPO"
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Define quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
# Load the base model with adapters on top
model = LlavaNextForConditionalGeneration.from_pretrained(
    REPO_ID,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
)
dataset = load_dataset(DATASET_ID, split="train")
dataset = dataset.filter(
        lambda row: row["origin_dataset"] == "OK-VQA"
    ).select(range(6900))

image = dataset['image']
# prepare image and prompt for the model
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text":dataset['question']},
        ],
    },
]
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
