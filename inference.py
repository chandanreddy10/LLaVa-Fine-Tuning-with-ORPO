from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
)
import torch
from config.model_config import MODEL_ID
import os
from PIL import Image
import random
import re

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
    image_token_index=1,
    quantization_config=quantization_config,
)
# prepare image and prompt for the model
general_prompts = [
    "Describe the mood of this scene.",
    "What objects are in the image?",
    "What is happening in this picture?",
    "Write a short description of this picture."
]
images = "images"

for image in os.listdir("images"):
    image_path = os.path.join(images, image)
    img = Image.open(image_path)

    random_question = random.choice(general_prompts)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": random_question},
            ],
        },
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=img, return_tensors="pt").to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    response = re.sub(r'\[INST\]|\[/INST\]|<\\s>','',generated_texts[0])
    print(image_path)
    print(response)
