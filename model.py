import torch
import lightning as L
from datasets import load_dataset
from transformers import LlavaNextProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch.utils.data import DataLoader
import os
import bitsandbytes as bnb
import re 
from sentence_transformers import SentenceTransformer


val_model = SentenceTransformer("all-MiniLM-L6-v2")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_LENGTH = 256
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

# Initialize processor
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"
processor.patch_size = 16

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load base model
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    image_token_index=1,
    torch_dtype=torch.float16,
    device_map=None )

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

    for exclude in ['lm_head', 'up_proj','down_proj','gate_proj']:
        if exclude in lora_module_names:
            lora_module_names.remove(exclude)

    return list(lora_module_names)

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

def train_collate_fn(examples):
    images, chosen_texts, rejected_texts = [], [], []
    
    for image, question, chosen, reject in examples:
        images.append(image)
        
        for text, texts_list in [(chosen, chosen_texts), (reject, rejected_texts)]:
            conversation = [
                {"role": "user", "content": 
                [{"type": "text", "text": question}]},
                {"role": "assistant", "content": [{"type": "text", "text": text}]}
            ]
            texts_list.append(processor.apply_chat_template(conversation))

    chosen_batch = processor(text=chosen_texts, images=images, padding=True, truncation=True, 
                           max_length=MAX_LENGTH, return_tensors="pt")
    reject_batch = processor(text=rejected_texts, images=images, padding=True, truncation=True, 
                           max_length=MAX_LENGTH, return_tensors="pt")

    for batch in [chosen_batch, reject_batch]:
        batch["labels"] = batch["input_ids"].clone()

    return (chosen_batch["input_ids"], chosen_batch["attention_mask"], chosen_batch["labels"],
            reject_batch["input_ids"], reject_batch["attention_mask"], reject_batch["labels"],
            chosen_batch["pixel_values"], chosen_batch["image_sizes"])
def val_collate_fn(examples):
    images, chosen_texts,questions = [], [], []
    
    for image, question, chosen in examples:
        images.append(image)
        chosen_texts.append(chosen)
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": question}]},
            
        ]
        batch = processor.apply_chat_template(conversation,add_generation_prompt=True)
        questions.append(batch)

    question_batch = processor(text=questions, images=images, padding=True, truncation=True, 
                           max_length=MAX_LENGTH, return_tensors="pt")
    question_ids = question_batch['input_ids']
    question_attention_mask = question_batch['attention_mask']
    pixel_values = question_batch['pixel_values']
    image_sizes = question_batch["image_sizes"]

    return question_ids, question_attention_mask, pixel_values, image_sizes, chosen_texts

class LlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model, alpha=0.25):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'processor'])
        self.config = config
        self.processor = processor
        self.model = model
        self.alpha = alpha
        self.batch_size = config.get("batch_size")

    def compute_logps(self, prompt_attention_mask, chosen_inputs, chosen_attention_mask, logits):
        mask = chosen_attention_mask[:, :-1] 
    
        per_token_logps = torch.gather(logits[:, :-1, :].log_softmax(-1), dim=2, 
                                     index=(mask * chosen_inputs[:, 1:]).unsqueeze(2)).squeeze(2)
        return torch.mul(per_token_logps, mask.to(dtype=torch.bfloat16)).sum(dim=1).to(dtype=torch.float64) / mask.sum(dim=1).to(dtype=torch.float64)

    def training_step(self, batch, batch_idx):
        
        pos_input_ids, pos_attention_mask, pos_labels, neg_input_ids, neg_attention_mask, neg_labels, pixel_values, image_sizes = batch
        
        # Forward passes on different GPUs
        outputs_pos = self.model(input_ids=pos_input_ids, attention_mask=pos_attention_mask,
                               pixel_values=pixel_values, image_sizes=image_sizes, labels=pos_labels,output_hidden_states=True)
                      
        outputs_neg = self.model(input_ids=neg_input_ids, attention_mask=neg_attention_mask,
                               pixel_values=pixel_values, image_sizes=image_sizes, labels=neg_labels,output_hidden_states=True)

        # Compute loss
        pos_prob = self.compute_logps(pos_attention_mask, pos_input_ids, pos_attention_mask, outputs_pos.logits)
        neg_prob = self.compute_logps(neg_attention_mask, neg_input_ids, neg_attention_mask, outputs_neg.logits)
        
        log_odds = (pos_prob - neg_prob) - (torch.log1p(-torch.exp(pos_prob)) - torch.log1p(-torch.exp(neg_prob)))
        loss = torch.mean(outputs_pos.loss - self.alpha * torch.log(torch.nn.functional.sigmoid(log_odds)))
        
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, image_sizes, desirable = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, image_sizes=image_sizes, max_new_tokens=128)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, desirable):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            predicted = val_model.encode(pred)
            preferred = val_model.encode(desirable[0])

            similarity = val_model.similarity(predicted, preferred)
            print(f"Predicted-{pred}")
            print(f"Chosen Response-{desirable[0]}")
            print(similarity)
        #     scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

        #     if self.config.get("verbose", False) and len(scores) == 1:
        #         print(f"Prediction: {pred}")
        #         print(f"    Answer: {answer}")
        #         print(f" Normed ED: {scores[0]}")

        # self.log("val_edit_distance", np.mean(scores))

        return scores
    def configure_optimizers(self):
        return bnb.optim.Adam8bit(self.parameters(), lr=self.config.get("lr"),min_8bit_size=16384)

    def train_dataloader(self):
        dataset = load_dataset("openbmb/RLAIF-V-Dataset", split='train')
        data = [(item['image'], item['question'], item['chosen'], item['rejected']) 
                for item in dataset.select(range(150))]
        return DataLoader(data, collate_fn=train_collate_fn, batch_size=self.config["batch_size"],
                         shuffle=True, num_workers=4)
    def val_dataloader(self):
        dataset = load_dataset("openbmb/RLAIF-V-Dataset", split='train')
        data = [(item['image'], item['question'], item['chosen']) 
                for item in dataset.select(range(150,160))]
        return DataLoader(data, collate_fn=val_collate_fn, batch_size=self.config["batch_size"],
        num_workers=4)
    

config = {
    "max_epochs": 1,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 16,
    "lr": 1e-4,
    "batch_size": 1,
    "check_val_every_n_epoch": 16,
}

trainer = L.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=config["max_epochs"],
    accumulate_grad_batches=config["accumulate_grad_batches"],
    gradient_clip_val=config["gradient_clip_val"],
    precision="16-mixed",
    num_sanity_val_steps=0,
)

model_module = LlavaModelPLModule(config, processor, model)
trainer.fit(model_module)