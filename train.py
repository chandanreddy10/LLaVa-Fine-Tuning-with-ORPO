import torch
import lightning as L
from model.model import create_lora_model
from data.dataset import return_dataloaders
from config.model_config import config
import re
import bitsandbytes as bnb
from sentence_transformers import SentenceTransformer

val_model = SentenceTransformer("all-MiniLM-L6-v2")

class LlavaModelPLModule(L.LightningModule):
    def __init__(
        self, model, config, processor, trainDataLoader, valDataLoader, alpha=0.25
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "processor"])
        self.config = config
        self.trainDataloader = trainDataLoader
        self.valDataloader = valDataLoader
        self.processor = processor
        self.model = model
        self.alpha = alpha
        self.batch_size = config.get("batch_size")

    def compute_logps(
        self, prompt_attention_mask, chosen_inputs, chosen_attention_mask, logits
    ):
        mask = chosen_attention_mask[:, :-1]

        per_token_logps = torch.gather(
            logits[:, :-1, :].log_softmax(-1),
            dim=2,
            index=(mask * chosen_inputs[:, 1:]).unsqueeze(2),
        ).squeeze(2)
        return torch.mul(per_token_logps, mask.to(dtype=torch.bfloat16)).sum(dim=1).to(
            dtype=torch.float64
        ) / mask.sum(dim=1).to(dtype=torch.float64)

    def training_step(self, batch, batch_idx):

        (
            pos_input_ids,
            pos_attention_mask,
            pos_labels,
            neg_input_ids,
            neg_attention_mask,
            neg_labels,
            pixel_values,
            image_sizes,
        ) = batch

        # Forward passes on different GPUs
        outputs_pos = self.model(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            labels=pos_labels,
            output_hidden_states=True,
        )

        outputs_neg = self.model(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            labels=neg_labels,
            output_hidden_states=True,
        )

        # Compute loss
        pos_prob = self.compute_logps(
            pos_attention_mask, pos_input_ids, pos_attention_mask, outputs_pos.logits
        )
        neg_prob = self.compute_logps(
            neg_attention_mask, neg_input_ids, neg_attention_mask, outputs_neg.logits
        )

        log_odds = (pos_prob - neg_prob) - (
            torch.log1p(-torch.exp(pos_prob)) - torch.log1p(-torch.exp(neg_prob))
        )
        loss = torch.mean(
            outputs_pos.loss
            - self.alpha * torch.log(torch.nn.functional.sigmoid(log_odds))
        )

        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, image_sizes, desirable = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            max_new_tokens=128,
        )
        predictions = self.processor.batch_decode(
            generated_ids[:, input_ids.size(1) :], skip_special_tokens=True
        )

        prediction_embeddings = val_model.encode(predictions)
        chosen_embeddings = val_model.encode([chosen[0] for chosen in desirable])
        similarities = val_model.similarity(prediction_embeddings, chosen_embeddings)
        scores = similarities.tolist()

        for pred, chosen, similarity in zip(predictions, desirable, similarities):
            print(f"Predicted: {pred}")
            print(f"Chosen Response: {chosen[0]}")
            print(f"Similarity: {similarity}")
        
        return scores

    def configure_optimizers(self):
        return bnb.optim.Adam8bit(
            self.parameters(), lr=self.config.get("lr"), min_8bit_size=16384
        )

    def train_dataloader(self):
        return self.trainDataloader

    def val_dataloader(self):
        return self.valDataloader

if __name__ == "__main__":
    model, processor = create_lora_model()
    trainDataloader, testDataloader = return_dataloaders(processor)
    
    model_module = LlavaModelPLModule(
        config=config,
        processor=processor,
        model=model,
        trainDataLoader=trainDataloader,
        valDataLoader=testDataloader,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config["max_epochs"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        gradient_clip_val=config["gradient_clip_val"],
        precision="16-mixed",
        num_sanity_val_steps=0,
    )

    trainer.fit(model_module)
#needs to add push to hub