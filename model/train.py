import lightning as L
from model.model import create_lora_model
from data.dataset import get_train_dataloader, get_val_dataloader
from config.model_config import config

class LlavaModelPLModule(L.LightningModule):
    # Define training and validation steps here, as well as optimizer setup
    pass

def train():
    model = create_lora_model()
    
    model_module = LlavaModelPLModule(config, processor, model)
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
