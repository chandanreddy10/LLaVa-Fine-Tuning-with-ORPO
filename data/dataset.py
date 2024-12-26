from datasets import load_dataset
from torch.utils.data import DataLoader
from config.model_config import MAX_LENGTH, BATCH_SIZE

DATASET_ID = "openbmb/RLAIF-V-Dataset"

def train_collate_fn(examples, processor):
    images, chosen_texts, rejected_texts = [], [], []

    for image, question, chosen, reject in examples:
        images.append(image)

        for text, texts_list in [(chosen, chosen_texts), (reject, rejected_texts)]:
            conversation = [
                {"role": "user", "content": [{"type": "text", "text": question}]},
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
            ]
            texts_list.append(processor.apply_chat_template(conversation))

    chosen_batch = processor(
        text=chosen_texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    reject_batch = processor(
        text=rejected_texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    for batch in [chosen_batch, reject_batch]:
        batch["labels"] = batch["input_ids"].clone()

    return (
        chosen_batch["input_ids"],
        chosen_batch["attention_mask"],
        chosen_batch["labels"],
        reject_batch["input_ids"],
        reject_batch["attention_mask"],
        reject_batch["labels"],
        chosen_batch["pixel_values"],
        chosen_batch["image_sizes"],
    )


def val_collate_fn(examples, processor):
    images, chosen_texts, questions = [], [], []

    for image, question, chosen in examples:
        images.append(image)
        chosen_texts.append(chosen)
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
        batch = processor.apply_chat_template(conversation, add_generation_prompt=True)
        questions.append(batch)

    question_batch = processor(
        text=questions,
        images=images,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    question_ids = question_batch["input_ids"]
    question_attention_mask = question_batch["attention_mask"]
    pixel_values = question_batch["pixel_values"]
    image_sizes = question_batch["image_sizes"]

    return (
        question_ids,
        question_attention_mask,
        pixel_values,
        image_sizes,
        chosen_texts,
        questions
    )

def return_dataloaders(processor, dataset_id=DATASET_ID, dataset_chosen="OK-VQA"):

    print("Loading Dataset.")
    dataset = load_dataset(dataset_id, split="train")
    if dataset_chosen:
        dataset = dataset.filter(
            lambda row: row["origin_dataset"] == dataset_chosen
        ).select(range(6750))

    train_dataset, val_dataset = dataset.train_test_split(test_size=0.01).values()
    train_dataset = [
        (item["image"], item["question"], item["chosen"], item["rejected"])
        for item in train_dataset
    ]
    val_dataset = [
        (item["image"], item["question"], item["chosen"]) for item in val_dataset
    ]
    trainDataLoader = DataLoader(
        train_dataset,
        collate_fn=lambda batch: train_collate_fn(batch, processor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    valDataLoader = DataLoader(
        val_dataset,
        collate_fn=lambda batch: val_collate_fn(batch, processor),
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    return trainDataLoader, valDataLoader
