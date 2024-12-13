import os
import random
import numpy as np
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

#train_file = "input/medium_dataset/train/buggy_pre_train.txt"
#train_file = "input/medium_dataset/train/fixed_pre_train.txt"
train_file = "input/collected_dataset/fixed_code/pretrain_dataset.txt"

def load_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [{"tokens": line.strip()} for line in lines]

train_data = load_data(train_file)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_list(train_data)

# Debugging: smaller dataset
# train_dataset = train_dataset.select(range(1000))

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess_function(examples):
    def mask_tokens(tokens):
        tokens = tokens.split()  
        if len(tokens) == 0:
            return "<pad>"  
        num_to_mask = max(1, int(0.15 * len(tokens)))
        mask_indices = np.random.choice(len(tokens), num_to_mask, replace=False) 
        for idx in mask_indices:
            tokens[idx] = "<mask>"
        return " ".join(tokens)

    masked_texts = [mask_tokens(tokens) for tokens in examples["tokens"]]
    model_inputs = tokenizer(
        masked_texts,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["tokens"]
)

training_args = TrainingArguments(
    output_dir="./pretrained_model",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=1000
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./pretrained_model")
