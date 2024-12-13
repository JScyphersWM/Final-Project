import os
import numpy as np
from datasets import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
)
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import pandas as pd
import time
import torch

torch.cuda.empty_cache()

# Set device to CPU
device = torch.device("cpu")
torch.set_default_device(device)

def load_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [{"tokens": line.strip()} for line in lines]

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

fine_tune_file = "input/medium_dataset/train/buggy_fine_tune.txt"
#fine_tune_file = "input/medium_dataset/train/fixed_fine_tune.txt"
#fine_tune_file = "input/collected_dataset/train/buggy_fine_tune.txt"
#fine_tune_file = "input/collected_dataset/train/fixed_fine_tune.txt"

testing_file ="input/medium_dataset/test/buggy.txt"
#testing_file ="input/medium_dataset/test/fixed.txt"
#testing_file ="input/collected_dataset/test/buggy.txt"
#testing_file ="input/collected_dataset/test/fixed.txt"

evaluation_file = "input/medium_dataset/eval/buggy.txt"
#evaluation_file = "input/medium_dataset/eval/fixed.txt"
#evaluation_file = "input/collected_dataset/eval/buggy.txt"
#evaluation_file = "input/collected_dataset/eval/fixed.txt"

pretrained_model_path = "models/buggy_prov_pretrained_model"
#pretrained_model_path = "models/fixde_prov_pretrained_model"

tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path)



fine_tune_data = load_data(fine_tune_file)
fine_tune_dataset = Dataset.from_list(fine_tune_data).map(
    preprocess_function, batched=True, remove_columns=["tokens"]
)

testing_data = load_data(testing_file)
testing_dataset = Dataset.from_list(testing_data).map(
    preprocess_function, batched=True, remove_columns=["tokens"]
)

evaluation_data = load_data(evaluation_file)
evaluation_dataset = Dataset.from_list(evaluation_data).map(
    preprocess_function, batched=True, remove_columns=["tokens"]
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)
model = T5ForConditionalGeneration.from_pretrained(pretrained_model_path).to(device)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=fine_tune_dataset,
    eval_dataset=fine_tune_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")

def evaluate_model(dataset, model, tokenizer):
    results = []
    start_time = time.time()

    for example in dataset:
        input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        reference_text = tokenizer.decode(example["labels"], skip_special_tokens=True)

        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").input_ids.to(device)

        model.to(device)

        outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        confidence_scores = model(inputs, labels=inputs).logits.softmax(dim=-1).max().item()
        bleu_score = sentence_bleu([reference_text.split()], predicted_text.split())
        rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = rouge.score(reference_text, predicted_text)

        results.append({
            "input_text": input_text,
            "predicted_text": predicted_text,
            "reference_text": reference_text,
            "confidence": confidence_scores,
            "bleu_score": bleu_score,
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
        })

    total_time = time.time() - start_time
    return results, total_time


testing_results, testing_time = evaluate_model(testing_dataset, model, tokenizer)
pd.DataFrame(testing_results).to_csv("testing_results.csv", index=False)
print(f"Testing completed in {testing_time:.2f} seconds.")

evaluation_results, evaluation_time = evaluate_model(evaluation_dataset, model, tokenizer)
pd.DataFrame(evaluation_results).to_csv("evaluation_results.csv", index=False)
print(f"Evaluation completed in {evaluation_time:.2f} seconds.")

print("Processing, fine-tuning, testing, and evaluation completed.")
