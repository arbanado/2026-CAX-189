from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = "./flan-market-research"
FINAL_DIR = "./flan-market-research-final"

# Load base model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Separate training and validation files. Held-out evaluation is NOT used here.
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train.jsonl",
        "validation": "data/validation.jsonl",
    },
)


def preprocess(batch):
    inputs = [
        f"Instruction: {instruction}\nContext: {context}"
        for instruction, context in zip(batch["instruction"], batch["context"])
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=384,
        truncation=True,
    )

    labels = tokenizer(
        text_target=batch["target"],
        max_length=160,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    predict_with_generate=True,
    logging_steps=50,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collator,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"Saved fine-tuned model to {FINAL_DIR}")
