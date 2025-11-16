from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import torch

# Load dataset
df = pd.read_csv("data/overall_ds.csv", encoding="ISO-8859-1")

df = df.rename(columns={
    "Input (Raw Sentence)": "source",
    "Target (Enhanced Resume Sentence)": "target"
})

df = df.dropna(subset=["source", "target"])

# Add prefix for clear learning
df["source"] = "enhance: " + df["source"].astype(str)

dataset = Dataset.from_pandas(df)

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

MAX_LEN = 128


def preprocess(example):
    # Tokenize inputs
    model_inputs = tokenizer(
        example["source"],
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )

    # Tokenize labels
    labels = tokenizer(
        example["target"],
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )["input_ids"]

    # Mask padding tokens with -100
    labels = [(lid if lid != tokenizer.pad_token_id else -100) for lid in labels]

    model_inputs["labels"] = labels
    return model_inputs


# Apply map (batched=False to avoid nested lists!)
dataset = dataset.map(preprocess)

# Training arguments
args = TrainingArguments(
    output_dir="resume_enhancer_t5_final",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=20,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

# Train
trainer.train()

# Save
trainer.save_model("resume_enhancer_t5_final")
tokenizer.save_pretrained("resume_enhancer_t5_final")

print("🎉 Training Complete! Model saved inside resume_enhancer_t5_final/")
