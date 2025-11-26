# train_gramformer.py
import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# -------- CONFIG --------
DATA_PATH = "data/overall_ds.csv"
MODEL_NAME = "t5-small"        # small & fast; replace with larger if you have GPU
OUTPUT_DIR = "gramformer_lora"
EPOCHS = 4
BATCH_SIZE = 8                 # lower if you run out of memory
LR = 3e-4
MAX_LEN = 128

# -------- load csv and normalize columns ----------
df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
# rename to common names:
df = df.rename(columns={
    "Input (Raw Sentence)": "source",
    "Target (Enhanced Resume Sentence)": "target"
})
df = df.dropna(subset=["source","target"]).reset_index(drop=True)

# Add task prefix for T5 style
df["source"] = "enhance: " + df["source"].astype(str).str.strip()

# convert pandas to HF dataset
dataset = Dataset.from_pandas(df)

# -------- tokenizer & model ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# -------- preprocess (non-batched to avoid nested lists issues) ----------
def preprocess(example):
    inp = tokenizer(example["source"], truncation=True, padding="max_length", max_length=MAX_LEN)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["target"], truncation=True, padding="max_length", max_length=MAX_LEN)
    # convert pad token ids in labels to -100 for loss ignoring
    labels_ids = [(lid if lid != tokenizer.pad_token_id else -100) for lid in labels["input_ids"]]
    inp["labels"] = labels_ids
    return inp

dataset = dataset.map(preprocess)

# remove unnecessary columns, keep only features needed
dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ["input_ids","attention_mask","labels"]])

# -------- PEFT LoRA config (efficient fine-tuning) ----------
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)

# -------- training args ----------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=50,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),   # use fp16 if GPU available
    predict_with_generate=True,
    remove_unused_columns=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -------- train ----------
trainer.train()
# save peft adapter + base tokenizer/config
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved model to", OUTPUT_DIR)
