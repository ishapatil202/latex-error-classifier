import os
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
import numpy as np
import json

# 1. Load Dataset
dataset = load_dataset("json", data_files="latex_error_dataset_with_no_error.jsonl")["train"]
print("Sample:", dataset[0])

# 2. Encode Labels
label_encoder = LabelEncoder()
label_encoder.fit(dataset["label"])
dataset = dataset.map(lambda x: {"label": label_encoder.transform([x["label"]])[0]})
label_list = list(label_encoder.classes_)
num_labels = len(label_list)
print("Label classes:", label_list)

# 3. Compute Class Weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_labels), y=dataset["label"])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# 4. Tokenize Inputs (Only Problem + Student Answer)
tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset, test_dataset = split["train"], split["test"]

# 5. Wrap with PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, hf_dataset): self.dataset = hf_dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.dataset[idx]["input_ids"]),
            "attention_mask": torch.tensor(self.dataset[idx]["attention_mask"]),
            "labels": torch.tensor(self.dataset[idx]["label"])
        }

train_dataset = CustomDataset(train_dataset)
test_dataset = CustomDataset(test_dataset)

# 6. Define Two-Layer MathBERT Classifier
class MathBERTTwoLayerClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

model = MathBERTTwoLayerClassifier("tbs17/MathBERT", num_labels)

# 7. Trainer Setup
training_args = TrainingArguments(
    output_dir="./bert_latex_classifier",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    logging_steps=10,
    save_strategy="no",
    logging_dir="./logs",
    load_best_model_at_end=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 8. Train and Save
trainer.train()
os.makedirs("bert_latex_classifier", exist_ok=True)
torch.save(model.state_dict(), "bert_latex_classifier/pytorch_model.bin")
tokenizer.save_pretrained("bert_latex_classifier")

print(" Training complete and model saved.")

