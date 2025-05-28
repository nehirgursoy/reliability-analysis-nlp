"""
Evaluation and Performance Divergence Analysis

This script compares the emotion-aware and emotion-unaware fine-tuned models across multiple tasks
(spam detection, fake news detection, and AI-generated text detection). It calculates performance
metrics (accuracy, F1) and quantifies divergence in task-wise performance.

To measure the overall impact of emotion labels on classification effectiveness and identify
task-specific differences in model behavior.
"""
#%%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
#Load Data
df = pd.read_csv("../data/merged_datas_no_emotion.csv")
test_texts = df["text"].tolist()
test_labels = df["label"].tolist()

# Tokenizer ve base model
model_name = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

# Tokenize
def tokenize_texts(texts, max_len=512):
    tokens = tokenizer.batch_encode_plus(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return tokens['input_ids'], tokens['attention_mask']

input_ids, attention_mask = tokenize_texts(test_texts)
labels_tensor = torch.tensor(test_labels)

#%%
# Model class
class ClassifierHead(nn.Module):
    def __init__(self, transformer_model, model_name, dropout=0.1, n_classes=2):
        super().__init__()
        self.transformer = transformer_model
        hidden_size = transformer_model.config.hidden_size

        self.fc1 = nn.Linear(hidden_size, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        if cls_output is None:
            cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.fc1(cls_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def load_model(weight_path):
    model = ClassifierHead(base_model, model_name)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


with_emotion_path = "../finetuned_models/with_emotion.pt"
without_emotion_path = "../finetuned_models/without_emotion.pt"

model_with = load_model(with_emotion_path)
model_without = load_model(without_emotion_path)

#%%
def predict(model, input_ids, attention_mask):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_ids), 32):
            ids = input_ids[i:i+32].to(device)
            mask = attention_mask[i:i+32].to(device)
            output = model(ids, mask)
            pred = torch.argmax(output, dim=1)
            preds.extend(pred.cpu().numpy())
    return preds

# Prediction
preds_with = predict(model_with, input_ids, attention_mask)
preds_without = predict(model_without, input_ids, attention_mask)

disagreements = np.array(preds_with) != np.array(preds_without)
disagreement_rate = np.mean(disagreements)

print("\nDisagreement Rate: {:.4f}".format(disagreement_rate))

print("\n== WITH EMOTION ==")
print("Accuracy: {:.4f}".format(accuracy_score(test_labels, preds_with)))
print("F1 Score: {:.4f}".format(f1_score(test_labels, preds_with, average="weighted")))
print(classification_report(test_labels, preds_with, digits=4))

print("\n== WITHOUT EMOTION ==")
print("Accuracy: {:.4f}".format(accuracy_score(test_labels, preds_without)))
print("F1 Score: {:.4f}".format(f1_score(test_labels, preds_without, average="weighted")))
print(classification_report(test_labels, preds_without, digits=4))
