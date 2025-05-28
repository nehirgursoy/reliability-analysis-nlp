"""
Decision-Level Divergence Analysis

This script analyzes prediction disagreement between the emotion-aware and baseline models on
identical input samples. It quantifies how often the models make different decisions and inspects
cases where +emotion-aware models correct or introduce misclassifications.
To understand how emotional context affects decision-making at the individual instance level.

Additionally, it uses LIME to visualize word-level feature importances for disagreement cases, providing insight 
into how emotional signals influence model behavior and decision-making.

"""
#%%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score
from lime.lime_text import LimeTextExplainer
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%

def sample_per_task(df, task_col='task', sample_size=2000):
    sampled_df = pd.concat([
        group.sample(n=sample_size, random_state=42) 
        for _, group in df.groupby(task_col)
        if len(group) >= sample_size
    ]).reset_index(drop=True)
    return sampled_df

def create_dataloader_from_df(df, tokenizer, max_seq_len=512, batch_size=32):
    encodings = tokenizer.batch_encode_plus(
        df['text'].tolist(),
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    labels = torch.tensor(df['label'].tolist())
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def decision_level_divergence_analysis(model_emotion, model_baseline, dataloader, device):
    model_emotion.eval()
    model_baseline.eval()

    total_samples = 0
    diff_count = 0
    emotion_corrects = 0
    emotion_wrong_introduced = 0

    all_diff_indices = []

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs_emotion = model_emotion(input_ids, attention_mask)
            outputs_baseline = model_baseline(input_ids, attention_mask)

            preds_emotion = torch.argmax(outputs_emotion, dim=1)
            preds_baseline = torch.argmax(outputs_baseline, dim=1)

            for i in range(len(labels)):
                total_samples += 1
                if preds_emotion[i] != preds_baseline[i]:
                    diff_count += 1
                    idx_global = batch_idx * dataloader.batch_size + i
                    all_diff_indices.append(idx_global)

                    # Emotion model doğru, baseline yanlışsa düzeltmiş demektir
                    if preds_emotion[i] == labels[i] and preds_baseline[i] != labels[i]:
                        emotion_corrects += 1

                    # Emotion model yanlış, baseline doğruysa yeni hata yapmış
                    if preds_emotion[i] != labels[i] and preds_baseline[i] == labels[i]:
                        emotion_wrong_introduced += 1

    diff_ratio = diff_count / total_samples
    return {
        'total_samples': total_samples,
        'diff_count': diff_count,
        'diff_ratio': diff_ratio,
        'emotion_corrects': emotion_corrects,
        'emotion_wrong_introduced': emotion_wrong_introduced,
        'diff_indices': all_diff_indices
    }
#%%
def predict_proba_fn_lime(model, tokenizer):
    def wrapped(texts):
        model.eval()
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu().numpy()
    return wrapped

def explain_with_lime(model, tokenizer, text, class_names=['not spam', 'spam'], output_dir='lime_outputs'):
    os.makedirs(output_dir, exist_ok=True)
    explainer = LimeTextExplainer(class_names=class_names)
    pred_fn = predict_proba_fn_lime(model, tokenizer)
    exp = explainer.explain_instance(text, pred_fn, num_features=10)
    filename = os.path.join(output_dir, f'lime_{hash(text)}.html')
    exp.save_to_file(filename)
    print(f"LIME saved to {filename}")
    return exp
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
#%%
model_name = 'FacebookAI/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

def load_model(weight_path):
    model = ClassifierHead(base_model, model_name)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    return model

with_emotion_path = "../finetuned_models/with_emotion.pt"
without_emotion_path = "../finetuned_models/without_emotion.pt"

model_with = load_model(with_emotion_path)
model_without = load_model(without_emotion_path)
#%%
df = pd.read_csv('../data/merged_datas_with_emotion.csv')  # text, label, task, emotion vs.

sampled_df = sample_per_task(df, task_col='task', sample_size=2000)
print(f"Total samples: {len(sampled_df)}")

test_dataloader = create_dataloader_from_df(sampled_df, tokenizer, max_seq_len=512, batch_size=32)

results = decision_level_divergence_analysis(model_with, model_without, test_dataloader, device)

print(f"Total number of samples: {results['total_samples']}")
print(f"Number of differing predictions: {results['diff_count']}")
print(f"Percentage of differing predictions: {results['diff_ratio']*100:.2f}%")
print(f"Number of corrections by emotion-aware model: {results['emotion_corrects']}")
print(f"Number of new errors introduced by emotion-aware model: {results['emotion_wrong_introduced']}")

#%%
diff_texts = sampled_df.iloc[results['diff_indices']]['text'].tolist()

print("\n LIME Explanations for Disagreement Cases")
for i, text in enumerate(diff_texts[:5]):
    print(f"\nExample {i+1}:")
    print(text)
    explain_with_lime(model_with, tokenizer, text)