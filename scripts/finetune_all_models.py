# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel

# specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

#%%
df = pd.read_csv("../data/amazon_review.csv") #Change based on the data 
df.head()
df.shape

#%%
#split dataset into train validation and test
train_text, temp_text, train_labels, temp_labels = train_test_split(
    df['text'], df['label'], 
    test_size=0.3, 
    random_state=42, 
    stratify=df['label'])

val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, temp_labels, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_labels)

# %%
# import pretrained model and its tokenizer
from transformers import AutoModel, AutoTokenizer

def load_model_and_tokenizer(model_name):
    print(f"Loading model and tokenizer for: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

models = {
    "1": "google-bert/bert-base-uncased",
    "2": "distilbert/distilbert-base-uncased",
    "3": "albert/albert-base-v2",
    "4": "FacebookAI/roberta-base",
    "5": "xlnet/xlnet-base-cased"
}

print("Select a model:")
for k, v in models.items():
    print(f"{k}: {v}")

while True:
    choice = input("Enter the number of the model you want to use: ").strip()
    if choice in models:
        model_name = models[choice]
        break
    else:
        print("This model number does not exist. Please try again.")

model, tokenizer = load_model_and_tokenizer(model_name)

# %%
# tokenize and encode sequences
max_seq_len = 512
def tokenize_data(tokenizer, texts, max_seq_len):
    return tokenizer.batch_encode_plus(
        texts.tolist(),
        max_length=max_seq_len,
        padding='max_length',      
        truncation=True,           
        return_token_type_ids=False, 
        return_attention_mask=True,
        return_tensors='pt'         
        )

tokens_train = tokenize_data(tokenizer, train_text, max_seq_len)
tokens_val = tokenize_data(tokenizer, val_text, max_seq_len)
tokens_test = tokenize_data(tokenizer, test_text, max_seq_len)
#%%

train_seq = tokens_train['input_ids']          
train_mask = tokens_train['attention_mask']    
train_y = torch.tensor(train_labels.tolist())

val_seq = tokens_val['input_ids']
val_mask = tokens_val['attention_mask']
val_y = torch.tensor(val_labels.tolist())

test_seq = tokens_test['input_ids']
test_mask = tokens_test['attention_mask']
test_y = torch.tensor(test_labels.tolist())
#%%
#Create Data Loaders
from torch.utils.data import TensorDataset, DataLoader

def create_dataloader(input_ids, attention_mask, labels, batch_size=32, shuffle=False):
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

train_dataloader = create_dataloader(train_seq, train_mask, train_y, batch_size=32, shuffle=True)

val_dataloader = create_dataloader(val_seq, val_mask, val_y, batch_size=32, shuffle=False)
test_dataloader = create_dataloader(test_seq, test_mask, test_y, batch_size=32, shuffle=False)

# %%
#Freeze parameters
def freeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = False

# %%
import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, transformer_model, model_name, dropout=0.1, n_classes=2):
        super(ClassifierHead, self).__init__()
        self.transformer = transformer_model
        self.model_name = model_name.lower()
        hidden_size = transformer_model.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input_ids, attention_mask):
        if 'distilbert' in self.model_name:
            outputs = self.transformer(input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # DistilBERT uses first token embedding
        elif 'xlnet' in self.model_name:
            outputs = self.transformer(input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
        else:
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
from torch.optim import AdamW
from transformers import get_scheduler

epochs = 3
gradient_accumulation_steps = 2

classifier_model = ClassifierHead(model, model_name)
classifier_model.to(device)

optimizer = AdamW(classifier_model.parameters(), lr=2e-5, eps=1e-8)

num_training_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# %%
import time
from sklearn.metrics import accuracy_score

cross_entropy = nn.CrossEntropyLoss()

# train function
def train():
    classifier_model.train()
    total_loss = 0
    total_preds = []

    for step, batch in enumerate(train_dataloader):
        batch = [r.to(device) for r in batch]
        input_ids, attention_mask, labels = batch

        outputs = classifier_model(input_ids, attention_mask)
        loss = cross_entropy(outputs, labels)
        loss = loss / gradient_accumulation_steps
        total_loss += loss.item()

        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            classifier_model.zero_grad()

        preds = outputs.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# evaluation function
from sklearn.metrics import f1_score, accuracy_score

def evaluate(dataloader):
    classifier_model.eval()
    total_loss = 0
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = [t.to(device) for t in batch]
            input_ids, attention_mask, labels = batch

            outputs = classifier_model(input_ids, attention_mask)
            loss = cross_entropy(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds, average='weighted')
    return avg_loss, acc, f1

# %%
# training loop
best_valid_loss = float('inf')
train_losses, valid_losses = [], []

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')
    print('-' * 30)

    train_loss, _ = train()
    val_loss, val_acc, val_f1 = evaluate(val_dataloader)

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(classifier_model.state_dict(), 'best_model.pt')
        print("✅ Best model saved!")

    train_losses.append(train_loss)
    valid_losses.append(val_loss)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

# %%
# load best model and evaluate on test set
classifier_model.load_state_dict(torch.load('best_model.pt'))
classifier_model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_dataloader:
        batch = [t.to(device) for t in batch]
        input_ids, attention_mask, labels = batch

        outputs = classifier_model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds, average='weighted')

print("\n📊 Test Results")
print(f"Accuracy: {test_acc:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))
