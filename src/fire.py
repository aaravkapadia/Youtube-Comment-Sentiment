from datasets import load_dataset
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer, RobertaForSequenceClassification, AutoConfig, TrainingArguments, Trainer
import torch
import evaluate
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CommentDataset(Dataset):
    
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.tokens['input_ids'][idx],
            'attention_mask': self.tokens['attention_mask'][idx],
            'labels': self.labels[idx]
        } 

df = pd.read_csv('datasets/comments.csv')
df = df.sample(100_000, random_state=42)
train_df = df.sample(frac = 0.9, random_state=42)
test_df = df.drop(train_df.index)
print(df.head())

tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
train_tokens = tokenizer(
    list(train_df['CommentText']),
    padding = True,
    truncation = True,
    max_length = 128,
    return_tensors = 'pt'
)
test_tokens = tokenizer(
    list(test_df['CommentText']),
    padding = True,
    truncation = True,
    max_length = 128,
    return_tensors = 'pt'
)
label_to_id = {"Negative": 0, "Neutral": 1, "Positive" : 2}
train_labels = torch.tensor(train_df['Sentiment'].map(label_to_id).values)
test_labels = torch.tensor(test_df['Sentiment'].map(label_to_id).values)
train_dataset = CommentDataset(train_tokens, train_labels)
test_dataset = CommentDataset(test_tokens, test_labels)

model = RobertaForSequenceClassification.from_pretrained('FacebookAI/roberta-base', num_labels = 3)
model = model.to(device)
metric = evaluate.load("accuracy")

def compute_metric(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir='results/',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metric
)

trainer.train()