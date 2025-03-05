import json
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from pathlib import Path

base_dir = Path.home() / "Desktop" / "deeplearning" / "project"

books_path = base_dir / "cleaned_books_data.json"
news_path = base_dir / "cleaned_news_data.json"


with open(books_path, 'r') as f:
    books = json.load(f)

with open(news_path, 'r') as f:
    news = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_books = books.map(tokenize_function, batched=True)

split_dataset = tokenized_books.train_test_split(test_size=0.1)

books = DatasetDict({
    'train': split_dataset['train'],
    'test': split_dataset['test']
})

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    books['train'].remove_columns(["english", "spanish"]),  
    shuffle=True,
    batch_size=256,
    collate_fn=data_collator
)

test_dataloader = DataLoader(
    books['test'].remove_columns(["english", "spanish"]), 
    shuffle=False,
    batch_size=256,
    collate_fn=data_collator
)

print("Train DataLoader ready with batch size:", train_dataloader.batch_size)
print("Test DataLoader ready with batch size:", test_dataloader.batch_size)
