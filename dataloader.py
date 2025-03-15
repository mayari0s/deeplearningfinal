from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch

current_path = os.getcwd()
data_path = os.path.join(current_path, "data")
full_path = os.path.join(data_path, "full_dataset.json")  

print("Full Path:", full_path)


with open(full_path, "r", encoding="utf-8") as f:
    full_data = json.load(f) 

full_dataset = Dataset.from_list(full_data)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["english"], truncation=True, padding="max_length", max_length=100)

tokenized_dataset = full_dataset.map(tokenize_function, batched=True)

split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

full_dataset_dict = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"]
})

full_dataset_dict.set_format(type="torch", columns=["input_ids", "attention_mask"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

train_dataloader = DataLoader(
    full_dataset_dict["train"],
    shuffle=True,
    batch_size=16, 
    collate_fn=data_collator
)

test_dataloader = DataLoader(
    full_dataset_dict["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator
)

train_iterator = iter(train_dataloader)
sample_batch = next(train_iterator)

print("Sample batch keys:", sample_batch.keys())
print("Input IDs shape:", sample_batch["input_ids"].shape)
