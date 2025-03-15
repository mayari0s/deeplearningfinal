{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "40a2f08a-598d-4cf5-88e0-e00116bd779a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2.4.1+cu121\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import sys\n",
    "import torch\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import json\n",
    "from datasets import Dataset, DatasetDict\n",
    "from transformers import AutoTokenizer, DataCollatorWithPadding\n",
    "from torch.utils.data import DataLoader\n",
    "import torch\n",
    "os.environ[\"TOKENIZERS_PARALLELISM\"] = \"false\"\n",
    "\n",
    "print(torch.__version__)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9b9f5d7f-e148-407c-9c13-cbc2ae77aaac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "books Path: /gpfs/home/mrios2/Desktop/Untitled Folder 1/cleaned_books_data.json\n"
     ]
    }
   ],
   "source": [
    "current_path = os.getcwd() \n",
    "data_path = os.path.join(current_path, \"data\")\n",
    "books = os.path.join(current_path, \"cleaned_books_data.json\") \n",
    "print(\"books Path:\", books)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "84d13b16-5fa1-4185-8282-a2ecfb49d65e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "news Path: /gpfs/home/mrios2/Desktop/Untitled Folder 1/cleaned_news_data.json\n"
     ]
    }
   ],
   "source": [
    "current_path = os.getcwd()\n",
    "news_path = os.path.join(current_path, \"data\")\n",
    "news = os.path.join(current_path, \"cleaned_news_data.json\") \n",
    "print(\"news Path:\", news)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "13ff1040-fdd0-40d9-853b-f535fd498385",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Books Path: /gpfs/home/mrios2/Desktop/Untitled Folder 1/data/cleaned_books_data.json\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e0eece8c49484d2eb348a63462c9311e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/93470 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sample batch keys: dict_keys(['input_ids', 'attention_mask'])\n",
      "Input IDs shape: torch.Size([16, 100])\n"
     ]
    }
   ],
   "source": [
    "acurrent_path = os.getcwd()\n",
    "data_path = os.path.join(acurrent_path, \"data\")\n",
    "books_path = os.path.join(data_path, \"cleaned_books_data.json\")\n",
    "\n",
    "print(\"Books Path:\", books_path)\n",
    "\n",
    "with open(books_path, \"r\", encoding=\"utf-8\") as f:\n",
    "    books = json.load(f)\n",
    "\n",
    "books_dataset = Dataset.from_list(books)\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(\"distilbert-base-uncased\")\n",
    "\n",
    "def tokenize_function(example):\n",
    "    return tokenizer(example[\"english\"], truncation=True, padding=\"max_length\", max_length=100)\n",
    "\n",
    "tokenized_books = books_dataset.map(tokenize_function, batched=True)\n",
    "\n",
    "split_dataset = tokenized_books.train_test_split(test_size=0.1)\n",
    "\n",
    "books = DatasetDict({\n",
    "    \"train\": split_dataset[\"train\"],\n",
    "    \"test\": split_dataset[\"test\"]\n",
    "})\n",
    "\n",
    "books.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\"])\n",
    "\n",
    "data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors=\"pt\")\n",
    "\n",
    "train_dataloader = DataLoader(\n",
    "    books[\"train\"],\n",
    "    shuffle=True,\n",
    "    batch_size=16,\n",
    "    collate_fn=data_collator\n",
    ")\n",
    "\n",
    "test_dataloader = DataLoader(\n",
    "    books[\"test\"],\n",
    "    shuffle=False,\n",
    "    batch_size=16,\n",
    "    collate_fn=data_collator\n",
    ")\n",
    "\n",
    "train_iterator = iter(train_dataloader)\n",
    "sample_batch = next(train_iterator)\n",
    "\n",
    "print(\"Sample batch keys:\", sample_batch.keys())\n",
    "print(\"Input IDs shape:\", sample_batch[\"input_ids\"].shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1975a005-e996-4939-86ee-2d8964864c20",
   "metadata": {},
   "outputs": [],
   "source": [
    "current_path = os.getcwd()\n",
    "data_path = os.path.join(current_path, \"data\")\n",
    "news_path = os.path.join(data_path, \"cleaned_news_data.json\")\n",
    "\n",
    "print(\"News Path:\", news_path)\n",
    "\n",
    "with open(news_path, \"r\", encoding=\"utf-8\") as f:\n",
    "    news = json.load(f)\n",
    "\n",
    "news_dataset = Dataset.from_list(news)\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(\"distilbert-base-uncased\")\n",
    "\n",
    "def tokenize_function(example):\n",
    "    return tokenizer(example[\"english\"], truncation=True, padding=\"max_length\", max_length=100)\n",
    "\n",
    "tokenized_news = news_dataset.map(tokenize_function, batched=True)\n",
    "\n",
    "split_dataset = tokenized_news.train_test_split(test_size=0.1)\n",
    "\n",
    "news = DatasetDict({\n",
    "    \"train\": split_dataset[\"train\"],\n",
    "    \"test\": split_dataset[\"test\"]\n",
    "})\n",
    "\n",
    "news.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\"])\n",
    "\n",
    "data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors=\"pt\")\n",
    "\n",
    "train_dataloader = DataLoader(\n",
    "    news[\"train\"],\n",
    "    shuffle=True,\n",
    "    batch_size=16,\n",
    "    collate_fn=data_collator\n",
    ")\n",
    "\n",
    "test_dataloader = DataLoader(\n",
    "    news[\"test\"],\n",
    "    shuffle=False,\n",
    "    batch_size=16,\n",
    "    collate_fn=data_collator\n",
    ")\n",
    "\n",
    "train_iterator = iter(train_dataloader)\n",
    "sample_batch = next(train_iterator)\n",
    "\n",
    "print(\"Sample batch keys:\", sample_batch.keys())\n",
    "print(\"Input IDs shape:\", sample_batch[\"input_ids\"].shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7830a712-4aac-4438-98b7-be427ead8038",
   "metadata": {},
   "outputs": [],
   "source": [
    "full_dataset = torch.utils.data.ConcatDataset([books_dataset, news_dataset])\n",
    "train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d77ec64d-db40-4f74-aa29-bf5e03d93974",
   "metadata": {},
   "outputs": [],
   "source": [
    "full_dataset_list = [dataset[i] for dataset in full_dataset.datasets for i in range(len(dataset))]\n",
    "\n",
    "json_filename = \"full_dataset.json\"\n",
    "with open(json_filename, \"w\", encoding=\"utf-8\") as json_file:\n",
    "    json.dump(full_dataset_list, json_file, ensure_ascii=False, indent=4)\n",
    "\n",
    "print(f\"Dataset saved as {json_filename}\")\n",
    "with open(json_filename, \"r\", encoding=\"utf-8\") as f:\n",
    "    full_data = json.load(f)  \n",
    "\n",
    "print(\"First item in dataset:\", full_data[0])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "977dbd27-0920-413b-a14b-bd9e3d422ea4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "from datasets import Dataset, DatasetDict\n",
    "from transformers import AutoTokenizer, DataCollatorWithPadding\n",
    "from torch.utils.data import DataLoader\n",
    "\n",
    "current_path = os.getcwd()\n",
    "data_path = os.path.join(current_path, \"data\")\n",
    "full_path = os.path.join(data_path, \"full_dataset.json\")\n",
    "\n",
    "print(\"Full Path:\", full_path)\n",
    "\n",
    "with open(full_path, \"r\", encoding=\"utf-8\") as f:\n",
    "    full_data = json.load(f)\n",
    "\n",
    "full_dataset = Dataset.from_list(full_data)\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(\"distilbert-base-uncased\")\n",
    "\n",
    "def tokenize_function(example):\n",
    "    return tokenizer(example[\"english\"], truncation=True, padding=\"max_length\", max_length=100)\n",
    "\n",
    "tokenized_dataset = full_dataset.map(tokenize_function, batched=True)\n",
    "\n",
    "split_dataset = tokenized_dataset.train_test_split(test_size=0.1)\n",
    "\n",
    "full_dataset_dict = DatasetDict({\n",
    "    \"train\": split_dataset[\"train\"],\n",
    "    \"test\": split_dataset[\"test\"]\n",
    "})\n",
    "\n",
    "full_dataset_dict.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\"])\n",
    "\n",
    "data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors=\"pt\")\n",
    "\n",
    "train_dataloader = DataLoader(\n",
    "    full_dataset_dict[\"train\"],\n",
    "    shuffle=True,\n",
    "    batch_size=16,\n",
    "    collate_fn=data_collator\n",
    ")\n",
    "\n",
    "test_dataloader = DataLoader(\n",
    "    full_dataset_dict[\"test\"],\n",
    "    shuffle=False,\n",
    "    batch_size=16,\n",
    "    collate_fn=data_collator\n",
    ")\n",
    "\n",
    "train_iterator = iter(train_dataloader)\n",
    "sample_batch = next(train_iterator)\n",
    "\n",
    "print(\"Sample batch keys:\", sample_batch.keys())\n",
    "print(\"Input IDs shape:\", sample_batch[\"input_ids\"].shape)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
