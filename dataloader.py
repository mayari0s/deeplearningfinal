import json
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# Define the paths to the cleaned data files
books_path = "/Users/mayarios/Desktop/deeplearning/project/cleaned_books_data.json"
news_path = "/Users/mayarios/Desktop/deeplearning/project/cleaned_news_data.json"

# Load the books data
with open(books_path, 'r') as f:
    books = json.load(f)

# Load the news data
with open(news_path, 'r') as f:
    news = json.load(f)

# Initialize the tokenizer (use the appropriate model name, e.g., "bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the books dataset (assuming 'text' column is what you want to tokenize)
tokenized_books = Dataset.from_dict({
    'text': [book['text'] for book in books]
})

# Tokenizing the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

# Apply the tokenizer to the books dataset
tokenized_books = tokenized_books.map(tokenize_function, batched=True)

# Split the tokenized dataset into train and test
split_dataset = tokenized_books.train_test_split(test_size=0.1)

# Create a new DatasetDict with the train-test split
books = DatasetDict({
    'train': split_dataset['train'],
    'test': split_dataset['test']
})

# Initialize the data collator (to pad sequences dynamically during batching)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create the train DataLoader
train_dataloader = DataLoader(
    books['train'].remove_columns(["english", "spanish"]),  # Remove unnecessary columns
    shuffle=True,
    batch_size=256,
    collate_fn=data_collator
)

# Create the test DataLoader
test_dataloader = DataLoader(
    books['test'].remove_columns(["english", "spanish"]),  # Remove unnecessary columns
    shuffle=False,
    batch_size=256,
    collate_fn=data_collator
)

# Print the results
print("Train DataLoader ready with batch size:", train_dataloader.batch_size)
print("Test DataLoader ready with batch size:", test_dataloader.batch_size)
