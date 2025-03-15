# Deep Learning Final Project: Context-Aware Literary Translation Enhancement

## Project Overview

This project leverages deep learning techniques to classify text data based on its origin—whether it comes from Spanish or English literature and news sources. This model aims to train the model on these literature and news tokens to predict whether a user input word or sentence comes from a fact or fictional source. 

The model's Recurrent Neural Network tokenizes Spanish and English news sentences from the European Union as a basis for its factual text classification. The project aims to demonstrate how deep learning models can be applied to natural language processing (NLP) tasks, specifically in text classification. It focuses on building a robust text classifier capable of distinguishing between English and Spanish text, which has applications in language processing, content categorization, and multilingual NLP systems.

## Data Overview

The dataset used in this project consists of text samples collected from English and Spanish literature and news sources. The data is structured as follows:

- **Literature Text**: A collection of text samples from English and Spanish novels within the public domain https://opus.nlpl.eu/sample/en&es/Books&v1/sample 
- **News Text**: A collection of text samples from English and Spanish news articles. https://opus.nlpl.eu/ELRC-1125-CORDIS_News/en&es/v1/ELRC-1125-CORDIS_News 

Each text sample is labeled with its corresponding language (`English` or `Spanish`) and combined into one JSON file. The dataset is split into training and testing sets to evaluate the model's performance. The model also takes twelve books published between the seventeenth and twentieth century, available in both Spanish and English within the public domain, as the basis for its fictional translation. 


## Instructions

Make sure the following packages are installed:
  - Torch
  - datasets
  - transformers
  - torchmetrics
  - matplotlib
  - tqdm
  - numpy
  - json
  - sys
  - numpy
  - pandas

1. Tokenize the dataset using the full_dataset.json file
2. Convert the list of dictionaries into a Hugging Face Dataset
3. Apply the tokenization function to the text and make sure the dataset is defined
4. Convert dataset to PyTorch format
5. Create DataLoader for the full dataset
6. Set RNN to cuda (if available) and load RNN model
7. Run training loop on 5 epochs
8. Plot the training loss
9. Employ the predict_text_class function to imput word or sentence to predict text sorce




## Results

I evaluated this model using Accuracy, F1 score, and loss function. Accuracy will measure the percentage of correctly classified texts (news vs. literature). F1 score will be a balanced measure of precision and recall used for ensuring that both categories are classified correctly. The loss function will monitor the loss during training and will help ensure the model is learning effectively without overfitting.

The results of the training and evaluation model was impressive: Accuracy: 99.2% after five Epochs, F1 Score: 0.9916, Train Loss: 0.0428. However, this did not translate to the post-training classifier algorithm. 

Despite achieving high accuracy in classifying text as either literature or news sources, the model exhibited significant limitations due to overfitting. The recurrent neural network (RNN) architecture led to a high degree of memorization, preventing the model from generalizing effectively to new, unseen data. Instead of learning underlying patterns, the model retained training examples too rigidly, resulting in poor adaptability to novel inputs.  

The RNN was vulnerable to exploding gradients, which contributed to an unstable training process. This issue led to a high loss function, further reducing the model’s effectiveness in real-world text classification tasks. While the training and evaluation algorithms performed well on the given dataset, the model's inability to generalize suggests that improvements such as regularization techniques, increased dataset diversity, or alternative architectures (e.g., transformers) may be necessary to enhance performance. The largest limitation of this model is that the high accuracy indicates that the model effectively distinguished tokenized text as literature or news but does not guarantee its reliability in real-world applications with new user-generated input.  





