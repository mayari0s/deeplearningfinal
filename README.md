# Deep Learning Final Project: Context-Aware Literary Translation Enhancement

## Project Overview

This project leverages deep learning techniques to classify text data based on its origin—whether it comes from Spanish or English literature and news sources. The model employs tokenization for pre-processing and a Simple Recurrent Neural Network (RNN) classifier to perform the classification task.

The project aims to demonstrate how deep learning models can be applied to natural language processing (NLP) tasks, specifically in text classification. It focuses on building a robust text classifier capable of distinguishing between English and Spanish text, which has applications in language processing, content categorization, and multilingual NLP systems.

## Data Overview

The dataset used in this project consists of text samples collected from English and Spanish literature and news sources. The data is structured as follows:

- **English Text**: A collection of text samples from English literature (e.g., novels, essays) and news articles.
- **Spanish Text**: A collection of text samples from Spanish literature and news sources.

Each text sample is labeled with its corresponding language (`English` or `Spanish`). The dataset is split into training and testing sets to evaluate the model's performance. Key characteristics of the dataset include:

The dataset is preprocessed using tokenization and sequence padding to prepare it for input into the RNN model.

## Key Features
- **Text Preprocessing**: Tokenization and sequence padding for input data.
- **Simple RNN Model**: A Recurrent Neural Network architecture for text classification.
- **Language Classification**: Distinguishes between English and Spanish text.

## Requirements
To run this project, you will need the following libraries:
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
