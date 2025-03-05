import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchmetrics.classification import Accuracy, F1Score

n_epochs = 1
test_loss = []
train_loss = []

model = SimpleRNNClassifier(vocab_size=vocab_size, embed_dim=50, output_dim=1, pad_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCELoss()

accuracy_metric = Accuracy()
f1_metric = F1Score()

for epoch in range(n_epochs):
    model.train()
    _loss = []
    
    # Training Loop
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        output = model(batch['input_ids'])
        loss = loss_fn(output, batch['labels'].float())
        loss.backward()
        optimizer.step()
        _loss.append(loss.item())
    
    # Save the last 10 training losses
    train_loss.append(np.mean(_loss[-10:]))
    
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        
        # Evaluation Loop
        for test_batch in test_dataloader:
            test_output = model(test_batch['input_ids'])
            test_label = test_batch['labels'].float()
            loss = loss_fn(test_output, test_label)
            test_loss.append(loss.item())
            
            # Store predictions and true labels for metrics
            all_preds.append(test_output)
            all_labels.append(test_label)

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # Calculate accuracy and F1 score
        accuracy = accuracy_metric(all_preds.round(), all_labels)
        f1 = f1_metric(all_preds.round(), all_labels)
        
    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss[-1]:.4f} - Val Loss: {np.mean(test_loss[-len(test_dataloader):]):.4f}")
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

