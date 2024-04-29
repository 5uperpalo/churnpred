import os
import sys

import torch
import pandas as pd
from transformers import (
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader

from churn_pred import utils
from churn_pred.preprocessing import surname_classification

print(os.getcwd())
surname_data = pd.read_csv("data/surnames/surname-nationality.csv")
splitted_data = pd.read_csv("data/surnames/surnames_with_splits.csv")


labeld_decoder = dict(
    zip(splitted_data["nationality"], splitted_data["nationality_index"])
)
inverse_label_dict = {v: k for k, v in labeld_decoder.items()}


# Split data into train, validation, and test sets
train_df = splitted_data[splitted_data["split"] == "train"]
val_df = splitted_data[
    splitted_data["split"] == "val"
]  # Assuming "val" is used for validation set
test_df = splitted_data[splitted_data["split"] == "test"]


# Tokenize each dataset
train_encodings = surname_classification.tokenize_data(train_df)
val_encodings = surname_classification.tokenize_data(val_df)
test_encodings = surname_classification.tokenize_data(test_df)


batch_size = 32

train_dataset = surname_classification.create_dataset(train_encodings, train_df)
val_dataset = surname_classification.create_dataset(val_encodings, val_df)
test_dataset = surname_classification.create_dataset(test_encodings, test_df)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


num_labels = splitted_data["nationality_index"].nunique()
num_epochs = 10
model = BertForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=num_labels, hidden_dropout_prob=0.4
)
# freeze layers
for param in model.bert.parameters():
    param.requires_grad = False

optimizer = surname_classification.create_optimizer(model, learning_rate=5e-5, eps=1e-8)
total_steps = len(train_dataloader) * num_epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.2,  # Usually a fraction of total_steps
    num_training_steps=total_steps,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


trained_model = surname_classification.train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    device,
    num_epochs=num_epochs,
)


loss, accuracy = surname_classification.evaluate_model(
    trained_model, test_dataloader, device
)
print(f"Evaluation on test dataset yield\nLoss: {loss}\nAccuracy: {accuracy}")


# Save the model
# torch.save(trained_model.state_dict(), "data/surnames/surname_model_state_dict.pth")
torch.save(trained_model, "data/surnames/surname_model.pth")
utils.json_dump(
    file_loc="data/surnames/inverse_label_dict.json", content=inverse_label_dict
)
