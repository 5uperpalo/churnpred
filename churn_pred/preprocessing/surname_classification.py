# approach shamelessly copied from https://www.kaggle.com/code/yonatankpl/surname-classification-with-bert
# import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW, BertTokenizer
from torch.utils.data import TensorDataset


def tokenize_surnames(surnames, max_length=128):
    """
    Tokenizes the surnames.
    Args:
    - surnames: List of surnames.
    - max_length: Maximum sequence length for tokenization.

    Returns:
    - Input IDs and attention masks.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    encoded_dict = tokenizer.batch_encode_plus(
        surnames,
        add_special_tokens=True,
        max_length=max_length,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoded_dict["input_ids"]
    attention_masks = encoded_dict["attention_mask"]

    return input_ids, attention_masks


def defreeze_all_bert_layers(model):
    """
    Defreezes all layers of the BERT model.
    Args:
    - model: The BERT model.
    """
    for param in model.bert.parameters():
        param.requires_grad = True


def train_model(
    model,
    train_dataloader,
    validation_dataloader,
    optimizer,
    scheduler,
    device,
    num_epochs=10,
    specific_epoch_to_defreeze=5,
):
    """
    Trains and evaluates the BERT model.
    Args:
    - model: The BERT model for classification.
    - train_dataloader: DataLoader for the training data.
    - validation_dataloader: DataLoader for the validation data.
    - optimizer: Optimizer for training.
    - device: Device to train on (e.g., 'cuda', 'cpu').
    - num_epochs: Number of training epochs.

    Returns:
    - The trained model.
    """
    model.to(device)

    for epoch in range(num_epochs):
        if (
            epoch == specific_epoch_to_defreeze
        ):  # Replace with the epoch number you choose
            defreeze_all_bert_layers(model)

        model.train()
        total_train_loss = 0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
            disable=False,
        )

        for batch in progress_bar:
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_labels = b_labels.to(device)

            model.zero_grad()

            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            progress_bar.set_postfix(
                {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
            )

        avg_train_loss = total_train_loss / len(train_dataloader)
        val_loss, val_accuracy = evaluate_model(model, validation_dataloader, device)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.3f}")
        print(
            f"Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}"
        )

    print("Training complete")
    return model


def evaluate_model(model, dataloader, device):
    """
    Evaluates the BERT model.
    Args:
    - model: The trained BERT model.
    - dataloader: DataLoader for the validation or test data.
    - device: Device for evaluation (e.g., 'cuda', 'cpu').

    Returns:
    - Average loss and accuracy of the model on the given data.
    """
    model.eval()
    model.to(device)
    total_loss, total_accuracy = 0, 0

    for batch in dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)

        with torch.no_grad():
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

        logits = outputs.logits
        loss = outputs.loss
        total_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()
        total_accuracy += flat_accuracy(logits, label_ids)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    return avg_loss, avg_accuracy


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def predict_nationality(model, surnames, tokenizer, inverse_label_dict, device):
    """
    Predicts and decodes the nationalities of given surnames.
    Args:
    - model: The trained BERT model.
    - surnames: List of surnames to predict.
    - tokenizer: The tokenizer used for BERT model.
    - inverse_label_dict: Dictionary for converting numeric labels back to nationalities.
    - device: Device for prediction (e.g., 'cuda', 'cpu').

    Returns:
    - Decoded predictions for each surname.
    """
    model.eval()
    model.to(device)

    predictions = []
    progress_bar = tqdm(
        surnames, desc="Predicting Nationalities", leave=False, disable=False
    )

    with torch.no_grad():
        for surname in progress_bar:
            encoded_surname = tokenizer.encode_plus(
                surname,
                add_special_tokens=True,
                max_length=128,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids = encoded_surname["input_ids"].to(device)
            attention_mask = encoded_surname["attention_mask"].to(device)

            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted_label = torch.argmax(probs, dim=1).cpu().numpy()[0]

            decoded_label = inverse_label_dict[predicted_label]
            predictions.append((surname, decoded_label))

    return predictions


def create_optimizer(model, learning_rate=5e-5, eps=1e-8):
    """
    Creates an optimizer for the BERT model.
    Args:
    - model: The BERT model.
    - learning_rate: Learning rate for the optimizer.
    - eps: Epsilon for the AdamW optimizer.

    Returns:
    - An AdamW optimizer.
    """
    # List of model parameters
    param_optimizer = list(model.named_parameters())

    # We will apply weight decay to all parameters except bias and layer normalization terms
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    # Create the optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps)

    return optimizer


def tokenize_data(df):
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    return tokenizer.batch_encode_plus(
        df["surname"].tolist(),
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )


def create_dataset(encodings, df):
    return TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(df["nationality_index"].values),
    )
