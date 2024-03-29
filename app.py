from transformers import TFBertForSequenceClassification, BertTokenizer
from transformers import glue_convert_examples_to_features
from pathlib import Path
from bidict import bidict
import pandas as pd
import numpy as np

model_str = 'KB/bert-base-swedish-cased'

def read_data(filepath, head=1000):
    # Clean up data
    df = pd.read_csv(filepath)
    df = df[df["sender"].notnull()]
    df["text"] = df["text"].str.replace('[^a-zäöA-ZÄÖ ]', '')
    df = df[df["text"].notnull()]
    df = df[~df["sender"].str.contains("bot")]
    df = df[~df["sender"].str.contains("Bot")]
    df = df[df["sender"] != "Pizza"]
    df = df[df["sender"] != "Spotify"]
    df = df[df["sender"] != "Modet"]
    df = df[df["sender"] != "Alko"]
    df = df[df["sender"] != "OngoPongo"]

    return df.head(head)

def train_test_datasets(df):
    # Build vocabulary
    senders = sorted(list(set(df["sender"])))
    label_vocab = bidict({sender: ix for ix, sender in enumerate(senders)})
    df["id"] = df["sender"].apply(lambda x: label_vocab[x])

    # Split into train, validation and test sets
    df["split"] = np.random.choice(["train", "test", "valid"], len(df), p=[0.7, 0.15, 0.15])
    train_df = df[df["split"] == "train"]
    train_texts = list(train_df["text"])
    train_labels = list(train_df["id"])
    train = (train_texts, train_labels)

    valid_df = df[df["split"] == "valid"]
    valid_texts = list(valid_df["text"])
    valid_labels = list(valid_df["id"])
    valid = (valid_texts, valid_labels)

    test_df = df[df["split"] == "test"]
    test_texts = list(test_df["text"])
    test_labels = list(test_df["id"])
    test = (test_texts, test_labels)

    return train, valid, test, label_vocab

print("Read data in ...")
data = read_data('data/rr.csv', head=5000)
train, valid, test, label_vocab = train_test_datasets(data)
print(train[0][:10])
print(train[1][:10])
print(label_vocab)

train_texts = train[0]
valid_texts = valid[0]
test_texts = test[0]

print("Tokenize...")
tokenizer = BertTokenizer.from_pretrained(model_str)
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

print("Generate datasets...")
import tensorflow as tf
train_labels = train[1]
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))
valid_labels = valid[1]
valid_dataset = tf.data.Dataset.from_tensor_slices((
    dict(valid_encodings),
    valid_labels
))
test_labels = test[1]
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_labels
))

print("Start training...")
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments

training_args = TFTrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=30,
)

with training_args.strategy.scope():
    model = TFBertForSequenceClassification.from_pretrained(
        model_str, num_labels=len(label_vocab),
        output_attentions=False, output_hidden_states=False)

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)
