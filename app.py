from transformers import TFBertForSequenceClassification, BertTokenizer
from transformers import glue_convert_examples_to_features
from pathlib import Path
from bidict import bidict
import pandas as pd
import numpy as np

model_str = 'KB/bert-base-swedish-cased'


print("Tokenize...")
tokenizer = BertTokenizer.from_pretrained(model_str)

train_texts = "hej jag vill studera propaganda och upplysning. Upplysningen, och andra böjningsformer, är bra att ha med"
train_encodings = tokenizer.tokenize(train_texts)

print(train_encodings)