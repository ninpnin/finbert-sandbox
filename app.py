from transformers import TFBertForSequenceClassification, BertTokenizer
from transformers import glue_convert_examples_to_features
import tensorflow as tf
import tensorflow_datasets as tfds

model_str = 'TurkuNLP/bert-base-finnish-cased-v1'
model = TFBertForSequenceClassification.from_pretrained(model_str)
tokenizer = BertTokenizer.from_pretrained(model_str)

data = tfds.load('glue/mrpc')

print(data["train"])
train_dataset = glue_convert_examples_to_features(data['train'],
    tokenizer, max_length=128, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)

print(train_dataset)

'''
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)
model.fit(train_dataset, epochs=2, steps_per_epoch=115)
'''
