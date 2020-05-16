import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from utils import *
import os

with open('names_final.txt', 'r') as fnames:
    names, maxlen = get_name_list(fnames, min_length=4, max_length=12, stripped=False)

data_size = len(names)

vocab = sorted(set(''.join([n for n in names])))
char2idx = {u: i + 1 for i, u in enumerate(vocab)}
idx2char = np.array([None] + vocab)

names_in_int = [[char2idx[c] for c in name] for name in names]
examples_per_epoch = len(names_in_int)
padded = pad_sequences(names_in_int, maxlen, truncating='pre')
dataset = tf.data.Dataset.from_tensor_slices(padded)
dataset = dataset.map(lambda x: (x[:-1], x[1:]))

'''
for input_example, target_example in dataset.take(1):
    print(input_example)
    print(target_example)
    i_arr = input_example.numpy()
    o_arr = target_example.numpy()
    i_arr = i_arr[i_arr != 0]
    o_arr = o_arr[o_arr != 0]
    print('Input data: ', repr(''.join(idx2char[i_arr])))
    print('Output data: ', repr(''.join(idx2char[o_arr])))
'''

# Batching and shuffling
BUFFER_SIZE = 1000
BATCH_SIZE = 64
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Build the model
vocab_size = len(vocab)
embedding_dim = 16
rnn_units = 32

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[BATCH_SIZE, None]),
    tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
    tf.keras.layers.LSTM(rnn_units, return_sequences=True),
    tf.keras.layers.Dense(vocab_size, activation='softmax'),
])

# Try the model:
for input_batch, target_batch in dataset.take(1):
    example_batch_predictions = model(input_batch)
    print(example_batch_predictions.shape)


model.summary()

# Try for the first example in the batch
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

# Print the output of the untrained model
sampled_input = input_batch[0].numpy()
sampled_output = sampled_indices
sampled_input = sampled_input[sampled_input != 0]
sampled_output = sampled_output[sampled_output != 0]

print('Input :', repr(''.join(idx2char[sampled_input])))
print('Output :', repr(''.join(idx2char[sampled_output])))


# Train the model
# Attach a loss function, and optimizer

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


for input_batch, target_batch in dataset.take(1):
    example_batch_loss = loss(target_batch, example_batch_predictions)
    print('Prediction shape:', example_batch_predictions.shape)
    print('scalar loss :', example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = 'training_checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

EPOCHS = 5
history = model.fit(dataset,
                    epochs=EPOCHS,
                    # callbacks=[checkpoint_callback],
                    )
