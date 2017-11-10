import tensorflow as tf
import numpy as np
import random

class hyperparameters():
    seq_size = 100
    hidden_size = 128
    batch_size = 1
    data_file = 'data/shakespeare.txt'
    num_epochs = 1


class input():
    def __init__(self,config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.seq_size = seq_size = config.seq_size
        self.epoch_size = epoch_size = len(data) // (batch_size*seq_size)
        self.input_data, self.targets = _batch_data(data, batch_size, seq_size)

def get_batch(raw_data, batch_size, seq_size):
    rand = [random.randint(0,n_batches) for i in range(batch_size)]
    X = []
    for b in range(n_batches):
        X.append([])
        for r in rand:
            X[b].append([raw_data[i+r*batch_size] for i in range(seq_size)])
    return X


if __name__ == "__main__":
    print("open data file")
    config = hyperparameters()

    with open(config.data_file) as file:
        data = file.read()

    vocab = list(set(data))
    char_to_idx = {char:i for i,char in enumerate(vocab)}
    idx_to_char = {i:char for i,char in enumerate(vocab)}

    W_embed = tf.get_variable("word_embeddings", [len(vocab),config.hidden_size])

    x_input = [char_to_idx[i] for i in data]
    labels = [char_to_idx[i] for i in data[1:]]
    labels.append(char_to_idx[data[0]])
 
    text_size = len(data)

    sequences = []
    # break data into n sequences
    for t_idx in range(text_size // seq_size):
        sequences.append([x_input[t_idx*seq_size:(t_idx+1)*seq_size],
            y_input[t_idx*seq_size:(t_idx+1)*seq_size]])

    n_batches = len(x_input) // (batch_size*seq_size)
    for batch in range(n_batches):
        
        X = tf.nn.embedding_lookup(W_embed, x_batch)

        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(config.hidden_size)
        initial_state = rnn_cell.zero_state([config.batch_size],dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(rnn_cell,X,initial_state=initial_state,dtype=tf.float32)
     
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run([outputs,state]))

 

 # data should have the following structure:
 # text is broken up into n: seq_size list of chars
 # from the n lists, we randomly select batch_size entries
 # this makes 1 batch.
 # The next batch is again randomly selected from the same n lists
 # 1 epoch is when we have "statistically" sampled the full body of text
 # 1 epoch = len(text)//(batch_size*seq_size) batches 