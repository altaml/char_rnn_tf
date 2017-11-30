import tensorflow as tf
import numpy as np
import random
import pickle

'''
  
'''
class hyperparameters():
    seq_size = 50
    hidden_size = 128
    batch_size = 50
    data_file = 'data/shakespeare.txt'
    vocab_file = 'data/vocab.pkl'
    num_epochs = 50
    learn_rate = 2e-3
    decay_rate = 0.99
    keep_prob = 0.9
    num_layers = 2
    len_vocab = 0
    ckpt_dir = './models/'
    n_batches = 0
    temp = 0.025
    train = False
    restart = False

def get_batch(raw_data, batch_size):
    rand = [random.randint(0,len(raw_data)-1) for i in range(batch_size)]
    X = []
    Y = []
    for r in rand:
        X.append(raw_data[r][0])
        Y.append(raw_data[r][1])
    return X,Y

def train_valid(raw_data, valid_size):
    rand = [random.randint(0,len(raw_data)-1) for i in range(valid_size)]
    train = []
    valid = []
    for i in range(len(raw_data)):
        if i in rand:
            valid.append(raw_data[i])
        else:
            train.append(raw_data[i])
    return np.asarray(train),np.asarray(valid)

def get_data(config):
    with open(config.data_file) as file:
        data = file.read()

    vocab = list(set(data))

    config.len_vocab = len(vocab)
    char_to_idx = {char:i for i,char in enumerate(vocab)}
    idx_to_char = {i:char for i,char in enumerate(vocab)}

    # save vocab for later sampling only if we dont have one we are using
    if config.restart == True:
        with open(config.vocab_file, 'wb') as output:
            pickle.dump((vocab,char_to_idx,idx_to_char),output)
    else:
        with open(config.vocab_file, 'rb') as input_pkl:
            vocab,char_to_idx,idx_to_char = pickle.load(input_pkl)

    x_input = [char_to_idx[i] for i in data]
    y_input = [char_to_idx[i] for i in data[1:]]
    y_input.append(char_to_idx[data[0]])
 
    text_size = len(data)

    # break data into n sequences
    sequences = []
    for t_idx in range(text_size // config.seq_size):
        sequences.append([x_input[t_idx*config.seq_size:(t_idx+1)*config.seq_size],
            y_input[t_idx*config.seq_size:(t_idx+1)*config.seq_size]])

    train_set,valid_set = train_valid(sequences,int(0.05*len(sequences)))
    config.n_batches = len(x_input) // (config.batch_size*config.seq_size)

    return train_set,valid_set,char_to_idx,idx_to_char,vocab

class model(object):
    def __init__(self,config,is_training=True):
        self.x = tf.placeholder(tf.int32,[config.batch_size, config.seq_size],name="x")
        self.y_ = tf.placeholder(tf.int32,[config.batch_size, config.seq_size],name="y_")
        self.kp = tf.placeholder(tf.float32,name="kp")
        self.initial_state = tf.placeholder(tf.float32, 
            [config.num_layers, 2, config.batch_size, config.hidden_size])

        self.rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(self.initial_state[idx][0], self.initial_state[idx][1])
            for idx in range(config.num_layers)])


        with tf.variable_scope("embeddings",reuse=tf.AUTO_REUSE):
            W_embed = tf.get_variable("word_embeddings", [config.len_vocab,config.hidden_size])
        X = tf.nn.embedding_lookup(W_embed, self.x)
        tf.summary.histogram("embeddings",W_embed)

        if is_training:
            X = tf.nn.dropout(X, self.kp)

        cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size, state_is_tuple=True)

        if is_training:
            cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.kp)

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([cell]*config.num_layers,state_is_tuple=True)

        with tf.variable_scope("rnn",reuse=tf.AUTO_REUSE):
            outputs, state = tf.nn.dynamic_rnn(
                cell=multi_rnn_cell,
                inputs=X,
                dtype=tf.float32,
                initial_state=self.rnn_tuple_state,
                time_major=False)

        self.final_state = state
        outputs = tf.reshape(outputs, [-1, config.hidden_size])

        with tf.variable_scope("softmax",reuse=tf.AUTO_REUSE): 
            self.W = tf.get_variable("W_softmax",[config.hidden_size,config.len_vocab])
            self.b = tf.get_variable("b_softmax",[config.len_vocab])
        tf.summary.histogram("W_softmax",self.W)
        tf.summary.histogram("b_softmax",self.b)

        self.logits = tf.matmul(outputs,self.W) + self.b

        self.probabilities = tf.nn.softmax(self.logits)

        self.y_list = tf.reshape(self.y_,[-1])
        self.y_list = tf.one_hot(indices=self.y_list,depth=len(vocab))
        
        correct_prediction = tf.equal(tf.argmax(self.y_list,1), tf.argmax(self.logits,1))
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',self.accuracy)

        with tf.name_scope('loss'):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_list,logits=self.logits))
        tf.summary.scalar('loss',self.cross_entropy)

        self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver(tf.global_variables())

        if not is_training:
            return

        trainable_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cross_entropy, trainable_vars),
            5.0)
        optimizer = tf.train.AdamOptimizer(config.learn_rate)
        self.train_optimizer = optimizer.apply_gradients(
            zip(grads,trainable_vars))


    def step(self, sess, batch_x, batch_y, init_state=None):
        if init_state == None:
            init_state = np.zeros((config.num_layers,2,config.batch_size,config.hidden_size))

        input_feed = {self.x: batch_x,
                      self.y_: batch_y,
                      self.kp: config.keep_prob,        
                      self.initial_state: init_state}

        output_feed = [self.train_optimizer,
                       self.cross_entropy,
                       self.accuracy,
                       self.final_state,
                       self.merged,
                       self.W]

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]

    def valid(self, sess, batch_x, batch_y, config, init_state=None):
        if init_state == None:
            init_state = np.zeros((config.num_layers,2,config.batch_size,config.hidden_size))

        input_feed = {self.x: batch_x,
                      self.y_: batch_y,
                      self.kp: 1.0,
                      self.initial_state: init_state}

        output_feed = [self.cross_entropy,
                       self.accuracy,
                       self.probabilities]

        outputs = sess.run(output_feed, input_feed)

        return outputs[0], outputs[1], outputs[2]

    def sample(self, sess, seed, config, sampling_type=1):
        # Initialize model with seed
        state = np.zeros((config.num_layers,2,config.batch_size,config.hidden_size))
        # print(seed,end='')
        for char_num, char in enumerate(seed[:-1]):
            word = np.array(char_to_idx[char]).reshape(1,1)
            input_feed = {self.x: word, self.kp: 1.0, self.initial_state: state}

            state,tuple_state,W = sess.run([self.final_state,self.rnn_tuple_state,self.W], feed_dict=input_feed)
            if(char_num == 0):
                print("Initial W is:\n",W)

        prev_char = seed[-1]
        for word_num in range(0,1000):
            # print(idx_to_char[char_to_idx[prev_char]])
            word = np.array(char_to_idx[prev_char]).reshape(1,1)
            feed_dict = {self.x: word, self.kp: 1.0, self.initial_state: state}

            probs, state, tuple_state,W= sess.run([self.probabilities,self.final_state,self.rnn_tuple_state,self.W],
                feed_dict=feed_dict)

            next_char_dist = probs[0]

            next_char_dist /= config.temp
            next_char_dist = np.exp(next_char_dist)
            next_char_dist /= sum(next_char_dist)

            if sampling_type != 0:
                choice_index = np.argmax(next_char_dist)
            else:
                choice_index = -1
                point = random.random()
                weight = 0.0
                for index in range(0,config.len_vocab):
                    weight += next_char_dist[index]
                    if weight >= point:
                        choice_index = index
                        break

            seed = idx_to_char[choice_index]
            prev_char = seed
            print(seed,end='')
        print('\n')
        return W

def create_model(sess, config, is_training):

    char_model = model(config,is_training)

    ckpt = tf.train.get_checkpoint_state(config.ckpt_dir)

    if config.restart:
        ckpt = False
    if ckpt and tf.train.checkpoint_exists(tf.train.latest_checkpoint(config.ckpt_dir)):
        print("Restoring model:",tf.train.latest_checkpoint(config.ckpt_dir))
        char_model.saver.restore(sess, tf.train.latest_checkpoint(config.ckpt_dir))
    else:
        sess.run(tf.global_variables_initializer())
    return char_model

def train(config,train_set,valid_set):
    with tf.Session() as sess:
        # training model
        model = create_model(sess, config, True)
        state = None

        # validation model
        valid_config = hyperparameters()
        valid_config.batch_size = len(valid_set)
        valid_config.len_vocab = config.len_vocab
        valid_model = create_model(sess, valid_config, False)

        # sample model
        sample_config = hyperparameters()
        sample_config.batch_size = 1
        sample_config.seq_size = 1
        sample_config.len_vocab = config.len_vocab
        sample_model = create_model(sess, sample_config, False)
        
        train_writer = tf.summary.FileWriter('logs/train',sess.graph)

        for batch in range(config.n_batches*config.num_epochs+1):
            epoch = batch / config.n_batches
            if epoch > 10 and batch % config.n_batches == 0:
                config.learn_rate = config.learn_rate*config.decay_rate
            print("\rRunning Batch: %s" % batch,end='')
            batch_x, batch_y = get_batch(train_set,config.batch_size)
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)

            _,loss,acc,state,train_summary, W = model.step(sess, batch_x, batch_y, state)
            train_writer.add_summary(train_summary,batch)

            if batch % 100 == 0 or epoch == config.num_epochs:
                print("\nEpoch %.1f:      \n\tTraining loss %.4f, Training Accuracy %.4f" % (epoch,loss,acc))
                model.saver.save(sess, config.ckpt_dir+'model', global_step=batch,write_meta_graph=True)

                loss,acc,probs = valid_model.valid(sess,valid_set[:,0],valid_set[:,1],valid_config)
                print("\tValidate loss %.4f, Validate Accuracy %.4f" % (loss,acc))
                print("**********************************************")
                W = sample_model.sample(sess, '\n',sample_config, 0)
                print("**********************************************")
            if epoch == config.num_epochs:
                print("Final trained W is:\n", W)

 

def sample(config):
    with tf.Session() as sess:
        model = create_model(sess, config, False)
        config.batch_size = 1
        config.seq_size = 1
        config.len_vocab = 65
        model.sample(sess, '\n', config,0)
        
 

if __name__ == "__main__":
    print("open data file")
    config = hyperparameters()

    if config.train:
        train_set,valid_set,char_to_idx,idx_to_char,vocab = get_data(config)
        train(config,train_set,valid_set)
    else:
        with tf.device('/cpu:0'):

            with open(config.vocab_file, 'rb') as pkl_file:
                vocab,char_to_idx,idx_to_char = pickle.load(pkl_file)
            config.batch_size = 1
            config.seq_size = 1
            config.len_vocab = len(vocab)
            sample(config)