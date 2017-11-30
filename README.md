# char_rnn_tf
Character based RNN to write text.

Requires python3, tensorflow.

Run using:
$ python3 my_rnn.py

To turn on sampling, edit the lines in my_rnn.py:
train = True => train = False
restart = True => restart = False

The model must be trained first so that there is a checkpoint to load.

TODO:
 - Add command line flags for sampling, temperature etc.
 - package model for integration into AltaML website.
 
