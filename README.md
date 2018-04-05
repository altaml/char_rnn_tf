# Character based RNN
This is one of my first attempts at an RNN. 
It was inspired by Andrej Kapathy's [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

### Sources
I used this project to teach myself how to use Tensorflow and LSTM's in general.
Because of this, I went through a lot of online tutorials which, in turn inspired parts of my code.
Below is an incomplete list of the sources that I used in developing this project.
 - [The Nerual Perspective](https://theneuralperspective.com/2016/10/04/05-recurrent-neural-networks-rnn-part-1-basic-rnn-char-rnn/)
 - [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
 - [TensorFlow RNN Tutorial](https://www.tensorflow.org/tutorials/recurrent)

### Requirements and Running
Requires python3, tensorflow. It will run better if you have a gpu and use tensorflow-gpu.

To train the network, edit the main file `my_rnn.py` to set:

`train = True`

`restart = True`

This tells the network not to load a checkpoint and re-generate the vocabulary file.

Run using:
$ python3 my_rnn.py

The model must be trained first so that there is a checkpoint to load.

TODO:
 - Add command line flags for sampling, temperature etc.
 - More code comments
 - Test on CPU only machine
 
