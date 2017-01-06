# theano-mnist
A great basic starter to using theano the right way (or my way at least)

Uses lasagne (model definition), fuel (data processing), and blocks (training loop) to create a clean and extensible MNIST classifier using LeNet, also can serve as a great template for any project

## why
In building neural networks, I like to separate things into model, data, and training loop. By using these three libraries, this becomes simple and easy. Blocks main loop and extensions become especially useful for extending the code to your specific needs.

### why lasagne and not blocks.bricks
I find the structuring of networks in lasagne to be more intuitive and more similar to tensorflow (and tf-slim), making it easier to implement some tensorflow models in theano

## requirements
install required libraries
* install [latest theano](http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions)
* install [latest blocks (and fuel)](http://blocks.readthedocs.io/en/latest/setup.html)
* install [latest lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html#bleeding-edge-version)
* install argh: `pip install argh`

prepare mnist data
* follow [this guide](http://fuel.readthedocs.io/en/latest/built_in_datasets.html) to download and create `mnist.hdf5`

## running 
`python train.py BATCH_SIZE LEARNING_RATE NUM_EPOCHS` 

## TODO
- add validation/testing
- add visualization (blocks-extras' live plotting or mimir logging into jupyter)
- add model saving/loading (add my own lasagne saver/loader)
