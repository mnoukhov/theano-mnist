# theano-mnist
A great basic starter to using theano the right way (or how to use as many libraries as possible)

Uses fuel (data processing), blocks (main loop), and lasagne (model definition) to create a clean and extensible MNIST classifier using LeNet


## install requirements
install [latest theano](http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions)
install [latest blocks (and fuel)](http://blocks.readthedocs.io/en/latest/setup.html)
install [latest lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html#bleeding-edge-version)
install argh: `pip install argh`

## prep data
follow [this guide](http://fuel.readthedocs.io/en/latest/built_in_datasets.html) to download and create `mnist.hdf5`
