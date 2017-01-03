from lasagne import layers as L
from lasagne.objectives import categorical_crossentropy
from lasagne.nonlinearities import softmax
from theano import tensor as T


class Model(object):
    def __init__(self, batch_size):
        self.images = T.tensor4('features')
        self.labels = T.ivector('targets')
        self.network = lenet(batch_size, 28, 28, 3)
        self.predictions = L.get_output(self.network, inputs=self.images)

    def get_loss(self):
        return categorical_crossentropy(self.predictions, self.labels)


def lenet(batch_size, width, height, channels):
        l_input = L.InputLayer(shape=(batch_size,
                                      width,
                                      height,
                                      channels),
                               name='input')
        l_conv_1 = L.Conv2DLayer(l_input,
                                 num_filters=20,
                                 filter_size=(5,5),
                                 name='conv1')
        l_pool_1 = L.MaxPool2DLayer(l_conv_1,
                                    pool_size=(2,2),
                                    name='pool1')
        l_conv_2 = L.Conv2DLayer(l_pool_1,
                                 num_filters=50,
                                 filter_size=(5,5),
                                 name='conv2')
        l_pool_2 = L.MaxPool2DLayer(l_conv_2,
                                    pool_size=(2,2),
                                    name='pool2')
        l_flatten = L.FlattenLayer(l_pool_2,
                                   name='flatten')
        l_dense_1 = L.DenseLayer(l_flatten,
                                 num_units=500,
                                 name='dense1')
        l_dense_2 = L.DenseLayer(l_dense_1,
                                 num_units=10,
                                 name='dense2',
                                 nonlinearity=softmax)
        return l_dense_2
