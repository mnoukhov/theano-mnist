from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream


class Dataset(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.dataset = MNIST('train')
        self.images_shape = self.dataset.source_shapes[0]

    def get_data_stream(self):
        data_stream = DataStream(
            self.dataset,
            iteration_scheme=ShuffledScheme(self.dataset.num_examples,
                                            self.batch_size))

        return data_stream
