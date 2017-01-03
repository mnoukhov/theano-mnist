import argh
from argh import arg
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop

from dataset import Dataset
from model import Model


@arg('batch_size', type=int)
@arg('learning_rate', type=float)
@arg('num_epochs', type=int)
def train(batch_size, learning_rate, num_epochs):
    dataset = Dataset(batch_size)
    data_stream = dataset.get_data_stream()

    import pdb
    pdb.set_trace()
    model = Model(batch_size)
    loss = model.get_loss()
    params = model.get_all_params()

    algorithm = GradientDescent(cost=loss,
                                parameters=params,
                                step_rule=Scale(learning_rate))

    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  TrainingDataMonitoring([loss]),
                  ProgressBar(),
                  Printing()]

    main_loop = MainLoop(algorithm,
                         data_stream,
                         extensions=extensions)
    main_loop.run()


if __name__ == '__main__':
    argh.dispatch_command(train)
