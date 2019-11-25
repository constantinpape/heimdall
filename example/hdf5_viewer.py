import h5py
from heimdall import view, to_source


def example(split_channels=False):
    """ Split channel toggles between old (False) and new (True) napari channel behaviour.
    """
    path = '/home/pape/Work/data/ilastik/mulastik/data/data.h5'
    with h5py.File(path) as f:
        raw = f['raw']
        pred = f['prediction']
        pred = to_source(pred, channel_axis=0, split_channels=split_channels)
        view(raw, pred)


if __name__ == '__main__':
    example()
