import h5py
from heimdall import view, to_source


def example():
    path = '/home/pape/Work/data/ilastik/mulastik/data/data.h5'
    with h5py.File(path) as f:
        raw = f['raw']
        pred = f['prediction']
        pred = to_source(pred, multichannel=True)
        view(raw, pred)


if __name__ == '__main__':
    example()
