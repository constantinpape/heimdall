import z5py
from heimdall import view


def example():
    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    with z5py.File(path) as f:
        raw = f['volumes/raw']
    view(raw)


if __name__ == '__main__':
    example()
