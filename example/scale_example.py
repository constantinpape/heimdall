import z5py
from heimdall import view, to_source


def example():
    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    with z5py.File(path) as f:
        raw = to_source(f['volumes/raw/s1'], scale=(1, 2, 2))

        seg = f['volumes/segmentation/groundtruth']
        seg.n_threads = 8

    view(raw, seg)


if __name__ == '__main__':
    example()
