import z5py
from heimdall import view


# this fails with out-of-range due to
# https://github.com/napari/napari/issues/699
def example():
    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    with z5py.File(path) as f:
        raw = f['volumes/raw/s0']
        raw.n_threads = 8

        seg = f['volumes/segmentation/groundtruth']
        seg.n_threads = 8

    view(raw, seg)


if __name__ == '__main__':
    example()
