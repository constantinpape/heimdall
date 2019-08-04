import z5py
from heimdall import view_arrays


def example_2d(z=0):
    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    with z5py.File(path) as f:
        ds = f['volumes/raw/s0']
        ds.n_threads = 8
        raw = ds[z]

        ds = f['volumes/segmentation/groundtruth']
        ds.n_threads = 8
        seg = ds[z]

    view_arrays([raw, seg])


def example_3d():
    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    with z5py.File(path) as f:
        ds = f['volumes/raw/s0']
        ds.n_threads = 8
        raw = ds[:]

        ds = f['volumes/segmentation/groundtruth']
        ds.n_threads = 8
        seg = ds[:]

    view_arrays([raw, seg])


if __name__ == '__main__':
    # example_2d()
    example_3d()
