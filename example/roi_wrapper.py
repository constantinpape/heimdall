import z5py
from functools import partial
from heimdall import view, to_source
from heimdall.source_wrappers import RoiWrapper, roi_wrapper_pyramid_factory


def example():
    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    f = z5py.File(path)

    # roi and factory
    roi_start = [25, 256, 256]
    roi_stop = [100, 768, 768]
    factory = partial(roi_wrapper_pyramid_factory, roi_start=roi_start, roi_stop=roi_stop)

    # raw pyramid source
    raw = f['volumes/raw']
    raw_source = to_source(raw, wrapper_factory=factory, name='raw')

    # label source
    labels = f['volumes/segmentation/groundtruth']
    label_source = RoiWrapper(to_source(labels), roi_start, roi_stop, name='labels')

    view(raw_source, label_source)


if __name__ == '__main__':
    example()
