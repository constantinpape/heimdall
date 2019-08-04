from ..sources import NumpySource, BigDataSource, PyramidSource
from ..source_wrappers import SourceWrapper


def add_numpy_source(viewer, source):
    layer_type = source.layer_type
    multichannel = source.multichannel

    if layer_type == 'raw':
        viewer.add_image(source.data, name=source.name,
                         multichannel=multichannel)
    elif layer_type == 'labels':
        viewer.add_labels(source.data, name=source.name)


def add_big_data_source(viewer, source):
    layer_type = source.layer_type
    multichannel = source.multichannel

    if layer_type == 'raw':
        viewer.add_image(source.data, name=source.name,
                         multichannel=multichannel,
                         clim_range=[source.min_val, source.max_val])
    elif layer_type == 'labels':
        viewer.add_labels(source.data, name=source.name)


def add_pyramid_source(viewer, source):
    layer_type = source.layer_type
    multichannel = source.multichannel

    if layer_type == 'raw':
        pyramid = source.get_pyramid()
        viewer.add_pyramid(pyramid, multichannel=multichannel,
                           clim_range=[source.min_val, source.max_val])
    # TODO does napari support label pyramids already?
    elif layer_type == 'labels':
        raise NotImplementedError


# TODO layer specific key-bindings
# TODO more layer customizations
def add_source_to_viewer(viewer, source, reference_shape):
    if source.shape != reference_shape:
        raise RuntimeError("Shape of source %s does not match the reference shape %s" % (str(source.shape),
                                                                                         str(reference_shape)))
    if isinstance(source, NumpySource):
        add_numpy_source(viewer, source)

    # pyramid needs to be checked before BigDataSource,
    # because the former inherits from the latter
    elif isinstance(source, PyramidSource):
        add_pyramid_source(viewer, source)

    # bigdata-source = source for Zarr or HDF5 dataset
    elif isinstance(source, BigDataSource):
        add_big_data_source(viewer, source)

    # TODO source wrappers are a bit more complicated...
    elif isinstance(source, SourceWrapper):
        pass

    else:
        raise ValueError("Unsupported source %s" % type(source))

    # layer specific key-bindings
