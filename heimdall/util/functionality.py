from ..sources import NumpySource, BigDataSource, PyramidSource
from ..source_wrappers import SourceWrapper


# TODO more layer customizations
def add_source(viewer, source):
    layer_type = source.layer_type
    multichannel = source.multichannel

    clim_range = None if isinstance(source, NumpySource)\
        else [source.min_val, source.max_val]

    if layer_type == 'raw':
        viewer.add_image(source.data, name=source.name,
                         multichannel=multichannel, clim_range=clim_range)
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


def add_source_wrapper(viewer, source):
    layer_type = source.layer_type
    multichannel = source.multichannel
    # TODO implement multi-channel support for source wrapper
    if multichannel:
        raise NotImplementedError

    if layer_type == 'raw':
        viewer.add_image(source, name=source.name,
                         multichannel=multichannel,
                         clim_range=[source.min_val, source.max_val])
    elif layer_type == 'labels':
        viewer.add_labels(source.data, name=source.name)


# TODO layer specific key-bindings
def add_source_to_viewer(viewer, source, reference_shape):
    if source.shape != reference_shape:
        raise RuntimeError("Shape of source %s does not match the reference shape %s" % (str(source.shape),
                                                                                         str(reference_shape)))
    # default in-memory or big-data sources
    if isinstance(source, (NumpySource, BigDataSource)):
        add_source(viewer, source)

    # pyramid needs to be checked before BigDataSource,
    # because the former inherits from the latter
    elif isinstance(source, PyramidSource):
        add_pyramid_source(viewer, source)

    # source wrapper
    elif isinstance(source, SourceWrapper):
        add_source_wrapper(viewer, source)

    else:
        raise ValueError("Unsupported source %s" % type(source))

    # layer specific key-bindings
