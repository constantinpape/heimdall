from ..sources import NumpySource, BigDataSource, PyramidSource, TorchSource
from ..source_wrappers import SourceWrapper


# TODO more layer customizations
def add_source(viewer, source, is_pyramid):
    layer_type = source.layer_type

    # the napari behaviour for multi-channel data has changed recently:
    # if we set a channel axis, all channels will be added as separate layers
    # this might not be the intended when using heimdall, so the source has
    # an attribute 'split_channels'. If set to 'False', it will restore the old
    # behaviour, where the channel is just added as extra dimension while still
    # being consistent with heimdall shape checks
    channel_axis = source.channel_axis
    split_channels = source.split_channels
    if channel_axis is not None and not split_channels:
        channel_axis = None

    contrast_limits = None if isinstance(source, (NumpySource, TorchSource))\
        else [source.min_val, source.max_val]

    if layer_type == 'raw':
        viewer.add_image(source.data, name=source.name, scale=source.scale,
                         channel_axis=channel_axis, contrast_limits=contrast_limits,
                         is_pyramid=is_pyramid)
    elif layer_type == 'labels':
        viewer.add_labels(source.data, name=source.name,
                          scale=source.scale, is_pyramid=is_pyramid)


# TODO we can unify this with add_source as well
def add_source_wrapper(viewer, source):
    layer_type = source.layer_type
    channel_axis = source.channel_axis
    # TODO implement multi-channel support for source wrapper
    if channel_axis:
        raise NotImplementedError

    if layer_type == 'raw':
        viewer.add_image(source, name=source.name,
                         channel_axis=channel_axis, scale=source.scale,
                         contrast_limits=[source.min_val, source.max_val],
                         is_pyramid=False)
    elif layer_type == 'labels':
        viewer.add_labels(source, name=source.name, is_pyramid=False,
                          scale=source.scale)


def normalize_shape(source):
    shape = source.shape
    # TODO don't hard-code channel axis to 0
    if source.channel_axis is not None:
        shape = shape[1:]
    return tuple(sh * sc for sh, sc in zip(shape, source.scale))


def check_shapes(source, reference_shape):
    shape = normalize_shape(source)
    if shape != reference_shape:
        raise RuntimeError("Shape of source %s does not match the reference shape %s" % (str(shape),
                                                                                         str(reference_shape)))


# TODO layer specific key-bindings
def add_source_to_viewer(viewer, source, reference_shape):
    check_shapes(source, reference_shape)

    # pyramid needs to be checked before BigDataSource,
    # because the former inherits from the latter
    if isinstance(source, PyramidSource):
        add_source(viewer, source, is_pyramid=True)

    # default in-memory or big-data sources
    elif isinstance(source, (NumpySource, BigDataSource, TorchSource)):
        add_source(viewer, source, is_pyramid=False)

    # source wrapper
    elif isinstance(source, SourceWrapper):
        add_source_wrapper(viewer, source)

    else:
        raise ValueError("Unsupported source %s" % type(source))

    # layer specific key-bindings
