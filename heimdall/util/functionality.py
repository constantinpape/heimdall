from ..sources import NumpySource, ZarrSource


def add_numpy_source(viewer, source):
    layer_type = source.layer_type
    multichannel = source.multichannel

    if layer_type == 'raw':
        viewer.add_image(source.data, name=source.name,
                         multichannel=multichannel)
    elif layer_type == 'labels':
        viewer.add_labels(source.data, name=source.name)


def add_zarr_source(viewer, source):
    pass


# TODO support other sources
# TODO layer specific key-bindings
# TODO more layer customizations
def add_source_to_viewer(viewer, source, reference_shape):
    if source.shape != reference_shape:
        raise RuntimeError("Shape of source %s does not match the reference shape" % (str(source.shape),
                                                                                      str(reference_shape)))
    if isinstance(source, NumpySource):
        add_numpy_source(viewer, source)

    elif isinstance(source, ZarrSource):
        add_zarr_source(viewer, source)

    # other sources
    else:
        raise NotImplementedError

    # layer specific key-bindings
