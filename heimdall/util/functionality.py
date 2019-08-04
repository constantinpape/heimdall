from ..sources import NumpySource, BigDataSource


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


# TODO layer specific key-bindings
# TODO more layer customizations
# TODO support more sources
def add_source_to_viewer(viewer, source, reference_shape):
    if source.shape != reference_shape:
        raise RuntimeError("Shape of source %s does not match the reference shape %s" % (str(source.shape),
                                                                                         str(reference_shape)))
    if isinstance(source, NumpySource):
        add_numpy_source(viewer, source)

    # bigdata-source = source for Zarr or HDF5 dataset
    elif isinstance(source, BigDataSource):
        add_big_data_source(viewer, source)

    # other sources
    else:
        raise NotImplementedError

    # layer specific key-bindings
