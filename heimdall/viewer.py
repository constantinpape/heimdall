import os
import numpy as np
import napari
import elf.io

from .sources import Source, NumpySource, BigDataSource, PyramidSource
from .sources import infer_pyramid_format
from .source_wrappers import SourceWrapper
from .util import add_source_to_viewer, add_keybindings, normalize_shape


def to_source(data, **kwargs):
    """ Convert the input data to a heimdall.Source.

    Type of the source is inferred from the type of data.
    """

    # we might have a source or source wrapper already -> do nothing
    if isinstance(data, (Source, SourceWrapper)):
        return data
    # source from in memory data
    elif isinstance(data, np.ndarray):
        return NumpySource(data, **kwargs)
    # source from dataset
    elif elf.io.is_dataset(data):
        return BigDataSource(data, **kwargs)
    # sources from n5/zarr or hdf5 (bdv) image pyramid
    elif elf.io.is_group(data):
        pyramid_format = infer_pyramid_format(data)
        if pyramid_format is None:
            raise ValueError("Group does not have one of the supported pyramid formats")
        return PyramidSource(data, pyramid_format=pyramid_format, **kwargs)
    else:
        raise ValueError("No source for %s available" % type(data))


def view(*sources, return_viewer=False):
    """ Open viewer for multiple sources.

    Arguments:
        sources [args]: sources to view, must be instances of heimdall.sources.Source
        return_viewer [bool]: whether to return the napari viewer object.
            If True, this function must be wrapped into napari.gui_qt like so:
            ```
            with napari.gui_qt():
                viewer = view(*sources, return_viewer=True)
                ...
            ```
            (default: False)
    """
    viewer_sources = [to_source(source) for source in sources]
    reference_shape = normalize_shape(viewer_sources[0])

    if return_viewer:
        viewer = napari.Viewer(title='Heimdall')
        for source in viewer_sources:
            add_source_to_viewer(viewer, source, reference_shape)
        add_keybindings(viewer)
        return viewer
    else:
        with napari.gui_qt():
            viewer = napari.Viewer(title='Heimdall')
            for source in viewer_sources:
                add_source_to_viewer(viewer, source, reference_shape)
            add_keybindings(viewer)


def view_arrays(data, labels=None, layer_types=None):
    """ Simple viewer for in-memory data.

    This is a legacy function compatible with
    https://github.com/constantinpape/cremi_tools/blob/master/cremi_tools/viewer/volumina/volumina.py#L5.
    Useful to quickly view in-memory data.

    Arguments:
        data [list[np.ndarray]]: list of arrays to display
        labels [list[str]]: list of layer names (default: None)
        layer_types [list[str]]: list of layer types, by default this is inferred from dtypes (default: None)
    """

    # dictionaries to translate legacy arguments / dtypes to layer types
    name_to_layer = {'Grayscale': 'raw',
                     'RandomColors': 'labels',
                     # legacy layer names to support cremi_tools.viewer.volumina.view syntax
                     'Red': 'raw',
                     'Green': 'raw',
                     'Blue': 'raw'}

    # validate the arguments
    assert all(isinstance(d, np.ndarray) for d in data)
    if labels is not None:
        assert len(labels) == len(data)
        assert all(isinstance(label, str) for label in labels)
    if layer_types is not None:
        assert len(layer_types) == len(data)
        assert all(lt in name_to_layer for lt in layer_types)

    # build the sources
    sources = []
    for i, d in enumerate(data):
        name = 'layer_%i' % i if labels is None else labels[i]
        layer_type = None if layer_types is None else name_to_layer[layer_types[i]]
        source = NumpySource(d, name=name, layer_type=layer_type)
        sources.append(source)

    # start the viewer
    view(*sources)


def view_container(path, ndim=3,
                   exclude_names=None, include_names=None,
                   load_into_memory=False, n_threads=1):
    """ Display contents of hdf5, n5/zarr or knossos file.

    Arguments:
        path [str]: path to the file
        ndim [int]: expected number of dimensions (default: 3)
        exclude_names [listlike]: will not load these names.
            Not compatible with include_names (default: None)
        include_names [listlike]: will ONLY load these names.
            Not compatible with exclude_names (default: None).
        load_into_memory [bool]: whether to load data into memory (default: False).
        n_threads [n_threads]: number of threads used by z5py (default: 1)
    """
    assert not ((exclude_names is not None) and (include_names is not None))
    with elf.io.open_file(path, mode='r') as f:
        if elf.io.is_knossos(f):
            sources = [to_source(f, n_threads=n_threads)]
        else:
            sources = load_sources_from_file(f, reference_ndim=ndim,
                                             exclude_names=exclude_names,
                                             include_names=include_names,
                                             load_into_memory=load_into_memory,
                                             n_threads=n_threads)
        view(*sources)


def is_pyramid_ds(name, node):

    def isint(x):
        try:
            int(x)
            return True
        except Exception:
            return False

    name = os.path.split(name)[-1]
    if elf.io.is_h5py(node) and name == 'cells':
        return True
    if elf.io.is_z5py(node) and name.startswith('s') and isint(name[1:]):
        return True
    return False


def load_sources_from_file(f, reference_ndim,
                           exclude_names=None, include_names=None,
                           load_into_memory=False, n_threads=1):
    sources = []

    def visitor(name, node):

        if elf.io.is_dataset(node):
            # check if this is in the exclude_names
            if exclude_names and name in exclude_names:
                return

            # check if this is in the include_names
            if include_names and name not in include_names:
                return

            # TODO it would be better to not visit the children in a pyramid group,
            # but I am not quite sure how to do this with the h5py(like) visitor pattern
            # check if this is a dataset of a pyramid group
            # and don't load if it is
            if is_pyramid_ds(name, node):
                return

            # set the number of threads (only has an effect for z5py datasets)
            # and load into memory if specified
            node.n_threads = n_threads
            if load_into_memory:
                node = node[:]

            # check the number of dimensions against the reference dimensionality
            ndim = node.ndim
            channel_axis = None
            if ndim == reference_ndim + 1:
                # channel axis is hard-coded to 0 for now
                channel_axis = 0
            elif ndim != reference_ndim:
                return

            print("Appending dataset source")
            sources.append(to_source(node, channel_axis=channel_axis, name=name))

        elif elf.io.is_group(node):
            pyramid_format = infer_pyramid_format(node)
            if pyramid_format is not None:
                # TODO infer the channel axis
                sources.append(to_source(node, name=name, pyramid_format=pyramid_format))

    f.visititems(visitor)
    return sources
