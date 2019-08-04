import numpy as np
import napari

from .sources import to_source, NumpySource
from .util import add_source_to_viewer, add_keybindings
from .util import open_file, load_sources_from_file


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
    reference_shape = viewer_sources[0].shape

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


def simple_view(data, labels=None, layer_types=None):
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
                     # TODO
                     'Red': None,
                     'Green': None,
                     'Blue': None}

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
    """ Display contents of hdf5 or n5/zarr file.

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
    with open_file(path, mode='r') as f:
        sources = load_sources_from_file(f, reference_ndim=ndim,
                                         exclude_names=exclude_names,
                                         include_names=include_names,
                                         load_into_memory=load_into_memory,
                                         n_threads=n_threads)
        view(*sources)
