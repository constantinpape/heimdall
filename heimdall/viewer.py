import numpy as np
import napari

from .sources import to_source, NumpySource
from .util import add_source_to_viewer, add_keybindings


def view(*sources):
    """ Open viewer for multiple sources.
    """
    viewer_sources = [to_source(source) for source in sources]
    reference_shape = sources[0].shape

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


# TODO
def view_container(path, exclude_names=None, include_names=None):
    """ Display all contents of hdf5 or n5/zarr file.
    """
    raise NotImplementedError
