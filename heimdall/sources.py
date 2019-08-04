import numpy as np
import h5py
import z5py


def to_source(data, **kwargs):
    """ Convert the input data to a heimdall.Source.

    Type of the source is inferred from the type of data.
    """

    # we might have a source already -> do nothing
    if isinstance(data, Source):
        return data
    # source from in memory data
    elif isinstance(data, np.ndarray):
        return NumpySource(data, **kwargs)
    # source from hdf5-based source
    elif isinstance(data, h5py.Dataset):
        return HDF5Source(data, **kwargs)
    # TODO implement BDVSource / H5 pyramid source
    elif isinstance(data, h5py.Group):
        raise NotImplementedError
    # sources from n5/zarr based source
    elif isinstance(data, z5py.Dataset):
        return ZarrSource(data, **kwargs)
    # TODO implement n5/zarr pyramid source
    elif isinstance(data, z5py.Group):
        raise NotImplementedError
    else:
        raise ValueError("No source for %s available" % type(data))


class Source:
    layer_types = ('raw', 'labels')
    default_layer_types = {'uint8': 'raw',
                           'uint16': 'raw',
                           'uint32': 'labels',
                           'uint64': 'labels',
                           'int8': 'raw',
                           'int16': 'raw',
                           'int32': 'labels',
                           'int64': 'labels',
                           'float32': 'raw',
                           'float64': 'labels'}

    def to_layer_type(self, layer_type, dtype):
        if layer_type is not None:
            if layer_type not in self.layer_types:
                raise ValueError("Layer type %s is not supported" % layer_type)
            return layer_type
        else:
            return self.default_layer_types[str(dtype)]

    # TODO
    @staticmethod
    def infer_multichannel(shape, layer_type):
        return False

    def __init__(self, data, layer_type=None, name=None, multichannel=None):
        self._data = data
        self._layer_type = self.to_layer_type(layer_type, data.dtype)
        self._name = name
        self._multichannel = self.infer_multichannel(data.shape,
                                                     self.layer_type) if multichannel is None else multichannel

    @property
    def data(self):
        return self._data

    @property
    def ndim(self):
        return self._data.ndim -1 if self.multichannel else self._data.ndim

    # TODO slice away the channel axis if we have a multi-channel source
    @property
    def shape(self):
        return self._data.shape

    @property
    def multichannel(self):
        return self._multichannel

    # TODO add setter
    @property
    def layer_type(self):
        return self._layer_type

    # TODO add setter
    @property
    def name(self):
        return self._name


# TODO pyramid sources ???

class NumpySource(Source):
    pass


class ZarrSource(Source):
    pass


class HDF5Source(Source):
    pass


class BDVSource(Source):
    pass


# source wrappers to resize / apply affines on the fly

class SourceWrapper(Source):
    pass


class ResizeWrapper(SourceWrapper):
    pass


class AffineWrapper(SourceWrapper):
    pass
