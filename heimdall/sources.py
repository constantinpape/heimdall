import numpy as np
import h5py
from z5py.dataset import Dataset as Z5Dataset
from z5py.group import Group as Z5Group


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
    elif isinstance(data, Z5Dataset):
        return ZarrSource(data, **kwargs)
    # TODO implement n5/zarr pyramid source
    elif isinstance(data, Z5Group):
        raise NotImplementedError
    else:
        raise ValueError("No source for %s available" % type(data))


class Source:
    """ Base class for data sources.
    """

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
        return self._data.ndim - 1 if self.multichannel else self._data.ndim

    @property
    def shape(self):
        return self._data.shape[1:] if self.multichannel else self._data.shape

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


class NumpySource(Source):
    """ Source from numpy array.
    """
    def __init__(self, data, **kwargs):
        if not isinstance(data, np.ndarray):
            raise ValueError("NumpySource expecsts a numpy array, not %s" % type(data))
        super().__init__(data, **kwargs)


class BigDataSource(Source):
    """ Base class for hdf5 or n5/zarr source.
    """

    @staticmethod
    def infer_min(dtype):
        if np.dtype(dtype) in (np.dtype('float32'), np.dtype('float64')):
            return 0.
        elif np.dtype(dtype) in (np.dtype('uint8'), np.dtype('uint16'),
                                 np.dtype('int8'), np.dtype('int16')):
            return np.iinfo(dtype).min
        # min-max for label dtypes is not relevant
        else:
            return None

    @staticmethod
    def infer_max(dtype):
        if np.dtype(dtype) in (np.dtype('float32'), np.dtype('float64')):
            return 1.
        elif np.dtype(dtype) in (np.dtype('uint8'), np.dtype('uint16'),
                                 np.dtype('int8'), np.dtype('int16')):
            return np.iinfo(dtype).max
        # min-max for label dtypes is not relevant
        else:
            return None

    def __init__(self, data, min_val=None, max_val=None, **kwargs):
        super().__init__(data, **kwargs)
        self._min_val = self.infer_min(data.dtype) if min_val is None else min_val
        self._max_val = self.infer_max(data.dtype) if max_val is None else max_val

    # TODO setter
    @property
    def min_val(self):
        return self._min_val

    # TODO setter
    @property
    def max_val(self):
        return self._max_val


class ZarrSource(BigDataSource):
    """ Source from zarr dataset.
    """
    def __init__(self, data, **kwargs):
        if not isinstance(data, Z5Dataset):
            raise ValueError("ZarrSource expecsts a z5 dataset, not %s" % type(data))
        super().__init__(data, **kwargs)


class HDF5Source(BigDataSource):
    """ Source from hdf5 dataset.
    """
    def __init__(self, data, **kwargs):
        if not isinstance(data, h5py.Dataset):
            raise ValueError("HDF5Source expects a h5py dataset, not %s" % type(data))
        super().__init__(data, **kwargs)


# TODO implement pyramid sources
# class BDVSource(Source):
#     pass


# source wrappers:
# TODO
# - resize on the fly
# - apply affines on the fly
# - data caching

class SourceWrapper(Source):
    pass


class ResizeWrapper(SourceWrapper):
    pass


class AffineWrapper(SourceWrapper):
    pass


class CacheWrapper(SourceWrapper):
    pass
