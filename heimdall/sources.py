import numpy as np
import elf.io


def check_consecutive(scales):
    scales = sorted(scales)
    is_consecutive = (scales[0] == 0) and (scales == list(range(scales[0], scales[-1] + 1)))
    return is_consecutive


def infer_pyramid_format(group):
    """ Infer pyramid format from group object.

    Checks for bdv multiscale format (hdf5) or format used by paintera (n5).
    Returns None if no format could be inferred.
    """
    keys = list(group.keys())

    # check for n5 multiscale format
    if elf.io.is_z5py(group) and elf.io.is_group(group):
        try:
            scales = [int(scale[1:]) for scale in keys]
            is_consecutive = check_consecutive(scales)
            if is_consecutive:
                return 'n5'
            else:
                return None
        except Exception:
            return None

    # check for bdv multiscale format
    elif elf.io.is_h5py(group) and elf.io.is_group(group):
        try:
            scales = [int(scale) for scale in keys]
            is_consecutive = check_consecutive(scales)
            if is_consecutive:
                return 'bdv'
            else:
                return None
        except Exception:
            return None

    return None


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

    def __init__(self, data, layer_type=None, name=None, multichannel=False):
        self._data = data
        self._layer_type = self.to_layer_type(layer_type, data.dtype)
        self._name = name
        self._multichannel = multichannel

    @property
    def data(self):
        return self._data

    @property
    def ndim(self):
        return self._data.ndim - 1 if self.multichannel else self._data.ndim

    # maybe we should also support channel dim last, but I don't know how napari deals with this
    @property
    def shape(self):
        return self._data.shape[1:] if self.multichannel else self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def multichannel(self):
        return self._multichannel

    @property
    def layer_type(self):
        return self._layer_type

    @layer_type.setter
    def layer_type(self, layer_type):
        if layer_type not in self.layer_types:
            raise ValueError("Invalid layer type %s, only %s supported" % (layer_type,
                                                                           str(self.layer_types)))
        self._layer_type = layer_type

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name


class NumpySource(Source):
    """ Source from numpy array.
    """
    def __init__(self, data, **kwargs):
        if not isinstance(data, np.ndarray):
            raise ValueError("NumpySource expecsts a numpy array, not %s" % type(data))
        super().__init__(data, **kwargs)


class BigDataSource(Source):
    """ Source wrapping an out-of-core dataset (e.g. h5py or z5py).

    Also base class for hdf5, n5/zarr and pyramid source.
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

    @property
    def min_val(self):
        return self._min_val

    @min_val.setter
    def min_val(self, min_val):
        if np.iinfo(self.dtype).min < min_val < np.iinfo(self.dtype).max:
            raise ValueError("Invalid min value")
        self._min_val = min_val

    @property
    def max_val(self):
        return self._max_val

    @max_val.setter
    def max_val(self, max_val):
        if np.iinfo(self.dtype).min < max_val < np.iinfo(self.dtype).max:
            raise ValueError("Invalid max value")
        self._max_val = max_val


class ZarrSource(BigDataSource):
    """ Source from zarr dataset.
    """
    def __init__(self, data, **kwargs):
        if not elf.io.is_z5py(data) and elf.io.is_dataset(data):
            raise ValueError("ZarrSource expecsts a z5 dataset, not %s" % type(data))
        super().__init__(data, **kwargs)


class HDF5Source(BigDataSource):
    """ Source from hdf5 dataset.
    """
    def __init__(self, data, **kwargs):
        if not elf.io.is_h5py(data) and elf.io.is_dataset(data):
            raise ValueError("HDF5Source expects a h5py dataset, not %s" % type(data))
        super().__init__(data, **kwargs)


class PyramidSource(BigDataSource):
    """ Source for pyramid dataset.

    For now, we support the bdv mipmap and the n5 mipmap format used by paintera.
    """
    def __init__(self, group, pyramid_format=None,
                 n_scales=None, n_threads=1, **kwargs):
        expected_format = infer_pyramid_format(group)
        if pyramid_format is None:
            if expected_format is None:
                raise ValueError("Invalid pyramid source")
        else:
            if expected_format != pyramid_format:
                raise ValueError("Expected format %s, got %s" % (expected_format,
                                                                 pyramid_format))
        self._format = expected_format
        self._group = group
        self._n_threads = n_threads
        # number of scales can be inferred from data or given
        self.max_n_scales = len(group)
        if n_scales is None:
            self._n_scales = self.max_n_scales
        else:
            if n_scales > self.max_n_scales:
                raise ValueError
            self._n_scales = n_scales

        super().__init__(self.get_level(0), **kwargs)

    @property
    def format(self):
        return self._format

    @property
    def group(self):
        return self._group

    @property
    def n_scales(self):
        return self._n_scales

    @n_scales.setter
    def n_scales(self, n_scales):
        if n_scales > self.max_n_scales:
            raise ValueError("Invalid number of scales")
        self._n_scales = n_scales

    @property
    def n_threads(self):
        return self._n_threads

    @n_threads.setter
    def n_threads(self, n_threads):
        self._n_threads = n_threads

    def get_level(self, level):
        """ Load the dataset at given level
        """
        ds = self.group['s%i' % level] if self.format == 'n5' else\
            self.group['%i/cells' % level]
        ds.n_threads = self.n_threads
        return ds

    def get_pyramid(self):
        """ Load the pyramid in format expected by napari.add_pyramid
        """
        pyramid = [self.get_level(scale) for scale in range(self.n_scales)]
        # for n5, we load the last pyramid level into memory,
        # because it cannot be passed to np.asarray, which is done by napari
        # see also https://github.com/constantinpape/z5/issues/120
        if self.format == 'n5':
            pyramid[-1] = pyramid[-1][:]
        return pyramid
