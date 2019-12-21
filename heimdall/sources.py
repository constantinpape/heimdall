from abc import ABC
import numpy as np
import elf.io
try:
    import torch
except ImportError:
    torch = None


def check_consecutive(scales, expected_start_id=0):
    scales = sorted(scales)
    is_consecutive = (scales[0] == expected_start_id) and\
        (scales == list(range(scales[0], scales[-1] + 1)))
    return is_consecutive


def is_bdv(keys):
    try:
        scales = [int(scale) for scale in keys]
        is_consecutive = check_consecutive(scales)
        return is_consecutive
    except Exception:
        return False


def is_imaris(keys):
    try:
        scales = [int(scale.split(' ')[1]) for scale in keys]
        is_consecutive = check_consecutive(scales)
        return is_consecutive
    except Exception:
        return False


def infer_pyramid_format(group):
    """ Infer pyramid format from group object.

    Checks for bdv / imaris multiscale format (hdf5) or format used by paintera (n5).
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
        if is_bdv(keys):
            return 'bdv'
        elif is_imaris(keys):
            return 'imaris'
        else:
            return None

    # check for knossos multiscale format
    elif elf.io.is_knossos(group) and elf.io.is_group(group):
        try:
            # all names are prefixed with 'mag'
            scales = [int(scale[3:]) for scale in keys]
            is_consecutive = check_consecutive(scales, 1)
            if is_consecutive:
                return 'knossos'
            else:
                return None
        except Exception:
            return None
        return 'knossos'

    return None


# TODO add 'rgb' attribute
class Source(ABC):
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
                           'float64': 'raw'}

    def to_layer_type(self, layer_type, dtype):
        if layer_type is not None:
            if layer_type not in self.layer_types:
                raise ValueError("Layer type %s is not supported" % layer_type)
            return layer_type
        else:
            return self.default_layer_types[str(dtype)]

    def to_scale(self, scale):
        if scale is None:
            return (1,) * self.ndim
        if isinstance(scale, int):
            return (scale,) * self.ndim
        elif isinstance(scale, (tuple, list)):
            if len(scale) != self.ndim:
                raise ValueError("Invalid length of scale value, expected %i, got %i" % (len(scale),
                                                                                         self.ndim))
            return tuple(scale)
        raise ValueError("Invald type of scale, expected one of (int, tuple, list), got %s" % type(scale))

    def __init__(self, data, layer_type=None, name=None,
                 channel_axis=None, scale=None, split_channels=False):
        self._data = data
        self._layer_type = self.to_layer_type(layer_type, data.dtype)
        self._name = name
        self._channel_axis = channel_axis
        self._split_channels = split_channels
        self._scale = self.to_scale(scale)

    def __getitem__(self, key):
        return self.data[key]

    # TODO we should have an option to declare a source as mutable / immutable
    # and disable setitem if appropriate
    def __setitem__(self, key, item):
        self.data[key] = item

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self.scale = self.to_scale(scale)

    @property
    def data(self):
        return self._data

    @property
    def ndim(self):
        return self._data.ndim - 1 if self.channel_axis else self._data.ndim

    # maybe we should also support channel dim last, but I don't know how napari deals with this
    @property
    def shape(self):
        return self._data.shape[1:] if self.channel_axis else self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def channel_axis(self):
        return self._channel_axis

    @property
    def split_channels(self):
        return self._split_channels

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


class TorchSource(Source):
    """ Source from torch tensor.
    """
    def __init__(self, data, **kwargs):
        assert torch is not None, "Need torch to support torch tensor source"
        if not torch.is_tensor(data):
            raise ValueError("TorchSource expecsts a torch tensor, not %s" % type(data))
        # bring the data to the cpu and squeeze the potential singleton
        # in the batch axis
        data = data.detach().cpu().numpy().squeeze(0)
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


class KnossosSource(BigDataSource):
    """ Source from Knossos dataset.
    """
    def __init__(self, data, **kwargs):
        if not elf.io.is_knossos(data) and elf.io.is_dataset(data):
            raise ValueError("KnossosSource expects a knossos dataset, not %s" % type(data))
        super().__init__(data, **kwargs)


class PyramidSource(BigDataSource):
    """ Source for pyramid dataset.

    For now, we support the following fomrats:
        - bdv mipmap (stored as h5)
        - imaris format (stored as h5)
        - n5 mipmap format used by paintera
        - pyknossos file

    Arguments:
        group [] - the root group of the pyramid store
        pyramid_format [str] - the pyramid format,
            will be infered fron `group` by default (default: None)
        n_scales [int] - the number of available scale levels.
            Set to the max number of scales by default (default: None)
        n_threads [int] - number of threads used in z5py backends (default: 1)
        wrapper_factory [callable] - factory for a wrapper function applied to each scale
            (default: None)
    """
    supported_formats = ('n5', 'knossos', 'bdv', 'imaris')

    def __init__(self, group, pyramid_format=None,
                 n_scales=None, n_threads=1, wrapper_factory=None,
                 **kwargs):
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

        # we need to set the wrapper factory to None before
        # we can calculate the scale factors and call super
        self._wrapper_factory = None
        self._scales = self._init_scales()
        super().__init__(self.get_level(0), **kwargs)

        # check if we have a wrapper factory
        if wrapper_factory is not None:
            if not callable(wrapper_factory):
                raise ValueError("Invalid wrapper factory")
            self._wrapper_factory = wrapper_factory
            # we need to override the init
            super().__init__(self.get_level(0), **kwargs)

    def _init_scales(self):
        ref_shape = self.get_level(0).shape
        ndim = len(ref_shape)
        scales = [ndim * (1,)]
        for level in range(1, self.n_scales):
            shape = self.get_level(level).shape
            scale = tuple(int(round(rsh / sh)) for rsh, sh in zip(ref_shape, shape))
            scales.append(scale)
        return scales

    @property
    def scales(self):
        return self._scales

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

    def wrap(self, source, level):
        factory = self._wrapper_factory
        if factory is None:
            return source

        # wrap source in a big data source, so we can pass
        # it to the source wrapper
        source = BigDataSource(source)
        # scale factor at the current level
        scale = self.scales[level]

        # we allow different arguments for the wrapper factory:
        # - factory(source, scale, level)
        # - factory(source, scale)
        # - factory(source, level)
        # - factory(source)
        try:
            source = factory(source, scale=scale, level=level)
            return source
        except TypeError:
            pass
        try:
            source = factory(source, scale=scale)
            return source
        except TypeError:
            pass
        try:
            source = factory(source, level=level)
            return source
        except TypeError:
            pass

        source = factory(source)
        return source

    def get_level(self, level):
        """ Load the dataset at given level
        """
        if self.format == 'n5':
            source = self.group['s%i' % level]
        elif self.format == 'bdv':
            source = self.group['%i/cells' % level]
        elif self.format == 'knossos':
            source = self.group['mag%i' % (level + 1)]
        elif self.format == 'imaris':
            # We don't support multi-channel / multi-time point
            # but should expose it somehow
            source = self.group['ResolutionLevel %i/TimePoint 0/Channel 0/Data' % level]
        source = self.wrap(source, level)
        return source

    def get_pyramid(self):
        """ Load the pyramid in format expected by napari.add_image(is_pyramid=True)
        """
        pyramid = [self.get_level(scale) for scale in range(self.n_scales)]
        # we load the last pyramid level into memory,
        # because it cannot be passed to np.asarray for z5py and knossos
        # (which is done by napar)
        pyramid[-1] = pyramid[-1][:]
        return pyramid
