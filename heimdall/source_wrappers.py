from abc import ABC
import elf.wrapper
from elf.util import normalize_index
from .sources import Source, BigDataSource, PyramidSource


class SourceWrapper(ABC):
    """ Source wrapper base class
    """
    def __init__(self, source):
        if not isinstance(source, (Source, SourceWrapper)):
            raise ValueError("SourceWrapper can only wrap a heimdall.Source or another source wrapper.")
        if isinstance(source, PyramidSource):
            raise ValueError("""SourceWrapper cannot wrap a PyramidSource.
                             Define a factory and use PyramidSource(..., wrapper_factory=factory) instead.""")
        self._source = source
        self._is_big_data_source = source.is_big_data_source if isinstance(source, SourceWrapper)\
            else isinstance(source, BigDataSource)

    @property
    def is_big_data_source(self):
        return self._is_big_data_source

    @property
    def source(self):
        return self._source

    # expose min val and max val
    @property
    def min_val(self):
        return self.source._min_val if self.is_big_data_source\
            else self.source.data.min()

    @property
    def max_val(self):
        return self.source._max_val if self.is_big_data_source\
            else self.source.data.max()

    # by default, we just return name, layer_type, multichanne, ndim, shape and dtype
    # of the source that is being wrapped and also use its get and set item
    # this needs to be overridden by the inheriting classes
    @property
    def name(self):
        return self.source.name

    @property
    def layer_type(self):
        return self.source.layer_type

    @property
    def multichannel(self):
        return self.source.multichannel

    @property
    def ndim(self):
        return self.source.ndim

    @property
    def shape(self):
        return self.source.shape

    @property
    def dtype(self):
        return self.source.dtype

    def __getitem__(self, key):
        return self.source[key]

    # TODO disable setitem for immutable sources (once we have them)
    def __setitem__(self, key, item):
        self.source[key] = item


# source wrappers:
# - roi
# - resize on the fly
# TODO
# - apply affines on the fly
# - data caching


class RoiWrapper(SourceWrapper):
    """ Wrapper to expose only a roi of the source.
    """
    def __init__(self, source, roi_start=None, roi_stop=None):
        super().__init__(source)
        self._roi_start = self.format_roi_start(roi_start, source.shape)
        self._roi_stop = self.format_roi_stop(roi_stop, source.shape)
        self._check_roi()

    @staticmethod
    def format_roi_start(roi_start, shape, perform_check=True):
        roi_start = (0,) * len(shape) if roi_start is None else roi_start
        if perform_check and any(rs >= sh for rs, sh in zip(roi_start, shape)):
            raise ValueError("Invalid roi start")
        return roi_start

    @staticmethod
    def format_roi_stop(roi_stop, shape, perform_check=True):
        roi_stop = (0,) * len(shape) if roi_stop is None else roi_stop
        if perform_check and any(rs > sh for rs, sh in zip(roi_stop, shape)):
            raise ValueError("Invalid roi stop")
        return roi_stop

    def _check_roi(self):
        if any(sta >= sto for sta, sto in zip(self.roi_start, self.roi_stop)):
            raise ValueError("Invalid roi")

    @property
    def roi_start(self):
        return self._roi_start

    @roi_start.setter
    def roi_start(self, roi_start):
        self._roi_start = self.format_roi_start(roi_start)
        self._check_roi()

    @property
    def roi_stop(self):
        return self._roi_stop

    @roi_stop.setter
    def roi_stop(self, roi_stop):
        self._roi_stop = self.format_roi_stop(roi_stop)
        self._check_roi()

    @property
    def shape(self):
        return tuple(sto - sta for sta, sto in zip(self.roi_start, self.roi_stop))

    def __getitem__(self, key):
        mapped_key = normalize_index(key, self.shape)
        mapped_key = tuple(slice(k.start - rs, k.stop - rs)
                           for k, rs in zip(mapped_key, self.roi_start))
        return self.source[mapped_key]

    # TODO
    def __setitem__(self, key, item):
        raise NotImplementedError


def roi_wrapper_pyramid_factory(source, scale, roi_start=None, roi_stop=None):
    """ Pyramid factory for the RoiWrapper.

    Use this by binding `roi_start` and `roi_stop` corresponding to level 0 with partial:
    ```
    # roi with respect to level 0
    roi_start = (0, 0, 0)
    roi_stop = tuple(sh // 2 for sh in source.shape)
    factory = partial(roi_wrapper_pyramid_factory, roi_start=roi_start, roi_stop=roi_stop)
    pyramid_source = PyramidSource(..., wrapper_factory=factory)
    ```

    Arguments:
        source [heimdall.Source] - source to be wraped
        scale [tuple[int]] - scale factor w.r.t. level 0
        roi_start [tuple[int]] - start coordinates of roi (default: None)
        roi_stop [tuple[int]] - stop coordinates of roi (default: None)
    """
    # this is the shape at level 0, which is monkey patched into to every level's source
    # before the wrapper function is called
    shape_zero = source.shape_zero
    roi_start = RoiWrapper.format_roi_start(roi_start, shape_zero)
    roi_start = tuple(rs // sc for rs, sc in zip(roi_start, scale))

    roi_stop = RoiWrapper.format_roi_start(roi_stop, shape_zero)
    roi_stop = tuple(rs // sc for rs, sc in zip(roi_stop, scale))

    return RoiWrapper(source, roi_start, roi_stop)


# TODO check if we even need this or if napari supports this out of the box already
class ResizeWrapper(SourceWrapper):
    """ Wraper to resize the source on the fly.
    """
    def __init__(self, source, shape, order=0):
        if source.multichannel:
            raise NotImplementedError
        super().__init__(source)
        self._resized = elf.wrapper.ResizedVolume(source, shape, order)

    @property
    def shape(self):
        return self._resized.shape

    def __getitem__(self, key):
        return self._resized[key]

    # TODO disable setitem for immutable sources (once we have them)
    def __setitem__(self, key, item):
        raise NotImplementedError


class AffineWrapper(SourceWrapper):
    pass


class CacheWrapper(SourceWrapper):
    pass
