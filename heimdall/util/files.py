import os
import h5py
import z5py
from z5py.dataset import Dataset as Z5Dataset
from z5py.group import Group as Z5Group
from ..sources import to_source, infer_pyramid_format

H5_EXTENSIONS = ('.h5', '.hdf', '.hdf5')
ZARR_EXTENSIONS = ('.zr', '.zarr', '.n5')


def open_file(path, mode='a'):
    ext = os.path.splitext(path)[1]
    if ext.lower() in H5_EXTENSIONS:
        return h5py.File(path, mode=mode)
    elif ext.lower() in ZARR_EXTENSIONS:
        return z5py.File(path, mode=mode)
    else:
        raise RuntimeError("Could not infer file type from extension %s" % ext)


def is_pyramid_ds(name, node):

    def isint(x):
        try:
            int(x)
            return True
        except Exception:
            return False

    name = os.path.split(name)[-1]
    if isinstance(node, h5py.Dataset) and name == 'cells':
        return True
    if isinstance(node, Z5Dataset) and name.startswith('s') and isint(name[1:]):
        return True
    return False


def load_sources_from_file(f, reference_ndim,
                           exclude_names=None, include_names=None,
                           load_into_memory=False, n_threads=1):
    sources = []

    def visitor(name, node):

        if isinstance(node, h5py.Dataset) or isinstance(node, Z5Dataset):
            # check if this is in the exclude_names
            if exclude_names is not None:
                if name in exclude_names:
                    return

            # check if this is in the include_names
            if include_names is not None:
                if name not in include_names:
                    return

            # check if this is a dataset of a pyramid group
            # and don't load if it is
            if is_pyramid_ds(node, name):
                return

            # set the number of threads (only has an effect for z5py datasets)
            # and load into memory if specified
            node.n_threads = n_threads
            if load_into_memory:
                node = node[:]

            # check the number of dimensions against the reference dimensionality
            ndim = node.ndim
            multichannel = False
            if ndim == reference_ndim + 1:
                multichannel = True
            elif ndim != reference_ndim:
                return

            sources.append(to_source(node, multichannel=multichannel))

        elif isinstance(node, h5py.Group) or isinstance(node, Z5Group):
            pyramid_format = infer_pyramid_format(node)
            if pyramid_format is not None:
                sources.append(to_source(node, pyramid_format=pyramid_format,
                                         multichannel=multichannel))

    f.visititems(visitor)
    return sources
