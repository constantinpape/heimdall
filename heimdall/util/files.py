import os
import h5py
import z5py
from z5py.dataset import Dataset as Z5Dataset
from ..sources import to_source

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


def load_sources_from_file(f, reference_ndim,
                           exclude_names=None, include_names=None,
                           load_into_memory=False, n_threads=1):
    sources = []

    def visitor(name, node):

        if isinstance(node, h5py.Dataset) or isinstance(node, Z5Dataset):
            # check if this is in the exclude names
            if exclude_names is not None:
                if name in exclude_names:
                    return

            # check if this is in the include names
            if include_names is not None:
                if name not in include_names:
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

        # TODO check for mult-scale groups
        # else:

    f.visititems(visitor)
    return sources
