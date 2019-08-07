import os
from elf.io import is_dataset, is_group, is_h5py, is_z5py
from ..sources import to_source, infer_pyramid_format


def is_pyramid_ds(name, node):

    def isint(x):
        try:
            int(x)
            return True
        except Exception:
            return False

    name = os.path.split(name)[-1]
    if is_h5py(node) and name == 'cells':
        return True
    if is_z5py(node) and name.startswith('s') and isint(name[1:]):
        return True
    return False


def load_sources_from_file(f, reference_ndim,
                           exclude_names=None, include_names=None,
                           load_into_memory=False, n_threads=1):
    sources = []

    def visitor(name, node):

        if is_dataset(node):
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
            if is_pyramid_ds(name, node):
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

        elif is_group(node):
            pyramid_format = infer_pyramid_format(node)
            if pyramid_format is not None:
                sources.append(to_source(node, pyramid_format=pyramid_format,
                                         multichannel=multichannel))

    f.visititems(visitor)
    return sources
