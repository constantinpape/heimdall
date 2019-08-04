import os
import h5py
import z5py

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
