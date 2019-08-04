# Heimdall

Python based viewer for large multi-dimensional datasets.

Based on [napari](https://github.com/napari/napari) and inspired by [BigDataViewer](https://imagej.net/BigDataViewer).
Can display datasets in memory or stored in [hdf5](https://www.hdfgroup.org/solutions/hdf5/),
[bdv-format](https://imagej.net/BigDataViewer#Exporting_Datasets_for_the_BigDataViewer), [zarr](https://github.com/zarr-developers/zarr-python) or [n5](https://github.com/saalfeldlab/n5).
This is work in progress.


## Installation

### From Source

After cloning this repository, you can install `heimdall` via
```
python setup.py install
```
It requires the following dependencies:
- `h5py`
- `napari`
- `z5py`

### Via conda

Coming soon ;).


## Usage

`Heimdall` is a wrapper around `napari` that makes common visualisation tasks for large volumetric data more convenient.
In particular, it supports visualizing data from `numpy` arrays and `hdf5` as well as `zarr/n5` datasets.
It also supports some pyramid specifications for these file formats.

The easisest way of use is through the convenience functions `simple_view`, which displays a list of `numpy` arrays and `view_container`, which displays the content of a `hdf5` or `zarr/n5` file:

```python
import numpy as np
from heimdall import simple_view

shape = (128,) * 3
x = np.random.rand(*shape)
y = np.random.randint(0, 1000, size=shape, dtype='uint32')
# Display x as raw data and y as labels (automatically inferred from the dtypes).
simple_view([x, y])
```

```python
from heimdall import view_container
path = '/path/to/file.h5'  # or .n5/.zarr
# Display all 3d datasets in the container.
# To exclude certain datasets, pass their names as list `exclude_names`.
# To only show selected datasets, pass their names as list `include_names`.
view_container(path, ndim=3)
```
`view_container` is also installed as script.

In order to use `heimdall` in a more flexible manner, you need to use the function `view`.
It can be called with `numpy` arrays, `z5py/h5py` datasets or groups (for pyramids).
It also supports [`heimdall.sources`](ttps://github.com/constantinpape/heimdall/blob/master/heimdall/sources.py), which allow to customize the viewer further.

```python
import numpy as np
import h5py
from heimdall import viewer, to_source

shape = (128,) * 3
x = np.random.rand(*shape)

path = '/path/to/file.h5'
dset_name = 'some/name'
with h5py.File(path, 'r') as f:
    ds = f[dset_name]

    # We wrap the h5 dataset in a source to specifiy additional options.
    # Here, we specify that the dataset has a channel dimension 
    # and set the min_val and max_val that will be used for normalization by napari.
    y = to_source(ds, min_val=0, max_val=100, multichannel=True)

    # All sources need to have the same shape, otherwise `view` will fail.
    assert x.shape == y.shape
    view(x, y)
```

### Source wrapper

### Interact with napari

`Heimdall` can be combined with `napari` in order to make use of additional functionality.
For this use `view` with `return_viewer=True` and wrap the function call into `napari.gui_qt()`.

```python
import numpy as np
import napari
from heimdall import view

shape = (128,) * 3
x = np.random.rand(*shape)
y = np.random.randint(0, 1000, size=shape, dtype='uint32')

with napari.gui_qt():
    viewer = view(x, y, return_viewer=True)

    # We add an additional napary points layer.
    points = np.array([[64, 64, 64], [32, 64, 96]])
    sizes = np.array([10, 25])
    viewer.add_points(points, size=size)
```

See `examples/` for additional usage examples.
