# Heimdall

Python based viewer for large multi-dimensional datasets.

Based on [napari](https://github.com/napari/napari) and inspired by [BigDataViewer](https://imagej.net/BigDataViewer).
Can display datasets in memory or stored in [hdf5](https://www.hdfgroup.org/solutions/hdf5/),
[bdv-format](https://imagej.net/BigDataViewer#Exporting_Datasets_for_the_BigDataViewer), [zarr](https://github.com/zarr-developers/zarr-python), [n5](https://github.com/saalfeldlab/n5) or [knossos file format](https://github.com/adwanner/PyKNOSSOS).
This is work in progress.


## Installation

### From Source

After cloning this repository, you can install `heimdall` via
```
python setup.py install
```
It requires the following dependencies:
- `napari`
- [`elf`](https://github.com/constantinpape/elf) (will be available via pip and conda soon)

Optionally dependencies for viewing big data:
- `h5py`
- `z5py`

### Via pip

Coming soon ;).

### Via conda

Coming soon ;).


## Usage

`Heimdall` is a wrapper around `napari` that makes common visualisation tasks for large volumetric data more convenient.
In particular, it supports visualizing data from `numpy` arrays and `hdf5` as well as `zarr/n5` datasets and `knossos` files.
It also supports some pyramid specifications for these file formats.

It is the easiest to use it through the convenience functions `view_arrays`, which displays a list of `numpy` arrays and `view_container`, which displays the content of a `hdf5` or `zarr/n5` file:

```python
import numpy as np
from heimdall import view_arrays

shape = (128,) * 3
x = np.random.rand(*shape)
y = np.random.randint(0, 1000, size=shape, dtype='uint32')
# Display x as raw data and y as labels (automatically inferred from the dtypes).
view_arrays([x, y])
```

```python
from heimdall import view_container
path = '/path/to/file.h5'  # or .n5/.zarr
# Display all 3d datasets in the container.
# To exclude selected datasets, pass their names as list `exclude_names`.
# To only show selected datasets, pass their names as list `include_names`.
view_container(path, ndim=3)
```
`view_container` is also installed as command-line script.

In order to use `heimdall` in a more flexible manner, use the function `view`.
It can be called with `numpy` arrays as well as `z5py/h5py` datasets or groups (for pyramids).
It also supports [`heimdall.sources`](https://github.com/constantinpape/heimdall/blob/master/heimdall/sources.py), which allow to customize the viewer further.

```python
import numpy as np
import h5py
from heimdall import view, to_source

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

### Pyramid sources

For now, `heimdall` supports three different multi-scale pyramid formats:
- [bdv-hdf5](https://imagej.net/BigDataViewer#Exporting_Datasets_for_the_BigDataViewer)
- [paintera-n5](https://imagej.net/BigDataViewer#Exporting_Datasets_for_the_BigDataViewe://github.com/saalfeldlab/paintera#raw)
- [knossos](https://github.com/adwanner/PyKNOSSOS)
- [imaris-hdf5](http://open.bitplane.com/Default.aspx?tabid=268)

You can load a pyramid, by passing the `z5py.Group` / `h5py.Group` or the corresponding knossos file to `view`,
or wrapping it into a `PyramidSource` with `to_source` in order to specify further options.

```python
import h5py
import z5py
from heimdall import view, to_source

f1 = z5py.File('/path/to/file.n5')
# this needs to be a group containing an n5 pyramid
pyramid1 = f1['n5-pyramid-group']

# this needs to be a bdv hdf5 file
with h5py.File('/path/to/file.h5', 'r') as f2:
    # this is the pyramid for timepoint 0, channel 0 in the bdv format
    pyramid2 = f2['t00000/s00']
    # we wrap it into a source to specify further options
    # (here: the maximum scale level to be loaded)
    pyramid2 = to_source(pyramid2, n_scales=3)

    # both pyramid data-sets need to have the same shape (at scale 0)
    # note that we can call shape on pyramid2 directly, because this is exposed by 
    # the `PyramidSource` 
    asserrt pyramid1['s0'].shape == pyramid2.shape
    view(pyramid1, pyramid2)
```

### Source wrappers

`Heimdall` provides [several source wrappers](https://github.com/constantinpape/heimdall/blob/master/heimdall/source_wrappers.py) - classes that wrap a source and
perform some tranformation on the fly.
For example, the `RoiWrapper` only exposes a sub-region of the data to the viewer:

```python
import z5py
from heimdall import view, to_source
from heimdall.source_wrappers import RoiWrapper

# load the dataset source we want to view
ds = z5py.File('/path/to/file.n5')['some/name']
# wrap it into a source (this is required in order to pass it to the wrapper)
source = to_source(ds)

# specify the roi (assuming this is 3d data with a matching shape!)
roi_start = (0, 100, 150)
roi_stop = (200, 250, 400)
# wrap the source
source = RoiWrapper(source, roi_start, roi_stop)
print(source.shape)
# (200, 150, 250)

view(source)
```

Source wrappers can be chained.

They can also be applied to pyramids, however a `PyramidSource` cannot be wrapped into
a `SourceWrapper` directly. Instead, the `PyramidSource` can be passed a factory function `factory(source, level, scale)`,
that needs to adjust the transformation to the individual scale level. For many source wrappers, this function is alrady implemented:

```python
from functools import partial
import z5py
from heimdall import view, to_source
from heimdall.source_wrappers import roi_wrapper_pyramid_factory

# load the pyramid source we want to view
g = z5py.File('/path/to/file.n5')['some/n5-pyramid']

# construct the wrapper factory function,
# the values for `roi_start` and `roi_stop` are specified with partial
roi_start = (0, 100, 150)
roi_stop = (200, 250, 400)
factory = partial(roi_wrapper_pyramid_factory, roi_start=roi_start, roi_stop=roi_stop)

# construct the source with wrapper factory
source = to_source(g, wrapper_factory=factory)

view(source)
```


### Interacting with napari

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
    viewer.add_points(points, size=sizes)
```

See `examples/` for additional usage examples.
