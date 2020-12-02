# pyembree

pyembree is a python package for the embree ray tracing technique. This is an installation guide of [pyembree](https://github.com/scopatz/pyembree/).

## 20201128 Version

### Installation

#### Using conda
```
conda install -c conda-forge pyembree
```

#### Using pip

First you need to install [embree2](https://github.com/embree/embree/tree/v2.16.0) which is a legacy version of [embree](https://www.embree.org/downloads.html), to get the include files.

Clone the repository of pyembree. Go into it and build with the following command.
```
pip install .
```

# Use

Note that using the library requires `source [embree2_path/embree-vars.sh]` as well, even after successful installation.
