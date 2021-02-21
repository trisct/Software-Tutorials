# GAPS

GAPS is a library for geometry processing.

### Installation

#### 1. Clone the repository
```
git clone https://github.com/tomfunkhouser/gaps
```

#### 2. Install dependencies
```
apt install mesa-common-dev libglu1-mesa-dev libosmesa6-dev libxi-dev libgl1-mesa-dev libgl1-mesa-glx libglew-dev
```

#### 3. Make project
```
cd gaps
make mesa -j8
```

__Known issue__: When compiling with gcc, the error might appear that `GLAPI does not name a type`. In this case, go to `/usr/include/GL` and open `osmesa.h` in an editor and change all `GLAPI` preceeding function definitions to `extern`.
