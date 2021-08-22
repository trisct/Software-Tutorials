# Building LibTorch App

Do as follows

1. Extract the include and library files downloaded from the official website to a predefined path

2. Add the library folder to the `LD_LIBRARY_PATH` environment variable (for dynamic linking)

   **Note that adding the LibTorch library path to `LD_LIBRARY_PATH` possibly causes PyTorch imports to fail with segmentation faults. So remember to reset this EV when switching back to PyTorch.**

3. Add the appropriate flags to the compiler (if you compile with terminal directly). See the custom `build.sh` for an example

