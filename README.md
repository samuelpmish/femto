I'm reworking the build system to get rid of submodules, so it's a bit of a mess right now. I'll try to write some better guides on build options soon.

----------

On Ubuntu 24.04 when enabling graphics, one should install a few packages in advance:

```
sudo apt install libglu1-mesa-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

## Build Instructions

Before building femto itself, we need to build its external dependencies.
To do this, configure and build the cmake project in the `dependencies` directory.

```
cd dependencies
cmake . -Bbuild -GNinja
cd build
ninja
```

After that, `cd` back to `femto`'s base directory and configure the main project.
Modify the provided `options.cmake` file to set the desired options

```
cd /path/to/femto
cmake . -C options.cmake -Bbuild -GNinja
```



