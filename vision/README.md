# Sample apps for analyzing facial emotion using Affectiva's Automotive SDK for Linux
### face-registration-demo

This sample demonstrates use of the vision::FaceRegistrar class, getting its input from a video file. It analyzes received frames and registers a single face it finds.  It can also list and remove existing face registrations.

After building, run the command `./face-registration-demo --help` for information on its command line options.

---

### frame-detector-webcam-demo

This sample demonstrates use of the SyncFrameDetector class, getting its input from a webcam. It analyzes received frames and displays the results on screen.

After building, run the command `./frame-detector-webcam-demo --help` for information on its command line options.

---

### frame-detector-video-demo

This sample demonstrates use of the SyncFrameDetector class, getting its input from a video file. It analyzes received frames and displays the results on screen.

After building, run the command `./frame-detector-video-demo --help` for information on its command line options.

---

## Dependencies

#### Affectiva Vision library

The Vision Library is packaged with the Automotive SDK, which is available upon request. To get access, please [contact us](https://auto.affectiva.com/).

#### Additional Dependencies

Install additional dependencies with the following command:  
Ubuntu:`$ sudo apt install -y wget git build-essential libopencv-dev cmake libgtk2.0-dev pkg-config libjpeg-dev libpng-dev libtiff-dev libavformat-dev libavcodec-dev libswscale-dev`  

[Click here](#ubuntu-18) for instructions on building sample apps for Ubuntu 18.

[Click here](#ubuntu-16) for instruction on building sample apps for Ubuntu 16.

## Ubuntu 18  

### Boost
Install boost directly from the package manager with the following command:   

Ubuntu: `$ sudo apt install -y libboost-dev libboost-filesystem1.65-dev libboost-program-options1.65-dev libboost-system1.65-dev`

### OpenCV  
We do not recommend running the sample apps with OpenCV 3.2.0 from the package manager. Instead, you must build opencv 2.4.13 from source. Other OpenCV versions 2.4.** may work but we recommend 2.4.13 as a tested depedency for the sample apps on Ubuntu 18.

Fetch the OpenCV source from github and checkout the 2.4.13 branch with the following commands:

`$ git clone -b 2.4.13 --depth 1 https://github.com/opencv/opencv.git` <br/>

To make OpenCV 2.4.13 compatible with gcc7, change the following code block that begins on line 67 in the file `OpenCVDetectCXXCompiler.cmake` (this file is located at /path/to/opencv/cmake/OpenCVDetectCXXCompiler.cmake):   

```
66 elseif(CMAKE_COMPILER_IS_GNUCXX)
67  execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
68                OUTPUT_VARIABLE CMAKE_OPENCV_GCC_VERSION_FULL

```

Replace the code block above with the one below:

```
66 elseif(CMAKE_COMPILER_IS_GNUCXX)
67  execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} -dumpfullversion
68                OUTPUT_VARIABLE CMAKE_OPENCV_GCC_VERSION_FULL

```


### Building OpenCV 2.4.13 with CMake

```
cd opencv
mkdir build install
cd build

CMAKE_ARGS="-DWITH_FFMPEG=ON \
-DBUILD_TESTS=OFF \
-DBUILD_PERF_TESTS=OFF \
-DCMAKE_BUILD_TYPE=RELEASE \
-DENABLE_PRECOMPILED_HEADERS=OFF \
-DCMAKE_INSTALL_PREFIX=/path/to/opencv/install"

cmake $CMAKE_ARGS ../
make -j4
make install
```
----
### Building Samples with CMake

Specify the following CMake variables to identify the locations of various dependencies:

- **AFFECTIVA_SDK_DIR**: set path to the folder where the Automotive SDK is installed (`/path/to/auto-sdk`)
- **BOOST_ROOT**: set path to the `/usr/` directory
- **OpenCV_DIR**: set path to the opencv src tree (`/path/to/opencv/install/share/OpenCV`)

#### Linux (x86_64)

Example script (replace directories starting with `/path/to` as appropriate):  
```
# create a build directory
mkdir vision-samples-build/ vision-samples-install/
cd vision-samples-build

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release \
-DAFFECTIVA_SDK_DIR=/path/to/auto-sdk \
-DBOOST_ROOT=/usr/ \
-DOpenCV_DIR=/path/to/opencv/install/share/OpenCV \
-DCMAKE_INSTALL_PREFIX=/path/to/cpp-sdk-samples/vision/vision-samples-install"

cmake $CMAKE_ARGS /path/to/cpp-sdk-samples/vision
make -j4
make install

```
Setup runtime configurations by running the `config.sh` bash script with the following commands:
```
$ export AFFECTIVA_SDK_DIR=/path/to/auto-sdk
$ ../config.sh

```

---

## Ubuntu 16

### Boost
```
$ mkdir boost-build
$ cd boost-build
$ wget https://sourceforge.net/projects/boost/files/boost/1.63.0/boost_1_63_0.tar.gz
$ tar -xzvf boost_1_63_0.tar.gz
$ cd boost_1_63_0
$ ./bootstrap.sh
$ sudo ./b2 -j $(nproc) cxxflags=-fPIC threading=multi runtime-link=shared \
      --with-filesystem --with-program_options install
```

### OpenCV
Install OpenCV directly from the package manager with the following command:

`$ sudo apt install -y libopencv-dev`

### Building with CMake

Specify the following CMake variables to identify the locations of various dependencies:

- **AFFECTIVA_SDK_DIR**: set path to the folder where the Automotive SDK is installed (`/path/to/auto-sdk`)
- **BOOST_ROOT**: set path to the Boost src tree (`/path/to/boost-build/`)
- **OpenCV_DIR**: set path to the `/usr/` directory


#### Linux (x86_64, aarch64)

Example script (replace directories starting with `/path/to` as appropriate):
```
# create a build directory
mkdir vision-samples-build/ vision-samples-install/
cd vision-samples-build

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release \
-DAFFECTIVA_SDK_DIR=/path/to/auto-sdk \
-DBOOST_ROOT=/path/to/boost-build \
-DOpenCV_DIR=/usr/ \
-DCMAKE_INSTALL_PREFIX=/path/to/cpp-sdk-samples/vision/vision-samples-install"

cmake $CMAKE_ARGS /path/to/cpp-sdk-samples/vision
make -j4
make install

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AFFECTIVA_SDK_DIR/lib
```

## Additional Note

The Affectiva SDK statically links a customized version of OpenCV, so if you run into double free or corruption errors, then you will need to preload the OpenCV library installed from package manager.

Use this command to find the path of libopencv_core.so.2.4

`ldconfig -p | grep libopencv_core.so.2.4`

Then set LD_PRELOAD to ensure it gets loaded first at runtime:
`export LD_PRELOAD=/path/to/libopencv_core.so.2.4`


##Docker Build Instructions

A Dockerfile is located in the top-level directory of this repo ([here](../Dockerfile)). To build the docker image, please refer to that file for instructions.
