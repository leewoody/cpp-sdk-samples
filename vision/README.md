# Sample app for analyzing facial emotion using Affectiva's ICS QC SDK (C++ native library for Android)

### frame-detector-video-demo

This sample demonstrates use of the SyncFrameDetector class, getting its input from a video file. It analyzes received frames and displays the results on screen.

After building, run the command `./frame-detector-video-demo --help` for information on its command line options.

---

## Dependencies

#### Affectiva Vision library

The Vision Library is packaged with the Automotive SDK, which is available upon request. To get access, please [contact us](https://affectiva.atlassian.net/wiki/spaces/auto).

### Building with CMake
Follow the Docker build instructions.

#### Linux (x86_64, aarch64)
Refer to ics-2.2 branch for linux build as ics-2.2-qc is only intended to build Android version

##Docker Build Instructions

A Dockerfile is located in the top-level directory of this repo ([here](../Dockerfile.android)). To build the docker image, please refer to that file for instructions.

Note: This example code can only run on Android device(Qualcomm Board)