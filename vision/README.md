# Sample app for analyzing facial emotion using Affectiva's ICS SDK (C++ native library for Android)

### frame-detector-video-demo

This sample demonstrates use of the SyncFrameDetector class, getting its input from a video file. It analyzes received frames and displays the results on screen.

After building, run the command `./frame-detector-video-demo --help` for information on its command line options.

---

## Dependencies

#### Affectiva Vision library

The Affectiva ICS SDK is available upon request. To get access, please [contact us](https://auto.affectiva.com).


#### Linux (x86_64, aarch64)
Branches like this one that have a name that ends in "qc", are Android-specific (ARM64). For samples that work on Linux, switch to the branch whose name that matches the release you're using.
##Instructions

### To build docker image and run it 
Using the Dockerfile, we'll build the artifact and push it to the target device (tested with Qualcomm SA-8155P development board)

A Dockerfile is located in the top-level directory of this repo ([here](../Dockerfile.android)). To build the docker image, please refer to the same file for instructions.

### To copy artifact from docker image and push it to the target device
Create and start a docker container.<br />
`$ docker create -ti --name affectiva-ics-sdk affectiva:ics-sdk-$GIT_DESCRIBE bash` 

Copy the generated artifact (which was created during docker build) to specified path.<br />
`$ docker cp affectiva-ics-sdk:/opt/testapp-artifact.tar.gz <target/path/to/copy/artifact>` 

Delete the created container.<br />
`$ docker rm -f affectiva-ics-sdk` 

Now, setup your adb device where you'll test the app, for more info on adb ([click here](https://developer.android.com/studio/command-line/adb )) 

It is recommended to run all the below commands after running `sudo -s` on the terminal.

Restart adb connection with root privileges (use this before starting any push operations).<br />
`$ adb root` 

After establishing successful connection to the target device (can be verified by `$ adb devices`), push the artifact to it by using following command

`$ abd push <path/where/the/artifact/resides> <path/on/the/target/deive>`

Usually it's recommended to push the application to `/data/loca/tmp/` directory on the target device.
One can either push the .tar.gz file directly and then extract it on the target device or extract the file locally and then push it to the target device. 
The former method is faster and easier

Copy the artifact, and use same approach to copy any other files to the target device. <br />
`$ adb push testapp-artifact.tar.gz /data/loca/tmp/` 

Note:: The above step might take up to 100 secs, depends on the transfer speed.

###To setup and run frame-detector-video-demo

Enter into android's shell env.<br />
`$ adb shell`  

Change the directory where the artifact is present.<br />
`$ cd /data/local/tmp/` 

Extract artifact into the current directory.<br />
`$ tar -xf testapp-artifact.tar.gz` 

Set $LD_LIBRARY_PATH env variable.<br />
`$ export LD_LIBRARY_PATH=/data/local/tmp/test/lib/arm64-v8a` 

Set data directory $AFFECTIVA_VISION_DATA_DIR env variable.<br />
`$ export AFFECTIVA_VISION_DATA_DIR=/data/local/tmp/test/data` 

Change the directory where the sample app is present.<br />
`$ cd bin` 

run the command `./frame-detector-video-demo --help` for information on how to run it.

eg: `$ ./frame-detector-video-demo -i XYZ.mp4 `

Above command will create .csv file in the same directory where the input video is present, in this case it will
 create XYZ_faces.csv. To pull this file from the target device to host machine. Do the following steps
 
 Log out from the target device.<br />
 `$ exit` 
 
 Copy/ pull csv file to current directory from target device.<br />
 `$ adb pull /data/local/tmp/bin/XYZ_faces.csv` 
 

Note: 
- Once you logout from the target device all the environment variables are deleted. 
- this example code has been tested on the Android device(SA-8155P Qualcomm Board)