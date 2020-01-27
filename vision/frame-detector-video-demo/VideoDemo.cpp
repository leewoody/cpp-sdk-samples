#include "PlottingImageListener.h"
#include "StatusListener.h"
#include "VideoReader.h"
#include "FileUtils.h"

#include <Core.h>
#include <FrameDetector.h>
#include <SyncFrameDetector.h>

#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <iomanip>

static const std::string DISPLAY_DATA_DIR_ENV_VAR = "AFFECTIVA_VISION_DATA_DIR";
static const affdex::str DATA_DIR_ENV_VAR = STR(DISPLAY_DATA_DIR_ENV_VAR);

using namespace std;
using namespace affdex;

int main(int argsc, char ** argsv) {

    const int precision = 2;
    std::cerr << std::fixed << std::setprecision(precision);
    std::cout << std::fixed << std::setprecision(precision);

    // cmd line args
    affdex::path data_dir;
    affdex::path video_path;
    unsigned int sampling_frame_rate;
    bool draw_display;
    unsigned int num_faces;
    bool loop = false;
    bool draw_id = false;
    bool disable_logging = false;

    namespace po = boost::program_options; // abbreviate namespace

    po::options_description description("Project for demoing the Affectiva FrameDetector class (processing video files).");
    description.add_options()
        ("help,h", po::bool_switch()->default_value(false), "Display this help message.")
#ifdef _WIN32
        ("data,d", po::wvalue<affdex::path>(&data_dir),
            std::string("Path to the data folder. Alternatively, specify the path via the environment variable "
                + DISPLAY_DATA_DIR_ENV_VAR + R"(=\path\to\data)").c_str())
        ("input,i", po::wvalue<affdex::path>(&video_path)->required(), "Video file to processs")
#else // _WIN32
        ("data,d", po::value< affdex::path >(&data_dir),
            (std::string("Path to the data folder. Alternatively, specify the path via the environment variable ")
            + DATA_DIR_ENV_VAR + "=/path/to/data").c_str())
        ("input,i", po::value< affdex::path >(&video_path)->required(), "Video file to processs")
#endif // _WIN32
        ("sfps", po::value<unsigned int>(&sampling_frame_rate)->default_value(0), "Input sampling frame rate. Default is 0, which means the app will respect the video's FPS and read all frames")
        ("draw", po::value<bool>(&draw_display)->default_value(true), "Draw video on screen.")
        ("numFaces", po::value<unsigned int>(&num_faces)->default_value(1), "Number of faces to be tracked.")
        ("loop", po::bool_switch(&loop)->default_value(false), "Loop over the video being processed.")
        ("face_id", po::bool_switch(&draw_id)->default_value(false), "Draw face id on screen. Note: Drawing to screen should be enabled.")
        ("quiet,q", po::bool_switch(&disable_logging)->default_value(false), "Disable logging to console")
        ;

    po::variables_map args;

    try {
        po::store(po::command_line_parser(argsc, argsv).options(description).run(), args);
        if (args["help"].as<bool>()) {
            std::cout << description << std::endl;
            return 0;
        }
        po::notify(args);
    }
    catch (po::error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << "For help, use the -h option." << std::endl << std::endl;
        return 1;
    }

    // set data_dir to env_var if not set on cmd line
    data_dir = validatePath(data_dir, DATA_DIR_ENV_VAR);

    if (draw_id && !draw_display) {
        std::cerr << "Can't draw face id while drawing to screen is disabled" << std::endl;
        std::cerr << description << std::endl;
        return 1;
    }

    unique_ptr<vision::SyncFrameDetector> detector;
    try {
        //initialize the output file
        boost::filesystem::path csv_path(video_path);
        csv_path.replace_extension(".csv");
        std::ofstream csv_file_stream(csv_path.c_str());

        if (!csv_file_stream.is_open()) {
            std::cerr << "Unable to open csv file " << csv_path << std::endl;
            return 1;
        }

        // create the FrameDetector
        detector = std::unique_ptr<vision::SyncFrameDetector>(new vision::SyncFrameDetector(data_dir, num_faces));

        // configure the FrameDetector by enabling features
        detector->enable({ vision::Feature::EMOTIONS, vision::Feature::EXPRESSIONS, vision::Feature::IDENTITY, vision::Feature::APPEARANCES});

        // prepare listeners
        PlottingImageListener image_listener(csv_file_stream, draw_display, !disable_logging, draw_id);
        StatusListener status_listener;

        // configure the FrameDetector by assigning listeners
        detector->setImageListener(&image_listener);
        detector->setProcessStatusListener(&status_listener);

        // start the detector
        detector->start();

        do {
            // the VideoReader will handle decoding frames from the input video file
            VideoReader video_reader(video_path, sampling_frame_rate);

            cv::Mat mat;
            timestamp timestamp_ms;
            while (video_reader.GetFrame(mat, timestamp_ms)) {
                // create a Frame from the video input and process it with the FrameDetector
                vision::Frame f(mat.size().width, mat.size().height, mat.data, vision::Frame::ColorFormat::BGR, timestamp_ms);
                detector->process(f);
                image_listener.processResults();
            }

            cout << "******************************************************************" << endl
            << "Processed Frame count: " << image_listener.getProcessedFrames() << endl
            << "Frames w/faces: " << image_listener.getFramesWithFaces() << endl
            << "Percent of frames w/faces: " << image_listener.getFramesWithFacesPercent() << "%" << endl
            << "******************************************************************" << endl;

            detector->reset();
            image_listener.reset();

        } while (loop);

        detector->stop();
        csv_file_stream.close();

        std::cout << "Output written to file: " << csv_path << std::endl;
    }
    catch (std::exception& ex) {
        std::cerr << ex.what();

        // if video_reader couldn't load the video/image, it will throw. Since the detector was started before initializing the video_reader, We need to call `detector->stop()` to avoid crashing
        detector->stop();
        return 1;
    }

    return 0;
}
