#include "PlottingImageListener.h"
#include "PlottingObjectListener.h"
#include "PlottingOccupantListener.h"
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
static const affdex::Str DATA_DIR_ENV_VAR = STR(DISPLAY_DATA_DIR_ENV_VAR);

using namespace std;
using namespace affdex;
namespace po = boost::program_options; // abbreviate namespace


struct ProgramOptions {
    // cmd line args
    affdex::Path data_dir;
    affdex::Path video_path;
    unsigned int sampling_frame_rate;
    bool draw_display;
    unsigned int num_faces;
    bool loop = false;
    bool draw_id = false;
    bool disable_logging = false;
    bool object_enabled = false;
    bool occupant_enabled = false;
};

void assemble_program_options(po::options_description& description, ProgramOptions& program_options) {

    description.add_options()
        ("help,h", po::bool_switch()->default_value(false), "Display this help message.")
#ifdef _WIN32
    ("data,d", po::wvalue<affdex::Path>(&data_dir),
        std::string("Path to the data folder. Alternatively, specify the path via the environment variable "
            + DISPLAY_DATA_DIR_ENV_VAR + R"(=\path\to\data)").c_str())
    ("input,i", po::wvalue<affdex::Path>(&video_path)->required(), "Video file to processs")
#else // _WIN32
        ("data,d", po::value<affdex::Path>(&program_options.data_dir),
         (std::string("Path to the data folder. Alternatively, specify the path via the environment variable ")
             + DATA_DIR_ENV_VAR + "=/path/to/data").c_str())
        ("input,i", po::value<affdex::Path>(&program_options.video_path)->required(), "Video file to processs")
#endif // _WIN32
        ("sfps",
         po::value<unsigned int>(&program_options.sampling_frame_rate)->default_value(0),
         "Input sampling frame rate. Default is 0, which means the app will respect the video's FPS and read all frames")
        ("draw", po::value<bool>(&program_options.draw_display)->default_value(true), "Draw video on screen.")
        ("numFaces", po::value<unsigned int>(&program_options.num_faces)->default_value(1), "Number of faces to be "
                                                                                            "tracked.")
        ("loop", po::bool_switch(&program_options.loop)->default_value(false), "Loop over the video being processed.")
        ("face_id",
         po::bool_switch(&program_options.draw_id)->default_value(false),
         "Draw face id on screen. Note: Drawing to screen should be enabled.")
        ("quiet,q",
         po::bool_switch(&program_options.disable_logging)->default_value(false),
         "Disable logging to console")
        ("object", "Enable object detection")
        ("occupant", "Enable occupant detection");
}

void process_object_video(unique_ptr<vision::SyncFrameDetector>& detector, std::ofstream& csv_file_stream,
                          ProgramOptions& program_options) {
    //TODO: something similar to process_face_stream

    // create the FrameDetector
    detector = std::unique_ptr<vision::SyncFrameDetector>(new vision::SyncFrameDetector(program_options.data_dir));

    // configure the FrameDetector by enabling features
    detector->enable({vision::Feature::CHILD_SEATS, vision::Feature::PHONES});

    // prepare listeners
    PlottingObjectListener object_listener(csv_file_stream,
                                           program_options.draw_display,
                                           !program_options.disable_logging,
                                           program_options.draw_id,
                                           {{vision::Feature::CHILD_SEATS, 1000}, {vision::Feature::PHONES, 1000}},
                                           detector->getCabinRegionConfig().getRegions());
    StatusListener status_listener;

    // configure the FrameDetector by assigning listeners
    detector->setObjectListener(&object_listener);
    detector->setProcessStatusListener(&status_listener);

    // start the detector
    detector->start();

    do {
        // the VideoReader will handle decoding frames from the input video file
        VideoReader video_reader(program_options.video_path, program_options.sampling_frame_rate);

        cv::Mat mat;
        Timestamp timestamp_ms;
        while (video_reader.GetFrame(mat, timestamp_ms)) {
            // create a Frame from the video input and process it with the FrameDetector
            vision::Frame
                f(mat.size().width, mat.size().height, mat.data, vision::Frame::ColorFormat::BGR, timestamp_ms);
            detector->process(f);
            object_listener.processResults();
        }

        cout << "******************************************************************" << endl
             << "Processed Frame count: " << object_listener.getProcessedFrames() << endl
             << "Frames w/faces: " << object_listener.getFramesWithFaces() << endl
             << "Percent of frames w/faces: " << object_listener.getFramesWithFacesPercent() << "%" << endl
             << "******************************************************************" << endl;

        detector->reset();
        object_listener.reset();
    } while (program_options.loop);
}

void process_occupant_video(unique_ptr<vision::SyncFrameDetector>& detector, std::ofstream& csv_file_stream,
                            ProgramOptions& program_options) {

    // create the FrameDetector
    detector = std::unique_ptr<vision::SyncFrameDetector>(new vision::SyncFrameDetector(program_options.data_dir));

    // configure the FrameDetector by enabling features
    detector->enable({vision::Feature::OCCUPANTS});

    // prepare listeners
    PlottingOccupantListener occupant_listener(csv_file_stream, program_options.draw_display, !program_options
        .disable_logging, program_options.draw_id, 500, detector->getCabinRegionConfig().getRegions());
    StatusListener status_listener;

    // configure the FrameDetector by assigning listeners
    detector->setOccupantListener(&occupant_listener);
    detector->setProcessStatusListener(&status_listener);

    // start the detector
    detector->start();

    do {
        // the VideoReader will handle decoding frames from the input video file
        VideoReader video_reader(program_options.video_path, program_options.sampling_frame_rate);

        cv::Mat mat;
        Timestamp timestamp_ms;
        while (video_reader.GetFrame(mat, timestamp_ms)) {
            // create a Frame from the video input and process it with the FrameDetector
            vision::Frame
                f(mat.size().width, mat.size().height, mat.data, vision::Frame::ColorFormat::BGR, timestamp_ms);
            detector->process(f);
            occupant_listener.processResults();
        }

        cout << "******************************************************************" << endl
             << "Processed Frame count: " << occupant_listener.getProcessedFrames() << endl
             << "Frames w/faces: " << occupant_listener.getFramesWithFaces() << endl
             << "Percent of frames w/faces: " << occupant_listener.getFramesWithFacesPercent() << "%" << endl
             << "******************************************************************" << endl;

        detector->reset();
        occupant_listener.reset();
    } while (program_options.loop);
}

void process_face_video(unique_ptr<vision::SyncFrameDetector>& detector,
                        std::ofstream& csv_file_stream,
                        ProgramOptions program_options) {
    // create the FrameDetector
    detector = std::unique_ptr<vision::SyncFrameDetector>(new vision::SyncFrameDetector(program_options.data_dir,
                                                                                        program_options.num_faces));

    // configure the FrameDetector by enabling features
    detector->enable({vision::Feature::EMOTIONS, vision::Feature::EXPRESSIONS, vision::Feature::IDENTITY,
                      vision::Feature::APPEARANCES});

    // prepare listeners
    PlottingImageListener
        image_listener
        (csv_file_stream, program_options.draw_display, !program_options.disable_logging, program_options.draw_id);
    StatusListener status_listener;

    // configure the FrameDetector by assigning listeners
    detector->setImageListener(&image_listener);
    detector->setProcessStatusListener(&status_listener);

    // start the detector
    detector->start();

    do {
        // the VideoReader will handle decoding frames from the input video file
        VideoReader video_reader(program_options.video_path, program_options.sampling_frame_rate);

        cv::Mat mat;
        Timestamp timestamp_ms;
        while (video_reader.GetFrame(mat, timestamp_ms)) {
            // create a Frame from the video input and process it with the FrameDetector
            vision::Frame
                f(mat.size().width, mat.size().height, mat.data, vision::Frame::ColorFormat::BGR, timestamp_ms);
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
    } while (program_options.loop);
}

int main(int argsc, char** argsv) {

    //setting up output precision
    const int precision = 2;
    std::cerr << std::fixed << std::setprecision(precision);
    std::cout << std::fixed << std::setprecision(precision);

    //Gathering program options
    ProgramOptions program_options;
    po::options_description
        description("Project for demoing the Affectiva FrameDetector class (processing video files).");
    assemble_program_options(description, program_options);

    //verifying command line arguments
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

    //To set CSV file's suffix
    std::string detection_type = "";

    //Check for object or occupant argument present or not. If nothing is present then enable face by default.

    if (args.count("object") && args.count("occupant")) {
        std::cout << "Can't turn on Object and occupant detection" << std::endl;
        std::cerr << "ERROR: Can't use --occupant and --object at the same time" << std::endl << std::endl;
        std::cerr << "For help, use the -h option." << std::endl << std::endl;
        return 1;
    }
    else if (args.count("object")) {
        std::cout << "Setting up object detection" << std::endl;
        program_options.object_enabled = true;
        detection_type = "_objects";
    }
    else if (args.count("occupant")) {
        std::cout << "Setting up occupant detection" << std::endl;
        program_options.occupant_enabled = true;
        detection_type = "_occupants";
    }
    else {
        detection_type = "_faces";
        std::cout << "Setting up face detection" << std::endl;
    }

    // set data_dir to env_var if not set on cmd line
    program_options.data_dir = validatePath(program_options.data_dir, DATA_DIR_ENV_VAR);

    if (program_options.draw_id && !program_options.draw_display) {
        std::cerr << "Can't draw face id while drawing to screen is disabled" << std::endl;
        std::cerr << description << std::endl;
        return 1;
    }

    unique_ptr<vision::SyncFrameDetector> detector;
    try {

        //initialize the output file
        std::string csv_path_new = (program_options.video_path.substr(0, program_options.video_path.find(".", 0)));
        csv_path_new += detection_type + ".csv";
        boost::filesystem::path csv_path(csv_path_new);
        std::ofstream csv_file_stream(csv_path.c_str());

        if (!csv_file_stream.is_open()) {
            std::cerr << "Unable to open csv file " << csv_path << std::endl;
            return 1;
        }
        if (detection_type == "_faces") {
            process_face_video(detector, csv_file_stream, program_options);
        }
        else if (detection_type == "_objects") {
            process_object_video(detector, csv_file_stream, program_options);
        }
        else if (detection_type == "_occupants") {
            process_occupant_video(detector, csv_file_stream, program_options);
        }

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


