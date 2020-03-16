#include "AFaceListener.h"
#include "PlottingImageListener.h"
#include "PlottingObjectListener.h"
#include "PlottingOccupantListener.h"
#include "StatusListener.h"
#include "FileUtils.h"

#include <Core.h>
#include <FrameDetector.h>
#include <SyncFrameDetector.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace affdex;
namespace po = boost::program_options; // abbreviate namespace


static const std::string DISPLAY_DATA_DIR_ENV_VAR = "AFFECTIVA_VISION_DATA_DIR";
static const affdex::Str DATA_DIR_ENV_VAR = STR(DISPLAY_DATA_DIR_ENV_VAR);

struct ProgramOptions {
    // cmd line args
    affdex::Path data_dir;
    affdex::Path output_file_path;
    std::vector<int> resolution;
    int process_framerate;
    int camera_framerate;
    int camera_id;
    unsigned int num_faces;
    bool draw_display = true;
    bool sync = false;
    bool draw_id = true;
    bool disable_logging = false;
    bool object_enabled = false;
    bool occupant_enabled = false;
};

void assemble_program_options(po::options_description& description, ProgramOptions& program_options) {
    const std::vector<int> DEFAULT_RESOLUTION{1280, 720};

    description.add_options()
        ("help,h", po::bool_switch()->default_value(false), "Display this help message.")
#ifdef _WIN32
    ("data,d", po::wvalue<affdex::Path>(&data_dir),
            std::string("Path to the data folder. Alternatively, specify the path via the environment variable "
                + DISPLAY_DATA_DIR_ENV_VAR + R"(=\path\to\data)").c_str())
#else //  _WIN32
        ("data,d", po::value<affdex::Path>(&program_options.data_dir),
         (std::string("Path to the data folder. Alternatively, specify the path via the environment variable ")
             + DATA_DIR_ENV_VAR + "=/path/to/data").c_str())
#endif // _WIN32
        ("resolution,r",
         po::value<std::vector<int>
         >(&program_options.resolution)->default_value(DEFAULT_RESOLUTION, "1280 720")->multitoken(),
         "Resolution in pixels (2-values): width height")
        ("pfps", po::value<int>(&program_options.process_framerate)->default_value(30), "Processing framerate.")
        ("cfps", po::value<int>(&program_options.camera_framerate)->default_value(30), "Camera capture framerate.")
        ("cid", po::value<int>(&program_options.camera_id)->default_value(0), "Camera ID.")
        ("numFaces",
         po::value<unsigned int>(&program_options.num_faces)->default_value(1),
         "Number of faces to be tracked.")
        ("draw", po::value<bool>(&program_options.draw_display)->default_value(true), "Draw metrics on screen.")
        ("sync",
         po::bool_switch(&program_options.sync)->default_value(false),
         "Process frames synchronously. Note this will process all frames captured by the camera and will ignore the value in --pfps")
        ("quiet,q",
         po::bool_switch(&program_options.disable_logging)->default_value(false),
         "Disable logging to console")
        ("face_id",
         po::value<bool>(&program_options.draw_id)->default_value(true),
         "Draw face id on screen. Note: Drawing to screen must be enabled.")
        ("file,f", po::value<affdex::Path>(&program_options.output_file_path), "Name of the output CSV file.")
        ("object", "Enable object detection")
        ("occupant", "Enable occupant detection");
}

void process_face_stream(unique_ptr<vision::Detector>& frame_detector, std::ofstream& csv_file_stream,
                         ProgramOptions program_options, StatusListener& status_listener, cv::VideoCapture& webcam) {

    // prepare listeners
    //std::ofstream csvFileStream;
    PlottingImageListener
        image_listener
        (csv_file_stream, program_options.draw_display, !program_options.disable_logging, program_options.draw_id);
    AFaceListener face_listener;


    // configure the FrameDetector by enabling features and assigning listeners
    frame_detector->enable({vision::Feature::EMOTIONS, vision::Feature::EXPRESSIONS, vision::Feature::IDENTITY,
                            vision::Feature::APPEARANCES});
    frame_detector->setImageListener(&image_listener);
    frame_detector->setFaceListener(&face_listener);
    frame_detector->setProcessStatusListener(&status_listener);
    const auto start_time = std::chrono::system_clock::now();

    //Start the frame detector thread.
    frame_detector->start();

    do {
        cv::Mat img;
        if (!webcam.read(img)) {   //Capture an image from the camera
            std::cerr << "Failed to read frame from webcam" << std::endl;
            break;
        }

        Timestamp ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - start_time).count();

        // Create a Frame from the webcam image and process it with the FrameDetector
        const vision::Frame f(img.size().width, img.size().height, img.data, vision::Frame::ColorFormat::BGR, ts);
        if (sync) {
            dynamic_cast<vision::SyncFrameDetector*>(frame_detector.get())->process(f);
        }
        else {
            dynamic_cast<vision::FrameDetector*>(frame_detector.get())->process(f);
        }

        image_listener.processResults();
    }
#ifdef _WIN32
        while (!GetAsyncKeyState(VK_ESCAPE) && status_listener.isRunning());
#else //  _WIN32
    while (status_listener.isRunning() && (cv::waitKey(20) != 27)); // ascii for ESC
#endif
}

void process_object_stream(unique_ptr<vision::Detector>& frame_detector, std::ofstream& csv_file_stream,
                           ProgramOptions program_options, StatusListener& status_listener, cv::VideoCapture& webcam) {

    // prepare listeners
    PlottingObjectListener object_listener(csv_file_stream,
                                           program_options.draw_display,
                                           !program_options.disable_logging,
                                           program_options.draw_id,
                                           {{vision::Feature::CHILD_SEATS, 1000}, {vision::Feature::PHONES, 1000}},
                                           frame_detector->getCabinRegionConfig().getRegions());


    // configure the FrameDetector by enabling features and assigning listeners
    frame_detector->enable({vision::Feature::CHILD_SEATS, vision::Feature::PHONES});
    frame_detector->setObjectListener(&object_listener);
    frame_detector->setProcessStatusListener(&status_listener);

    const auto start_time = std::chrono::system_clock::now();

    //Start the frame detector thread.
    frame_detector->start();

    do {
        cv::Mat img;
        if (!webcam.read(img)) {   //Capture an image from the camera
            std::cerr << "Failed to read frame from webcam" << std::endl;
            break;
        }

        Timestamp ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - start_time).count();

        // Create a Frame from the webcam image and process it with the FrameDetector
        const vision::Frame f(img.size().width, img.size().height, img.data, vision::Frame::ColorFormat::BGR, ts);
        if (sync) {
            dynamic_cast<vision::SyncFrameDetector*>(frame_detector.get())->process(f);
        }
        else {
            dynamic_cast<vision::FrameDetector*>(frame_detector.get())->process(f);
        }

        object_listener.processResults();
    }
#ifdef _WIN32
        while (!GetAsyncKeyState(VK_ESCAPE) && status_listener.isRunning());
#else //  _WIN32
    while (status_listener.isRunning() && (cv::waitKey(20) != 27)); // ascii for ESC
#endif
}

void process_occupant_stream(unique_ptr<vision::Detector>& frame_detector,
                             std::ofstream& csv_file_stream,
                             ProgramOptions program_options,
                             StatusListener& status_listener,
                             cv::VideoCapture& webcam) {

    // prepare listeners
    PlottingOccupantListener occupant_listener(csv_file_stream, program_options.draw_display, !program_options
        .disable_logging, program_options.draw_id, 500, frame_detector->getCabinRegionConfig().getRegions());


    // configure the FrameDetector by enabling features and assigning listeners
    frame_detector->enable({vision::Feature::CHILD_SEATS, vision::Feature::PHONES});
    frame_detector->setOccupantListener(&occupant_listener);
    frame_detector->setProcessStatusListener(&status_listener);

    const auto start_time = std::chrono::system_clock::now();

    //Start the frame detector thread.
    frame_detector->start();

    do {
        cv::Mat img;
        if (!webcam.read(img)) {   //Capture an image from the camera
            std::cerr << "Failed to read frame from webcam" << std::endl;
            break;
        }

        Timestamp ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - start_time).count();

        // Create a Frame from the webcam image and process it with the FrameDetector
        const vision::Frame f(img.size().width, img.size().height, img.data, vision::Frame::ColorFormat::BGR, ts);
        if (sync) {
            dynamic_cast<vision::SyncFrameDetector*>(frame_detector.get())->process(f);
        }
        else {
            dynamic_cast<vision::FrameDetector*>(frame_detector.get())->process(f);
        }

        occupant_listener.processResults();
    }
#ifdef _WIN32
        while (!GetAsyncKeyState(VK_ESCAPE) && status_listener.isRunning());
#else //  _WIN32
    while (status_listener.isRunning() && (cv::waitKey(20) != 27)); // ascii for ESC
#endif
}

int main(int argsc, char** argsv) {

    std::cout << "Hit ESCAPE key to exit app.." << endl;

    try {
        //setting up output precision
        const int precision = 2;
        std::cerr << std::fixed << std::setprecision(precision);
        std::cout << std::fixed << std::setprecision(precision);

        //Gathering program options
        ProgramOptions program_options;

        po::options_description description
            ("Project for demoing the Affdex SDK FrameDetector class (grabbing and processing frames from the camera).");
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

        //Check for object or occupant argument present or not. If nothing is present then enable face by default.
        {
            if (args.count("object") && args.count("occupant")) {
                std::cout << "Can't turn on Object and occupant detection" << std::endl;
                std::cerr << "ERROR: Can't use --occupant and --object at the same time" << std::endl << std::endl;
                std::cerr << "For help, use the -h option." << std::endl << std::endl;
                return 1;
            }
            else if (args.count("object")) {
                std::cout << "Setting up object detection" << std::endl;
                program_options.object_enabled = true;
            }
            else if (args.count("occupant")) {
                std::cout << "Setting up occupant detection" << std::endl;
                program_options.occupant_enabled = true;
            }
            else {
                std::cout << "Setting up face detection" << std::endl;
            }
        }

        // set data_dir to env_var if not set on cmd line
        program_options.data_dir = validatePath(program_options.data_dir, DATA_DIR_ENV_VAR);

        if (program_options.resolution.size() != 2) {
            std::cerr << "Only two numbers must be specified for resolution." << std::endl;
            return 1;
        }

        if (program_options.resolution[0] <= 0 || program_options.resolution[1] <= 0) {
            std::cerr << "Resolutions must be positive numbers." << std::endl;
            return 1;
        }

        if (program_options.draw_id && !program_options.draw_display) {
            std::cerr << "Can't draw face id while drawing to screen is disabled" << std::endl;
            std::cerr << description << std::endl;
            return 1;
        }

        //initialize the output file
        boost::filesystem::path csv_path(program_options.output_file_path);
        csv_path.replace_extension(".csv");
        std::ofstream csv_file_stream(csv_path.c_str());

        if (!csv_file_stream.is_open()) {
            std::cerr << "Unable to open csv file " << program_options.output_file_path << std::endl;
            return 1;
        }

        // create the FrameDetector
        unique_ptr<vision::Detector> frame_detector;
        if (sync) {
            frame_detector = std::unique_ptr<vision::Detector>(new vision::SyncFrameDetector(program_options.data_dir,
                                                                                             program_options.num_faces));
        }
        else {
            frame_detector =
                std::unique_ptr<vision::Detector>(new vision::FrameDetector(program_options.data_dir,
                                                                            program_options.process_framerate,
                                                                            program_options.num_faces));
        }
        // prepare listeners
        StatusListener status_listener;

        // Connect to the webcam and configure it
        cv::VideoCapture web_cam(program_options.camera_id);

        // Note: not all webcams support these configuration properties
        web_cam.set(CV_CAP_PROP_FPS, program_options.camera_framerate);
        web_cam.set(CV_CAP_PROP_FRAME_WIDTH, program_options.resolution[0]);
        web_cam.set(CV_CAP_PROP_FRAME_HEIGHT, program_options.resolution[1]);

        const auto start_time = std::chrono::system_clock::now();
        if (!web_cam.isOpened()) {
            std::cerr << "Error opening webcam" << std::endl;
            return 1;
        }

        if (!program_options.occupant_enabled && !program_options.object_enabled) {
            process_face_stream(frame_detector, csv_file_stream, program_options, status_listener, web_cam);
        }
        else if (program_options.object_enabled) {
            process_object_stream(frame_detector, csv_file_stream, program_options, status_listener, web_cam);
        }
        else if (program_options.occupant_enabled) {
            process_occupant_stream(frame_detector, csv_file_stream, program_options, status_listener, web_cam);
        }
        else {
            std::cerr << "This should never happen " << csv_path << std::endl;
            return 1;
        }

        frame_detector->stop();
        csv_file_stream.close();

        if (boost::filesystem::exists(program_options.output_file_path)) {
            std::cout << "Output written to file" << program_options.output_file_path << std::endl;
        }
    }
    catch (...) {
        std::cerr << "Encountered an exception " << std::endl;
        return 1;
    }

    return 0;
}
