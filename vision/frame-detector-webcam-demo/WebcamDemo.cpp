#include "AFaceListener.h"
#include "PlottingImageListener.h"
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

static const std::string DISPLAY_DATA_DIR_ENV_VAR = "AFFECTIVA_VISION_DATA_DIR";
static const affdex::str DATA_DIR_ENV_VAR = STR(DISPLAY_DATA_DIR_ENV_VAR);

int main(int argsc, char ** argsv) {
    namespace po = boost::program_options; // abbreviate namespace

    std::cout << "Hit ESCAPE key to exit app.." << endl;

    try {

        const std::vector<int> DEFAULT_RESOLUTION {1280, 720};

        // cmd line options
        affdex::path data_dir;
        affdex::path output_file_path;
        std::vector<int> resolution;
        int process_framerate;
        int camera_framerate;
        int camera_id;
        unsigned int num_faces;
        bool draw_display = true;
        bool sync = false;
        bool draw_id = true;
        bool disable_logging = false;

        const int precision = 2;
        std::cerr << std::fixed << std::setprecision(precision);
        std::cout << std::fixed << std::setprecision(precision);

        po::options_description description("Project for demoing the Affdex SDK FrameDetector class (grabbing and processing frames from the camera).");
        description.add_options()
            ("help,h", po::bool_switch()->default_value(false), "Display this help message.")
#ifdef _WIN32
            ("data,d", po::wvalue<affdex::path>(&data_dir),
                std::string("Path to the data folder. Alternatively, specify the path via the environment variable "
                    + DISPLAY_DATA_DIR_ENV_VAR + R"(=\path\to\data)").c_str())
#else //  _WIN32
            ("data,d", po::value< affdex::path >(&data_dir),
                (std::string("Path to the data folder. Alternatively, specify the path via the environment variable ")
                + DATA_DIR_ENV_VAR + "=/path/to/data").c_str())
#endif // _WIN32
            ("resolution,r", po::value< std::vector<int> >(&resolution)->default_value(DEFAULT_RESOLUTION, "1280 720")->multitoken(), "Resolution in pixels (2-values): width height")
            ("pfps", po::value< int >(&process_framerate)->default_value(30), "Processing framerate.")
            ("cfps", po::value< int >(&camera_framerate)->default_value(30), "Camera capture framerate.")
            ("cid", po::value< int >(&camera_id)->default_value(0), "Camera ID.")
            ("numFaces", po::value< unsigned int >(&num_faces)->default_value(1), "Number of faces to be tracked.")
            ("draw", po::value< bool >(&draw_display)->default_value(true), "Draw metrics on screen.")
            ("sync", po::bool_switch(&sync)->default_value(false), "Process frames synchronously. Note this will process all frames captured by the camera and will ignore the value in --pfps")
            ("quiet,q", po::bool_switch(&disable_logging)->default_value(false), "Disable logging to console")
            ("face_id", po::value< bool >(&draw_id)->default_value(true), "Draw face id on screen. Note: Drawing to screen must be enabled.")
            ("file,f", po::value< affdex::path >(&output_file_path), "Name of the output CSV file.")
            ;

        po::variables_map args;
        try {
            po::store(po::command_line_parser(argsc, argsv).options(description).run(), args);
            if (args["help"].as<bool>())
            {
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

        if (resolution.size() != 2) {
            std::cerr << "Only two numbers must be specified for resolution." << std::endl;
            return 1;
        }

        if (resolution[0] <= 0 || resolution[1] <= 0) {
            std::cerr << "Resolutions must be positive numbers." << std::endl;
            return 1;
        }

        if (draw_id && !draw_display) {
            std::cerr << "Can't draw face id while drawing to screen is disabled" << std::endl;
            std::cerr << description << std::endl;
            return 1;
        }

        //initialize the output file
        boost::filesystem::path csv_path(output_file_path);
        csv_path.replace_extension(".csv");
        std::ofstream csv_file_stream(csv_path.c_str());

        if (!csv_file_stream.is_open()) {
            std::cerr << "Unable to open csv file " << output_file_path << std::endl;
            return 1;
        }

        // create the FrameDetector
        unique_ptr<vision::Detector> frame_detector;
        if (sync) {
            frame_detector = std::unique_ptr<vision::Detector>(new vision::SyncFrameDetector(data_dir, num_faces));
        }
        else {
            frame_detector = std::unique_ptr<vision::Detector>(new vision::FrameDetector(data_dir, process_framerate, num_faces));
        }

        // prepare listeners
        //std::ofstream csvFileStream;
        PlottingImageListener image_listener(csv_file_stream, draw_display, !disable_logging, draw_id);
        AFaceListener face_listener;
        StatusListener status_listener;


        // configure the FrameDetector by enabling features and assigning listeners
        frame_detector->enable({ vision::Feature::EMOTIONS, vision::Feature::EXPRESSIONS, vision::Feature::IDENTITY, vision::Feature::APPEARANCES});
        frame_detector->setImageListener(&image_listener);
        frame_detector->setFaceListener(&face_listener);
        frame_detector->setProcessStatusListener(&status_listener);

        // Connect to the webcam and configure it
        cv::VideoCapture webcam(camera_id);

        // Note: not all webcams support these configuration properties
        webcam.set(CV_CAP_PROP_FPS, camera_framerate);
        webcam.set(CV_CAP_PROP_FRAME_WIDTH, resolution[0]);
        webcam.set(CV_CAP_PROP_FRAME_HEIGHT, resolution[1]);

        const auto start_time = std::chrono::system_clock::now();
        if (!webcam.isOpened()) {
            std::cerr << "Error opening webcam" << std::endl;
            return 1;
        }

        //Start the frame detector thread.
        frame_detector->start();

        do {
            cv::Mat img;
            if (!webcam.read(img)) {   //Capture an image from the camera
                std::cerr << "Failed to read frame from webcam" << std::endl;
                break;
            }

            timestamp ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count();

            // Create a Frame from the webcam image and process it with the FrameDetector
            const vision::Frame f(img.size().width, img.size().height, img.data, vision::Frame::ColorFormat::BGR, ts);
            if (sync) {
                dynamic_cast<vision::SyncFrameDetector *>(frame_detector.get())->process(f);
            }
            else {
                dynamic_cast<vision::FrameDetector *>(frame_detector.get())->process(f);
            }

            image_listener.processResults();
        }

#ifdef _WIN32
        while (!GetAsyncKeyState(VK_ESCAPE) && status_listener.isRunning());
#else //  _WIN32
        while (status_listener.isRunning() && (cv::waitKey(20) != 27)); // ascii for ESC
#endif
        frame_detector->stop();
        csv_file_stream.close();

        if (boost::filesystem::exists(output_file_path) ) {
            std::cout << "Output written to file" << output_file_path << std::endl;
        }
    }
    catch (...) {
        std::cerr << "Encountered an exception " << std::endl;
        return 1;
    }

    return 0;
}
