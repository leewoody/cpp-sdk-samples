#include "Application.h"
#include "FileUtils.h"
#include "FaceDb.h"

#include <iostream>
#include <signal.h>

using namespace affdex;
using namespace std;

namespace po = boost::program_options;

namespace {
    const std::string DISPLAY_DATA_DIR_ENV_VAR = "AFFECTIVA_VISION_DATA_DIR";
    const affdex::str DATA_DIR_ENV_VAR = STR(DISPLAY_DATA_DIR_ENV_VAR);
}

Application::Application(int argsc, char **argsv) : command_line_(argsc, argsv) {}

Application::~Application() {}

void Application::run() {
    const string webcam_label = "webcam-register";
    const string video_label = "video-register";
    const string list_label = "list";
    const string unreg_label  = "unregister";
    const string unreg_all_label = "unregister-all";

    // Parse command line and invoke appropriate function
    if (command_line_.cmd_ == video_label) {
        setCommonOptions(" " + video_label + " <identifier> <video file>\n\n"
                         "Registers a face found in the <video> file with the provided <identifier>");

        unsigned int frame_sampling_rate = 0;
        options_->add_options()("sample_rate,s", po::value<unsigned int>(&frame_sampling_rate),
                                "sample at most N frames per second (0=all)");

        if (parse(2)) {
            faceDb_->videoRegister(atoi(command_line_.required_[0].c_str()), command_line_.required_[1],
                          frame_sampling_rate);
        }

    } else if (command_line_.cmd_ == webcam_label) {
        setCommonOptions(" " + webcam_label + " <identifier>\n\n"
                         "Registers a face found using webcam with the provided <identifier>");

        int camera_id = 0;
        options_->add_options()("cid,c", po::value<int>(&camera_id),
                                "Camera ID");

        if (parse(1)) {
            faceDb_->webcamRegister(atoi(command_line_.required_[0].c_str()), camera_id);
        }

    } else if (command_line_.cmd_ == list_label) {
        setCommonOptions(" " + list_label + "\n\n"
                         "Lists ids for all registered faces");
        if (parse(0)) {
            faceDb_->list();
        }

    } else if (command_line_.cmd_ == unreg_label) {
        setCommonOptions(" " + unreg_label + " <identifier>\n\n"
                         "Removes registration information from database for provided <identifier>");
        if (parse(1)) {
            faceDb_->unregister(atoi(command_line_.required_[0].c_str()));
        }

    } else if (command_line_.cmd_ == unreg_all_label) {
        setCommonOptions(" " + unreg_all_label + "\n\n"
                         "Removes all registration information from the database");
        if (parse(0)) {
            faceDb_->unregisterAll();
        }

    } else {
        cout << "Application for demoing the Affectiva face registration\n"
             << "\n"
             << "usage " + command_line_.app_name_ + " <"
             << video_label << "|"
             << webcam_label << "|"
             << list_label << "|"
             << unreg_label << "|"
             << unreg_all_label << "> ... " << endl;
        return;
    }
}

void Application::setCommonOptions(const string &usage) {
    options_.reset(new po::options_description("usage: " + command_line_.app_name_
                                               + usage+ " [options]\n\nOptions:"));

    options_->add_options()("help,h", po::bool_switch()->default_value(false), "Display this help message");
#ifdef _WIN32
    options_->add_options()(
        "data,d", po::wvalue<path>(&data_dir_),
        (string("Path to the data folder. Alternatively, specify the path via the environment variable ") +
         DISPLAY_DATA_DIR_ENV_VAR + "=\path\to\data")
        .c_str());
#else // _WIN32
    options_->add_options()(
        "data,d", po::value<path>(&data_dir_),
        (string("Path to the data folder. Alternatively, specify the path via the environment variable ") +
         DATA_DIR_ENV_VAR + "=/path/to/data")
        .c_str());
#endif
    options_->add_options()
        ("preview,p", po::bool_switch(&preview_)->default_value(false),
         "Show preview window");

}

bool Application::parse(const int required_param_count){
    bool successful = command_line_.process(*options_, required_param_count);
    if (successful){
        data_dir_ = validatePath(data_dir_, DATA_DIR_ENV_VAR);
        faceDb_.reset(new FaceDb(data_dir_, preview_));
    }
    return successful;
}
