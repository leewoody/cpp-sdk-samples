#include "FaceDb.h"
#include "VideoReader.h"
#include "FileUtils.h"

#include <iostream>
#include <iomanip>
#include <signal.h>

using namespace affdex;
using namespace std;

bool FaceDb::exitLoop_ = false;

void FaceDb::interrupt(int sig) {
    cout << "EXITING" << endl;
    exitLoop_ = true;
}

FaceDb::FaceDb(const string data_dir, const bool preview)
    : data_dir_(data_dir), preview_(preview) {}

affdex::vision::FaceRegistrar& FaceDb::faceRegistrar() const {
    if (!face_registrar_) {
        face_registrar_ = unique_ptr<affdex::vision::FaceRegistrar>(new affdex::vision::FaceRegistrar(data_dir_));
        signal(SIGINT, FaceDb::interrupt);
    }
    return *face_registrar_;
}

string FaceDb::registeredIds() const {
    vector<identity> IDs = faceRegistrar().getIdentities();

    string separator;
    string output;
    for (auto id : IDs) {
        output += separator + std::to_string(id);
        separator = " + ";
    }
    return output;
}

void FaceDb::showResult(const vision::FaceRegistrationResult result) const {
    string hint_text = "Hint:      ";
    if (result.orientation_hints.size() > 0) {
        for (auto hint : result.orientation_hints) {
            string item = " ";
            switch (hint) {
                case vision::FaceOrientation::CENTER: {
                    item = "c";
                    break;
                }
                case vision::FaceOrientation::UP: {
                    item = "u";
                    break;
                }
                case vision::FaceOrientation::DOWN: {
                    item = "d";
                    break;
                }
                case vision::FaceOrientation::LEFT: {
                    item = "l";
                    break;
                }
                case vision::FaceOrientation::RIGHT: {
                    item = "r";
                    break;
                }
            }
            hint_text.replace(static_cast<int>(hint) + 6, 1, item);
        }
    }
    cout << "\rScore=" << setw(3) << result.score << " " << hint_text;
}

vision::FaceRegistrationResult FaceDb::processFrame(const cv::Mat& frame, const int identifier) {
    vision::Frame image(frame.size().width, frame.size().height, frame.data, vision::Frame::ColorFormat::BGR);

    // We use the bounding box for the entire video frame here.  If there are multiple faces in the frame,
    // it is undefined which face is chosen.  Choosing a smaller bounding box for a known location
    // will restrict the faces chosen to this region.
    vision::BoundingBox frame_bb =
        vision::BoundingBox(vision::Point(0, 0), vision::Point(image.getWidth(), image.getHeight()));

    vision::FaceRegistrationResult result = faceRegistrar().registerFace(identifier, image, frame_bb);
    showResult(result);
    if (result.face_found) {
        if (preview_) {
            imshow(title_, frame);
            cv::waitKey(30);
        }
    };

    return result;
}

void FaceDb::videoRegister(const int identifier, path filename,
                           const int frame_sampling_rate) {
    filename = validatePath(filename);
    title_ = filename;
    VideoReader video_reader(filename, frame_sampling_rate);

    cout << endl << "App will exit when end of video is reached, Reg score reaches 100, or by pressing ctrl-C." << endl
         << endl;

    vision::FaceRegistrationResult result;
    result.score = 0;
    showResult(result);

    cv::Mat frame;
    timestamp timestamp_ms;
    // Iterate over frames of video and attempt to register a face in it.
    while (!exitLoop_ && result.score < 100 && video_reader.GetFrame(frame, timestamp_ms)) {
        result = processFrame(frame, identifier);
    }
    cout << endl << endl;
}

void FaceDb::webcamRegister(const int identifier,
                            const int camera_id) {
    title_ = "Live";

    // Connect to the webcam
    cv::VideoCapture webcam;
    int apiID = CV_CAP_ANY;
    webcam.open(camera_id + apiID);
    if (!webcam.isOpened()) {
        throw runtime_error("Error opening webcam");
    }

    cout << endl << "App will exit when Reg score reaches 100, or by pressing ctrl-C." << endl << endl;

    vision::FaceRegistrationResult result;
    result.score = 0;
    showResult(result);

    cv::Mat frame;
    // Iterate over frames of video and attempt to register a face in it.
    while (!exitLoop_ && result.score < 100) {
        if (!webcam.read(frame)) {
            throw runtime_error("Failed to read frame from webcam");
        }

        result = processFrame(frame, identifier);
    }
    cout << endl << endl;
}

void FaceDb::list() const {
    cout << "Registered IDs: " << registeredIds() << endl;
}

void FaceDb::unregister(const int identifier) const {
    faceRegistrar().unregister(identifier);
    cout << "Unregistered ID: " << identifier << endl;
}

void FaceDb::unregisterAll() const {
    cout << "IDs to unregister: " << registeredIds() << endl;;
    faceRegistrar().unregisterAll();
    cout << "Successfully unregistered all IDs." << endl;
}
