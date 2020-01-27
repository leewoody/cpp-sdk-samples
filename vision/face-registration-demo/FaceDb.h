#pragma once

#include "FaceRegistrar.h"
#include <Core.h>

class FaceDb {
  public:
    FaceDb(const std::string data_dir, const bool preview);

    void videoRegister(const int identifier,
                       affdex::path filename,
                       const int frame_span);
    void webcamRegister(const int identifier,
                        const int camera_id);

    void list() const;
    void unregister(const int identifier) const;
    void unregisterAll() const;

private:
    std::string registeredIds() const;
    void showResult(const affdex::vision::FaceRegistrationResult result) const;
    affdex::vision::FaceRegistrationResult processFrame(const cv::Mat &frame, const int identifier);

    affdex::path data_dir_;
    mutable std::unique_ptr<affdex::vision::FaceRegistrar> face_registrar_;
    affdex::vision::FaceRegistrar &faceRegistrar() const;

    static void interrupt(int sig);
    static bool exitLoop_;

    const bool preview_;
    std::string title_;
};
