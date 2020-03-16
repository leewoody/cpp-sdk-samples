#pragma once

#include <FaceListener.h>
#include <iostream>

using namespace affdex;

class AFaceListener : public vision::FaceListener {
    void onFaceFound(affdex::Timestamp timestamp, vision::FaceId faceId) {
        std::cout << "Face id " << faceId << " found at timestamp " << timestamp << std::endl;
    }

    void onFaceLost(affdex::Timestamp timestamp, vision::FaceId faceId) {
        std::cout << "Face id " << faceId << " lost at timestamp " << timestamp << std::endl;
    }
};
