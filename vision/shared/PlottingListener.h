
#pragma once

#include "Visualizer.h"
#include "Frame.h"

#include <deque>
#include <mutex>
#include <fstream>
#include <condition_variable>

#include <iostream>
#include <iomanip>

using namespace affdex;

template<typename T> class PlottingListener {

public:
    PlottingListener(std::ofstream& csv, bool draw_display, bool enable_logging, bool draw_face_id) :
        draw_display(draw_display),
        process_last_ts(0),
        process_fps(0),
        out_stream(csv),
        start(std::chrono::system_clock::now()),
        processed_frames(0),
        frames_with_faces(0),
        draw_face_id(draw_face_id),
        logging_enabled(enable_logging) {
    }

    unsigned int getProcessingFrameRate() {
        std::lock_guard<std::mutex> lg(mtx);
        return process_fps;
    }

    int getDataSize() {
        std::lock_guard<std::mutex> lg(mtx);
        return results.size();
    }

    unsigned int getProcessedFrames() {
        return processed_frames;
    }

    unsigned int getFramesWithFaces() {
        return frames_with_faces;
    }

    double getFramesWithFacesPercent() {
        return (static_cast<double>(frames_with_faces) / processed_frames) * 100;
    }

    std::pair<vision::Frame, std::map<vision::Id, T>> getData() {
        std::lock_guard<std::mutex> lg(mtx);
        std::pair<vision::Frame, std::map<vision::Id, T>> dpoint = results.front();
        results.pop_front();
        return dpoint;
    }

    virtual void outputToFile(const std::map<vision::Id, T>& id_type_map, double time_stamp) = 0;

    virtual void draw(const std::map<vision::Id, T>& id_type_map, const vision::Frame& image) = 0;

    virtual void processResults() = 0;

    virtual void reset() = 0;

protected:
    using frame_type_id_pair = std::pair<vision::Frame, std::map<vision::Id, T>>;
    bool draw_display;
    std::mutex mtx;
    std::deque<frame_type_id_pair> results;

    Timestamp process_last_ts;
    unsigned int process_fps;
    std::ofstream& out_stream;
    std::chrono::time_point<std::chrono::system_clock> start;

    Visualizer viz;

    unsigned int processed_frames;
    unsigned int frames_with_faces;
    bool draw_face_id;
    bool logging_enabled;
};
