
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
    PlottingListener(std::ofstream& csv, bool draw_display, bool enable_logging) :
        out_stream_(csv),
        image_data_(),
        start_(std::chrono::system_clock::now()),
        process_last_ts_(0),
        draw_display_(draw_display),
        processed_frames_(0),
        logging_enabled_(enable_logging) {
    }

    int getDataSize() {
        std::lock_guard<std::mutex> lg(mtx);
        return results_.size();
    }

    unsigned int getProcessedFrames() {
        return processed_frames_;
    }

    std::pair<vision::Frame, std::map<vision::Id, T>> getData() {
        std::lock_guard<std::mutex> lg(mtx);
        std::pair<vision::Frame, std::map<vision::Id, T>> dpoint = results_.front();
        results_.pop_front();
        return dpoint;
    }

    //Needed to get Image data to create output video
    cv::Mat getImageData() { return image_data_; }

    virtual void outputToFile(const std::map<vision::Id, T>& id_type_map, double time_stamp) = 0;

    virtual void draw(const std::map<vision::Id, T>& id_type_map, const vision::Frame& image) = 0;

    virtual void processResults() = 0;

    virtual void reset() = 0;

protected:
    using frame_type_id_pair = std::pair<vision::Frame, std::map<vision::Id, T>>;
    std::ofstream& out_stream_;
    Visualizer viz_;
    cv::Mat image_data_;

    std::chrono::time_point<std::chrono::system_clock> start_;
    std::mutex mtx;

    std::deque<frame_type_id_pair> results_;
    Timestamp process_last_ts_;
    bool draw_display_;
    unsigned int processed_frames_;
    bool logging_enabled_;
};
