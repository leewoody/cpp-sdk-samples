
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
    PlottingListener(std::ofstream& csv, bool write_video, bool enable_logging) :
        out_stream_(csv),
        image_data_(),
        start_(std::chrono::system_clock::now()),
        process_last_ts_(0),
        write_video_(write_video),
        processed_frames_(0),
        logging_enabled_(enable_logging) {
    }

    int getDataSize() {
        std::lock_guard<std::mutex> lg(mtx);
        return results_.size();
    }

    int getProcessedFrames() {
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

    void drawRecentFrame() {
        if (write_video_) {
            if (most_recent_frame_.getTimestamp() - time_callback_received_ <= timeout_) {
                draw(latest_data_.second, most_recent_frame_);
            }
            else {
                draw({}, most_recent_frame_);
            }
        }
    }

    void processResults(const vision::Frame& frame) {
        most_recent_frame_ = frame;
        if (getDataSize() > 0) {
            time_callback_received_ = most_recent_frame_.getTimestamp();
            processResults();
        }
        else {
            drawRecentFrame();
        }
    }

protected:
    using frame_type_id_pair = std::pair<vision::Frame, std::map<vision::Id, T>>;
    std::ofstream& out_stream_;
    Visualizer viz_;
    cv::Mat image_data_;

    std::chrono::time_point<std::chrono::system_clock> start_;
    std::mutex mtx;

    std::deque<frame_type_id_pair> results_;
    Timestamp process_last_ts_;
    bool write_video_;
    int processed_frames_;
    bool logging_enabled_;
    frame_type_id_pair latest_data_;
    vision::Frame most_recent_frame_;
    Timestamp time_callback_received_ = 0;
    Duration timeout_ = 500;
};
