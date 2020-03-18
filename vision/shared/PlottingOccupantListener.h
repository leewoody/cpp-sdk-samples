#pragma once

#include "OccupantListener.h"
#include "PlottingListener.h"

using namespace affdex;

class PlottingOccupantListener : public vision::OccupantListener, public PlottingListener<vision::Occupant> {

public:

    PlottingOccupantListener(std::ofstream& csv, bool draw_display, bool enable_logging, bool draw_face_id, const
    Duration callback_interval, std::vector<CabinRegion> cabin_regions) :
        PlottingListener(csv, draw_display, enable_logging, draw_face_id), callback_interval_(callback_interval),
        cabin_regions_(cabin_regions) {
        out_stream << "TimeStamp, occupantId, confidence, regionId,  upperLeftX, upperLeftY, lowerRightX, lowerRightY";

        for (const auto& cr :cabin_regions_) {
            out_stream << "," << "Region " << cr.id;
        }

        out_stream << std::endl;
        out_stream.precision(2);
        out_stream << std::fixed;
    }

    Duration getCallbackInterval() const override {
        return callback_interval_;
    }

    void onOccupantResults(const std::map<vision::OccupantId, vision::Occupant>& occupants,
                           vision::Frame frame) override {
        std::lock_guard<std::mutex> lg(mtx);
        results.emplace_back(frame, occupants);
        process_fps = 1000.0f / (frame.getTimestamp() - process_last_ts);
        process_last_ts = frame.getTimestamp();

        processed_frames++;
        if (occupants.size() > 0) {
            frames_with_faces++;
        }
    };

    void outputToFile(const std::map<vision::OccupantId, vision::Occupant>& occupants, double time_stamp) override {
        if (occupants.empty()) {
            // TimeStamp occupantId confidence regionId upperLeftX upperLeftY lowerRightX lowerRightY"
            out_stream << time_stamp << ",nan,nan,nan,nan,nan,nan,nan";
            for (const auto& cr :cabin_regions_) {
                out_stream << ",nan";
            }
            out_stream << std::endl;
        }

        for (const auto& occupant_id_pair : occupants) {
            vision::Occupant occup = occupant_id_pair.second;
            std::vector<vision::Point> bbox({occup.boundingBox.getTopLeft(), occup.boundingBox.getBottomRight()});

            out_stream << time_stamp << ","
                       << occupant_id_pair.first << ","
                       << occup.matchedSeat.matchConfidence << "," << occup.matchedSeat.cabinRegion.id << ","
                       << std::setprecision(0) << bbox[0].x << "," << bbox[0].y << "," << bbox[1].x << "," << bbox[1].y
                       << std::setprecision(4);

            for (const auto& cr :cabin_regions_) {
                if (cr.id == occup.matchedSeat.cabinRegion.id) {
                    out_stream << "," << occup.matchedSeat.matchConfidence;
                }
                else {
                    out_stream << "," << 0;
                }
            }

            out_stream << std::endl;
        }
    }

    void draw(const std::map<vision::OccupantId, vision::Occupant>& occupants, const vision::Frame& image) override {
        std::shared_ptr<unsigned char> imgdata = image.getBGRByteArray();
        const cv::Mat img = cv::Mat(image.getHeight(), image.getWidth(), CV_8UC3, imgdata.get());
        viz.updateImage(img);

        for (const auto& face_id_pair : occupants) {
            const vision::Occupant occup = face_id_pair.second;
            viz.drawOccupantMetrics(occup);
        }
        viz.showImage();
        image_data = viz.getImageData();
    }

    void processResults() override {
        while (getDataSize() > 0) {
            const std::pair<vision::Frame, std::map<vision::OccupantId, vision::Occupant>> dataPoint = getData();
            vision::Frame frame = dataPoint.first;
            const std::map<vision::OccupantId, vision::Occupant> occupants = dataPoint.second;

            if (draw_display) {
                draw(occupants, frame);
            }

            outputToFile(occupants, frame.getTimestamp());

            if (logging_enabled) {
                std::cout << "timestamp: " << frame.getTimestamp()
                          << " pfps: " << getProcessingFrameRate()
                          << " occupants: " << occupants.size() << std::endl;
            }
        }
    }

    void reset() override {
        std::lock_guard<std::mutex> lg(mtx);
        process_last_ts = 0;
        process_fps = 0;
        start = std::chrono::system_clock::now();
        processed_frames = 0;
        frames_with_faces = 0;
        results.clear();
    }

private:
    Duration callback_interval_;
    std::vector<CabinRegion> cabin_regions_;
};
