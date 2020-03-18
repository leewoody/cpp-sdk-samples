#pragma once

#include "ObjectListener.h"
#include "PlottingListener.h"

using namespace affdex::vision;

class PlottingObjectListener : public ObjectListener, public PlottingListener<Object> {

public:

    PlottingObjectListener(std::ofstream& csv, bool draw_display, bool enable_logging, bool draw_face_id, const
    std::map<Feature, Duration> callback_intervals, std::vector<CabinRegion> cabin_regions) :
        PlottingListener(csv, draw_display, enable_logging, draw_face_id), callback_intervals_(callback_intervals),
        cabin_regions_(cabin_regions) {
        out_stream << "TimeStamp, objectId, confidence, upperLeftX, upperLeftY, lowerRightX, lowerRightY, ObjectType";

        for (const auto& cr :cabin_regions_) {
            out_stream << "," << "Region " << cr.id;
        }

        out_stream << std::endl;
        out_stream.precision(2);
        out_stream << std::fixed;
    }

    std::map<Feature, Duration> getCallbackIntervals() const override {
        return callback_intervals_;
    }

    void onObjectResults(const std::map<ObjectId, Object>& objects, vision::Frame frame) override {
        std::lock_guard<std::mutex> lg(mtx);
        results.emplace_back(frame, objects);
        process_fps = 1000.0f / (frame.getTimestamp() - process_last_ts);
        process_last_ts = frame.getTimestamp();

        processed_frames++;
        if (objects.size() > 0) {
            frames_with_faces++;
        }
    };

    static std::string typeToString(const Object::Type& type) {
        switch (type) {
            case Object::Type::UNKNOWN: {
                return "UNKNOWN";
            }
            case Object::Type::PHONE: {
                return "PHONE";
            }
            case Object::Type::CHILD_SEAT: {
                return "CHILD_SEAT";
            }
            default: {
                throw std::runtime_error{
                    std::string("Object::typeToString encountered unrecognized Type: ")
                        + std::to_string(static_cast<int>(type))};
            }
        }
    }

    static Object::Type stringToType(const std::string& label) {
        if (label.compare("UNKNOWN") == 0) {
            return Object::Type::UNKNOWN;
        }
        else if (label.compare("PHONE") == 0) {
            return Object::Type::PHONE;
        }
        else if (label.compare("CHILD_SEAT") == 0) {
            return Object::Type::CHILD_SEAT;
        }
        else {
            throw std::runtime_error{
                std::string("Object::stringToType encountered unrecognized Type: ") + label};
        }
    }

    void outputToFile(const std::map<ObjectId, Object>& objects, double time_stamp) override {
        if (objects.empty()) {
            // TimeStamp objectId confidence upperLeftX upperLeftY lowerRightX lowerRightY ObjectType"
            out_stream << time_stamp << ",nan,nan,nan,nan,nan,nan,nan";
            for (const auto& cr :cabin_regions_) {
                out_stream << ",nan";
            }
            out_stream << std::endl;
        }

        for (const auto& object_id_pair : objects) {
            Object obj = object_id_pair.second;
            std::vector<Point> bbox({obj.boundingBox.getTopLeft(), obj.boundingBox.getBottomRight()});

            out_stream << time_stamp << ","
                       << object_id_pair.first << ","
                       << obj.confidence << ","
                       << std::setprecision(0) << bbox[0].x << "," << bbox[0].y << "," << bbox[1].x << "," << bbox[1].y
                       << "," << std::setprecision(4)
                       << typeToString(obj.type);

            for (const auto& cr :cabin_regions_) {
                bool found = false;
                for (const auto& o : obj.matchedRegions) {
                    if (cr.id == o.cabinRegion.id) {
                        out_stream << "," << o.matchConfidence;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    out_stream << "," << 0;
                }
            }
            out_stream << std::endl;
        }
    }

    void draw(const std::map<ObjectId, Object>& objects, const vision::Frame& image) override {
        std::shared_ptr<unsigned char> img_data = image.getBGRByteArray();
        const cv::Mat img = cv::Mat(image.getHeight(), image.getWidth(), CV_8UC3, img_data.get());
        viz.updateImage(img);

        for (const auto& face_id_pair : objects) {
            const Object obj = face_id_pair.second;
            //default color ==GRAY
            cv::Scalar color(128, 128, 128);
            std::string obj_type;
            if (typeToString(obj.type) == "PHONE") {
                //phone color == YELLOW
                color = {0, 255, 255};
                obj_type = "PHONE";
            }
            else if (typeToString(obj.type) == "CHILD_SEAT") {
                //child seat color == RED
                color = {0, 0, 255};
                obj_type = "CHILD_SEAT";
            }

            viz.drawObjectMetrics(obj, color, obj_type);
        }
        viz.showImage();
        image_data = viz.getImageData();
    }

    void processResults() override {
        while (getDataSize() > 0) {
            const std::pair<vision::Frame, std::map<ObjectId, Object>> dataPoint = getData();
            vision::Frame frame = dataPoint.first;
            const std::map<ObjectId, Object> objects = dataPoint.second;

            if (draw_display) {
                draw(objects, frame);
            }

            outputToFile(objects, frame.getTimestamp());

            if (logging_enabled) {
                std::cout << "timestamp: " << frame.getTimestamp()
                          << " pfps: " << getProcessingFrameRate()
                          << " objects: " << objects.size() << std::endl;
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
    std::map<Feature, Duration> callback_intervals_;
    std::vector<CabinRegion> cabin_regions_;
};
