#pragma once

#include "Visualizer.h"

#include <ImageListener.h>

#include <deque>
#include <mutex>
#include <fstream>
#include <condition_variable>

#include <iostream>
#include <iomanip>


using namespace affdex;

class PlottingImageListener : public vision::ImageListener {

public:

    PlottingImageListener(std::ofstream &csv, bool draw_display, bool enable_logging, bool draw_face_id) :
        draw_display(draw_display),
        capture_last_ts(0),
        capture_fps(0),
        process_last_ts(0),
        process_fps(0),
        out_stream(csv),
        start(std::chrono::system_clock::now()),
        processed_frames(0),
        frames_with_faces(0),
        draw_face_id(draw_face_id),
        logging_enabled(enable_logging) {
        out_stream << "TimeStamp,faceId,upperLeftX,upperLeftY,lowerRightX,lowerRightY,confidence,interocularDistance,";
        for (const auto& angle : viz.HEAD_ANGLES) out_stream << angle.second << ",";
        for (const auto& emotion : viz.EMOTIONS) out_stream << emotion.second << ",";
        for (const auto& expression : viz.EXPRESSIONS) out_stream << expression.second << ",";
        out_stream << "mood,dominantEmotion,dominantEmotionConfidence";
        out_stream << "identity,identityConfidence,age,ageConfidence,ageCategory,gender,genderConfidence";
        out_stream << std::endl;
        out_stream.precision(2);
        out_stream << std::fixed;
    }

    unsigned int getProcessingFrameRate() {
        std::lock_guard<std::mutex> lg(mtx);
        return process_fps;
    }

    unsigned int getCaptureFrameRate() {
        std::lock_guard<std::mutex> lg(mtx);
        return capture_fps;
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

    std::pair<vision::Frame, std::map<vision::FaceId, vision::Face>> getData() {
        std::lock_guard<std::mutex> lg(mtx);
        std::pair<vision::Frame, std::map<vision::FaceId, vision::Face>> dpoint = results.front();
        results.pop_front();
        return dpoint;
    }

    void onImageResults(std::map<vision::FaceId, vision::Face> faces, vision::Frame image) override {
        std::lock_guard<std::mutex> lg(mtx);
        results.emplace_back(image, faces);
        process_fps = 1000.0f / (image.getTimestamp() - process_last_ts) ;
        process_last_ts = image.getTimestamp();

        processed_frames++;
        if (faces.size() > 0)
        {
            frames_with_faces++;
        }
    };

    void onImageCapture(vision::Frame image) override {
        std::lock_guard<std::mutex> lg(mtx);
        capture_fps = 1000.0f / (image.getTimestamp() - capture_last_ts);
        capture_last_ts = image.getTimestamp();
    };

    void outputToFile(const std::map<vision::FaceId, vision::Face> faces, const double timeStamp) {
        if (faces.empty()) {
            out_stream << timeStamp
                << ",nan,nan,nan,nan,nan,nan,nan,"; // face ID, bbox UL X, UL Y, BR X, BR Y, confidence, interocular distance
            for (const auto& angle : viz.HEAD_ANGLES) out_stream << "nan,";
            for (const auto& emotion : viz.EMOTIONS) out_stream << "nan,";
            for (const auto& expression : viz.EXPRESSIONS) out_stream << "nan,";
            // mood, dominant emotion, dominant emotion confidence, identity, identity_confidence, age, age_confidence, age_category, gender, gender_confidence
            out_stream << "nan,nan,nan,nan,nan,nan,nan,nan,nan,nan";
            out_stream << std::endl;
        }

        for (auto & face_id_pair : faces) {
            vision::Face f = face_id_pair.second;
            std::vector<vision::Point> bbox(f.getBoundingBox());

            out_stream << timeStamp << ","
                << f.getId() << ","
                << std::setprecision(0) << bbox[0].x << "," << bbox[0].y << "," << bbox[1].x << "," << bbox[1].y << "," << std::setprecision(4)
                << f.getConfidence() << ","
                << f.getMeasurements().at(vision::Measurement::INTEROCULAR_DISTANCE) << ",";

            auto measurements = f.getMeasurements();
            for (auto m : viz.HEAD_ANGLES) {
                out_stream << measurements.at(m.first) << ",";
            }

            auto emotions = f.getEmotions();
            for (auto emo : viz.EMOTIONS) {
                out_stream << emotions.at(emo.first) << ",";
            }

            auto expressions = f.getExpressions();
            for (auto exp : viz.EXPRESSIONS) {
                out_stream << expressions.at(exp.first) << ",";
            }

            vision::Mood mood = f.getMood();
            out_stream << viz.MOODS[mood] << ",";

            vision::DominantEmotionMetric dominant_emotion_metric = f.getDominantEmotion();
            out_stream << viz.DOMINANT_EMOTIONS[dominant_emotion_metric.dominantEmotion] << "," << dominant_emotion_metric.confidence << ",";

            auto identity_metric = f.getIdentityMetric();
            out_stream << identity_metric.id << "," << identity_metric.confidence << ",";

            auto age_metric = f.getAgeMetric();
            out_stream << age_metric.years << "," << age_metric.confidence << ",";

            auto age_category = f.getAgeCategory();
            out_stream << viz.AGE_CATEGORIES.at(age_category) << ",";

            auto gender_metric = f.getGenderMetric();
            out_stream << viz.GENDER.at(gender_metric.gender) << "," << gender_metric.confidence << ",";

            out_stream << std::endl;
        }
    }

    void draw(const std::map<vision::FaceId, vision::Face> faces, const vision::Frame& image) {
        std::shared_ptr<unsigned char> imgdata = image.getBGRByteArray();
        const cv::Mat img = cv::Mat(image.getHeight(), image.getWidth(), CV_8UC3, imgdata.get());
        viz.updateImage(img);

        for (auto & face_id_pair : faces) {
            vision::Face f = face_id_pair.second;

            std::map<vision::FacePoint, vision::Point> points = f.getFacePoints();

            // Draw bounding box
            auto bbox = f.getBoundingBox();
            const float valence = f.getEmotions().at(vision::Emotion::VALENCE);
            viz.drawBoundingBox(bbox, valence);

            // Draw Facial Landmarks Points
            viz.drawPoints(f.getFacePoints());

            // Draw a face on screen
            viz.drawFaceMetrics(f, bbox, draw_face_id);
        }

        viz.showImage();
    }

    void processResults() {
        while (getDataSize() > 0) {
            const std::pair<vision::Frame, std::map<vision::FaceId, vision::Face> > dataPoint = getData();
            vision::Frame frame = dataPoint.first;
            const std::map<vision::FaceId, vision::Face> faces = dataPoint.second;

            if (draw_display) {
                draw(faces, frame);
            }

            outputToFile(faces, frame.getTimestamp());

            if (logging_enabled) {
                std::cout << "timestamp: " << frame.getTimestamp()
                << " cfps: " << getCaptureFrameRate()
                << " pfps: " << getProcessingFrameRate()
                << " faces: "<< faces.size() << std::endl;
            }
        }
    }

    void reset() {
        std::lock_guard<std::mutex> lg(mtx);
        capture_last_ts = 0;
        capture_fps = 0;
        process_last_ts = 0;
        process_fps = 0;
        start = std::chrono::system_clock::now();
        processed_frames = 0;
        frames_with_faces = 0;
        results.clear();
    }

private:
    bool draw_display;
    std::mutex mtx;
    std::deque<std::pair<vision::Frame, std::map<vision::FaceId, vision::Face> > > results;

    timestamp capture_last_ts;
    unsigned int capture_fps;
    timestamp process_last_ts;
    unsigned int process_fps;
    std::ofstream &out_stream;
    std::chrono::time_point<std::chrono::system_clock> start;

    Visualizer viz;

    unsigned int processed_frames;
    unsigned int frames_with_faces;
    bool draw_face_id;
    bool logging_enabled;
};
