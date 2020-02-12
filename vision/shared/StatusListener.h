#pragma once

#include <ProcessStatusListener.h>

#include <iostream>
#include <mutex>

using namespace affdex;

class StatusListener : public vision::ProcessStatusListener {
public:

    StatusListener() : is_running(true) {};

    void onProcessingException(Exception& ex) override {
        printException(ex);
        m.lock();
        is_running = false;
        m.unlock();
    };

    bool isRunning() {
        bool ret = true;
        m.lock();
        ret = is_running;
        m.unlock();
        return ret;
    };

private:
    // prints the explanatory string of an exception. If the exception is nested,
    // recurses to print the explanatory string of the exception it holds
    static void printException(const std::exception& e, int level = 0) {
        std::cerr << std::string(level, ' ') << "exception: " << e.what() << '\n';
        try {
            std::rethrow_if_nested(e);
        }
        catch (const std::exception& e) {
            printException(e, level + 1);
        }
        catch (...) {
        }
    }

    std::mutex m;
    bool is_running;
};
