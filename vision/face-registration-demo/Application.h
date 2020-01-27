#pragma once

#include "SubCommandParser.h"
#include <Core.h>

class FaceDb;

class Application {
  public:
    Application(int argsc, char **argsv);
    ~Application();

    void run();

  private:
    void setCommonOptions(const std::string &usage);
    bool parse(const int required_param_count);

    SubCommandParser command_line_;
    affdex::path data_dir_;
    std::unique_ptr<boost::program_options::options_description> options_;
    bool preview_ = false;

    std::unique_ptr<FaceDb> faceDb_;
};
