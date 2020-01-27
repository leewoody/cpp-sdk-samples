#pragma once

#include <boost/program_options.hpp>

class SubCommandParser {

  public:
    SubCommandParser(int argc, char **argv);
    bool process(const boost::program_options::options_description &opt, const int required_count = 0);

    std::string cmd_;
    const std::string app_name_;
    std::vector<std::string> args_;

    std::vector<std::string> required_;
};
