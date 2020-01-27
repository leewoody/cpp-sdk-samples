#include "SubCommandParser.h"

#include <iostream>

using namespace std;
namespace po = boost::program_options;

SubCommandParser::SubCommandParser(int argc, char **argv) : app_name_(argv[0]), args_(argv, argv + argc) {

    if (args_.size() < 2) {
        return;
    } else {
        cmd_ = args_[1];
        args_.erase(args_.begin());
        args_.erase(args_.begin());
    }
}

bool SubCommandParser::process(const po::options_description &opt, const int required_count) {
    po::parsed_options parsed = po::command_line_parser(args_).options(opt).run();
    required_ = po::collect_unrecognized(parsed.options, po::include_positional);

    po::variables_map vm;
    po::store(parsed, vm);

    if (vm["help"].as<bool>() || required_.size() < required_count) {
        cout << opt << endl;
        return false;
    }
    po::notify(vm);
    return true;
}
