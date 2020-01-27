#include "Application.h"

#include <iostream>

using namespace std;

int main(int argsc, char **argsv) {
    try {
        Application app(argsc, argsv);
        app.run();
    } catch (const exception &err) {
        cerr << err.what() << endl;
        return -1;
    }

    return 0;
}
