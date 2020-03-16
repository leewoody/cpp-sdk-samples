// From: https://github.com/htailor/cpp_progress_bar
//
// MIT License

// Copyright (c) 2016 Hemant Tailor

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#ifdef _WINDOWS
#include <windows.h>
#else
#include <sys/ioctl.h>

#endif

#include <iostream>
#include <iomanip>
#include <cstring>

#define TOTAL_PERCENTAGE 100.0
#define CHARACTER_WIDTH_PERCENTAGE 4

class ProgressBar {

public:

    ProgressBar();
    ProgressBar(unsigned long n, const char* description = "", std::ostream& out_l = std::cerr);

    void setFrequencyUpdate(unsigned long frequency_update);
    void setStyle(const char* unit_bar, const char* unit_space);

    void progressed(unsigned long idx);

private:

    unsigned long n_;
    unsigned int desc_width_;
    unsigned long frequency_update_;
    std::ostream* out_;

    const char* description_;
    const char* unit_bar_;
    const char* unit_space_;

    void clearBarField();
    int getConsoleWidth();
    int getBarLength();
};

