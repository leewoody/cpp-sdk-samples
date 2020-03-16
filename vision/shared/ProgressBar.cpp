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

#include "ProgressBar.h"

ProgressBar::ProgressBar() {}

ProgressBar::ProgressBar(unsigned long n, const char* description, std::ostream& out_l) {

    n_ = n;
    frequency_update_ = n;
    description_ = description;
    out_ = &out_l;

    unit_bar_ = "=";
    unit_space_ = " ";
    desc_width_ = std::strlen(description_);  // character width of description_ field

}

void ProgressBar::setFrequencyUpdate(unsigned long frequency_update) {

    if (frequency_update > n_) {
        frequency_update_ = n_;    // prevents crash if freq_updates_ > n_
    }
    else {
        frequency_update_ = frequency_update;
    }
}

void ProgressBar::setStyle(const char* unit_bar, const char* unit_space) {

    unit_bar_ = unit_bar;
    unit_space_ = unit_space;
}

int ProgressBar::getConsoleWidth() {

    int width;

#ifdef _WINDOWS
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    width = csbi.srWindow.Right - csbi.srWindow.Left;
#else
    struct winsize win;
    ioctl(0, TIOCGWINSZ, &win);
    width = win.ws_col;
#endif

    //if console width too big or too small.
    if (width > 255 || width <= 0) {
        width = 255;
    }

    return width;
}

int ProgressBar::getBarLength() {

    // get console width and according adjust the length of the progress bar

    int bar_length = static_cast<int>((getConsoleWidth() - desc_width_ - CHARACTER_WIDTH_PERCENTAGE) / 2.);

    return bar_length;
}

void ProgressBar::clearBarField() {

    for (int i = 0; i < getConsoleWidth(); ++i) {
        *out_ << " ";
    }
    *out_ << "\r" << std::flush;
}

void ProgressBar::progressed(unsigned long idx) {
    try {
        if (idx > n_) {
            throw idx;
        }

        // determines whether to update the progress bar from frequency_update_
        if ((idx != n_) && (idx % (n_ / frequency_update_) != 0)) {
            return;
        }

        // calculate the size of the progress bar
        int bar_size = getBarLength();

        // calculate percentage of progress
        double progress_percent = idx * TOTAL_PERCENTAGE / n_;

        // calculate the percentage value of a unit bar
        double percent_per_unit_bar = TOTAL_PERCENTAGE / bar_size;

        // display progress bar
        *out_ << " " << description_ << " [";

        for (int bar_length = 0; bar_length <= bar_size - 1; ++bar_length) {
            if (bar_length * percent_per_unit_bar < progress_percent) {
                *out_ << unit_bar_;
            }
            else {
                *out_ << unit_space_;
            }
        }

        *out_ << "]" << std::setw(CHARACTER_WIDTH_PERCENTAGE + 1) << std::setprecision(1) << std::fixed
              << progress_percent << "%\r\n" << std::flush;
    }
    catch (unsigned long e) {
        clearBarField();
        std::cerr << "PROGRESS_BAR_EXCEPTION: _idx (" << e << ") went out of bounds, greater than n (" << n_ << ")."
                  << std::endl << std::flush;
    }
}
