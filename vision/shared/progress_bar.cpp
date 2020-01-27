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

#include "progress_bar.h"

ProgressBar::ProgressBar() {}

ProgressBar::ProgressBar(unsigned long n_, const char* description_, std::ostream& out_){

    n = n_;
    frequency_update = n_;
    description = description_;
    out = &out_;

    unit_bar = "=";
    unit_space = " ";
    desc_width = std::strlen(description);  // character width of description field

}

void ProgressBar::SetFrequencyUpdate(unsigned long frequency_update_){

    if(frequency_update_ > n){
        frequency_update = n;    // prevents crash if freq_updates_ > n_
    }
    else{
        frequency_update = frequency_update_;
    }
}

void ProgressBar::SetStyle(const char* unit_bar_, const char* unit_space_){

    unit_bar = unit_bar_;
    unit_space = unit_space_;
}

int ProgressBar::GetConsoleWidth(){

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

    return width;
}

int ProgressBar::GetBarLength(){

    // get console width and according adjust the length of the progress bar

    int bar_length = static_cast<int>((GetConsoleWidth() - desc_width - CHARACTER_WIDTH_PERCENTAGE) / 2.);

    return bar_length;
}

void ProgressBar::ClearBarField(){

    for(int i=0;i<GetConsoleWidth();++i){
        *out << " ";
    }
    *out << "\r" << std::flush;
}

void ProgressBar::Progressed(unsigned long idx_)
{
    try{
        if(idx_ > n) throw idx_;

        // determines whether to update the progress bar from frequency_update
        if ((idx_ != n) && (idx_ % (n/frequency_update) != 0)) return;

        // calculate the size of the progress bar
        int bar_size = GetBarLength();

        // calculate percentage of progress
        double progress_percent = idx_* TOTAL_PERCENTAGE/n;

        // calculate the percentage value of a unit bar
        double percent_per_unit_bar = TOTAL_PERCENTAGE/bar_size;

        // display progress bar
        *out << " " << description << " [";

        for(int bar_length=0;bar_length<=bar_size-1;++bar_length){
            if(bar_length*percent_per_unit_bar<progress_percent){
                *out << unit_bar;
            }
            else{
                *out << unit_space;
            }
        }

        *out << "]" << std::setw(CHARACTER_WIDTH_PERCENTAGE + 1) << std::setprecision(1) << std::fixed << progress_percent << "%\r" << std::flush;
    }
    catch(unsigned long e){
        ClearBarField();
        std::cerr << "PROGRESS_BAR_EXCEPTION: _idx (" << e << ") went out of bounds, greater than n (" << n << ")." << std::endl << std::flush;
    }

}
