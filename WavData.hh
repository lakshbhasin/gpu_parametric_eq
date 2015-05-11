#ifndef WAVDATA_HH
#define WAVDATA_HH

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string.h> 

// The header file information is found here:
// http://www.topherlee.com/software/pcm-tut-wavformat.html

using std::cout;
using std::endl;

/* Class for WAV data */
class WavData
{
    public:
        /* Array of data */
        short* data;

        /* Size of actual audio data in bytes */
        unsigned short actualSize;

        /* Number of channels */
        unsigned short numChannels;

        /* Sample rate or frequency */
        unsigned short frequency;

        /* Resolution or bits per sample */
        unsigned short resolution;

        WavData();
        void loadData(const char *fname);
        ~WavData();
};

#endif // WAVDATA_HH

