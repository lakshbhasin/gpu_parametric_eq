#ifndef WAVDATA_HH
#define WAVDATA_HH

#include <cmath> // for ceil
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string.h> 

// The header file information is found here:
// http://www.topherlee.com/software/pcm-tut-wavformat.html

using std::cout;
using std::cerr;
using std::endl;
using std::int16_t;
using std::uint16_t;

/* Class for WAV data */
class WavData
{
public:
    /* Whether verbose output is printed out. */
    const bool verbose;
    
    /* Array of 16-bit PCM data. */
    int16_t *data;

    /* Size of actual audio data in bytes */
    uint32_t actualSize;

    /* Number of channels */
    uint16_t numChannels;

    /* The number of samples per channel */
    uint32_t numSamplesPerChannel;

    /* Sampling rate */
    uint32_t samplingRate;

    /* Bits per sample (e.g. 16 for 16-bit PCM sound). */
    uint16_t bitsPerSample;

    WavData(const bool verbose);
    void loadData(const char *fname);
    float duration();
    ~WavData();
};

#endif // WAVDATA_HH

