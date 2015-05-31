/**
 * A subclass of SFML's SoundStream. This is used to add processed data,
 * without interrupting playback.
 */

#ifndef EQ_STREAM_HH
#define EQ_STREAM_HH

/* Standard includes */
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

/* SFML includes */
#include <SFML/Audio.hpp>
#include <SFML/System/Time.hpp>

using std::cout;
using std::cerr;
using std::endl;

class EQStream : public sf::SoundStream
{
private:
    
    // The sampling rate of our song.
    uint32_t samplingRate;

    // The number of channels in our song.
    uint16_t numChannels;
    
    // The buffer size (in samples per channel) we use for streaming.
    uint32_t numBufSampPerChannel;

    // The sample that we're currently adding to our stream in
    // addBufferData().
    uint64_t currentSample;
    
    // The number of samples played so far.
    uint64_t samplesPlayed;
    
    // A mutex to prevent concurrent access to "processedSamples".
    std::mutex processedSamplesMutex;

    // This is used to tell our stream to stop playing
    bool signalToStop = false;

    virtual bool onGetData(sf::SoundStream::Chunk &data);
    virtual void onSeek(sf::Time timeOffset);
    
public:

    EQStream(uint32_t samplingRate, uint16_t numChannels,
             uint32_t numBufSampPerChannel, uint64_t totalNumSamples);
     
    void addBufferData(const sf::SoundBuffer &buffer);
    void signalStop();

    // The vector of "processed audio" which we'll use to store the samples
    // we've encountered so far. We'll add on to this array as we process
    // more data.
    std::vector<int16_t> processedSamples;

    ~EQStream();
    
};


#endif // EQ_STREAM_HH
