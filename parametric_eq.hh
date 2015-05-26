/**
 * This class encapsulates the parametric equalizer and its interactions
 * with the CUDA back-end. 
 */

#ifndef PARAMETRIC_EQ_HH
#define PARAMETRIC_EQ_HH

/* Standard includes */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>

/* CUDA-related includes */
#include "parametric_eq_cuda.cuh"
#include <cuda_runtime.h>
#include <cufft.h>

/* Boost-related includes */
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

/* SFML includes */
#include <SFML/Audio.hpp>

/* Custom classes' includes */
#include "WavData.hh"
#include "eq_stream.hh"

using std::cerr;
using std::cout;
using std::endl;


class ParametricEQ
{
private:
    
    // The number of filters used by this parametric EQ. This is
    // initialized once and should not change over time.
    const uint16_t numFilters;

    // The WavData object corresponding to the song that is currently
    // chosen.
    WavData *song = NULL;

    // Track whether the Parametric EQ is currently processing (and
    // playing) audio.
    bool processing = false;

    // Track whether the Parametric EQ has currently paused playing and
    // processing audio.
    bool paused = false;

    // Track whether the device-side filters' properties have been
    // initialized for this ParametricEQ.
    bool initializedDevFilterProps = false;

    // We also track if processAudio() and playAudio() have acknowledged a
    // pause command (since these run on separate threads).
    bool processAudioPaused = false;
    bool playAudioPaused = false;

    // Tracks whether this is the first playAudio() call.
    bool firstPlayCall = true;

    // The number of threads per block to use. This should only be
    // changeable if the EQ is not processing.
    uint16_t threadsPerBlock = 0;

    // The number of blocks to use. Also only changeable if the EQ is not
    // processing.
    uint16_t numBlocks = 0;

    // The number of samples to use per buffer and **per channel**. This is
    // only changeable if the EQ is not processing.
    uint32_t numBufSamples = 0;

    // The amount of time it takes to play a full buffer of audio, in
    // microseconds.
    uint64_t bufTimeMuS = 0;

    // An io_service object that keeps our code running until all
    // processing and playing are done.
    boost::asio::io_service *io = NULL;

    // The buffer we'll use to load sound into.
    sf::SoundBuffer *buffer = NULL;

    // The sound stream that we'll load audio into while buffering.
    EQStream *soundStream = NULL;

    // Tracks whether we've completed the first process call to
    // processAudio() for this song. This will signal the first playback.
    bool doneWithFirstProcessCall = false;

    // An array of Filters, to keep on the host's side. Note that the
    // elements in this array will point to device-side property structs,
    // since this makes it much easier to cudaMemcpy the data over to the
    // GPU.
    Filter *hostFilters = NULL;

    // Device-side array of filters.
    Filter *devFilters = NULL;

    // This tells us whether hostFilters was changed. If this is true, then
    // that signals to us that we need to copy over the new filter data to
    // the device, and recalculate the transfer function.
    bool hostFiltersChanged = false;
    
    // The number of samples we've currently processed.
    uint32_t samplesProcessed = 0;

    // The number of samples we've currently played.
    uint32_t samplesPlayed = 0;
    
    // An array of host "input" audio buffers. This will have numChannels
    // * numBufSamples entries, with the first numBufSamples corresponding
    // to channel 0, the next numBufSamples corresponding to channel 1, etc
    // (in order to work well with cuFFT). This array will be pinned for
    // fast access.
    //
    // Note that floats are used for this, to conform with cuFFT.
    float *hostInputAudioBuf = NULL;
    
    // An array of host "clipped output" audio buffers. This is laid out in
    // the same way as hostInputAudioBuf, with contiguous data storage for
    // channels. This array will also be pinned for fast access.
    int16_t *hostClippedAudioBuf = NULL;

    // An array of host "output" audio buffers. This will also have
    // NUM_CHANNELS * numBufSamples entries, except the channels' samples
    // will be interleaved (e.g. the first entry will be from channel 0,
    // thesecond entry will be from channel 1, etc). This layout allows for
    // easy playback via SFML.
    //
    // Note that 16-bit integers are used in order to conform to SFML.
    int16_t *hostOutputAudioBuf = NULL;

    // An array of CUDA streams. Each stream will process one channel's
    // data.
    cudaStream_t *streams = NULL;
    
    // An array of forward and inverse FFT plans. Each plan will handle one
    // channel's data.
    cufftHandle *forwardPlans = NULL;
    cufftHandle *inversePlans = NULL;

    // The device-side transfer function. This will be
    // floor(numBufSamples/2) + 1 in length (since our FFT only includes
    // positive frequencies).
    cufftComplex *devTransferFunc = NULL;

    // The device-side input audio buffer. This is laid out in the same way
    // as hostInputAudioBuf.
    float *devInputAudioBuf = NULL;

    // The device-side FFT'd version of the input audio buffer. This will
    // have numChannels * ( floor(numBufSamples/2) + 1 ) entries, since
    // each channel's FFT will have floor(numBufSamples/2) + 1 elements.
    // Each channel's data will be laid out contiguously (i.e. not
    // interleaved), like in devInputAudioBuf.
    cufftComplex *devFFTAudioBuf = NULL;

    // The device-side unclipped version of the post-processed data. This
    // will be laid out in the same way as devInputAudioBuf, with each
    // channel's data stored contiguously.
    float *devUnclippedAudioBuf = NULL;

    // The device-side clipped version of the post-processed data. This
    // will be laid out in the same way as devInputAudioBuf, with each
    // channel's data stored contiguously. In order to play back on the
    // host, interleaving (of channels' samples) will have to be carried
    // out while copying data back to the host.
    int16_t *devClippedAudioBuf = NULL;
    
    // Timing events that are used to check if a channel had finished
    // interleaving its data back into the host output buffer (where audio
    // gets played from). The number of entries in this array will equal
    // numChannels.
    cudaEvent_t *finishedInterleaving;

    // A mutex to make sure multiple threads don't try to free back-end
    // memory at the same time.
    std::mutex freeBackendMemMutex;
    
    // A mutex to make sure multiple threads don't try to stop processing
    // audio at the same time.
    std::mutex stopProcessingMutex;

    static void interleaveCallback(cudaStream_t stream, cudaError_t status,
                                   void *userData);

    void processAudio(const boost::system::error_code &e,
                      boost::asio::deadline_timer *processTimer);
    void playAudio(const boost::system::error_code &e,
                   boost::asio::deadline_timer *playTimer);
    void freeFilterProperties();
    void freeAllBackendMemory();

public:
    
    ParametricEQ(uint16_t numFilters, const Filter *filters);

    // Setter functions
    void setSong(const char *fileName);
    void setNumBufSamples(uint32_t numBufSamp);
    void setFilters(const Filter *filters);
    void setThreadsPerBlock(uint16_t tPerBlock);
    void setMaxBlocks(uint16_t maxBlocks);

    // Getter functions (only used in interleaveCallback()).
    int16_t *getHostOutputAudioBuf();
    int16_t *getHostClippedAudioBuf();
    WavData *getSong();
    uint32_t getNumBufSamples();
    int getPlayedTime();

    // Song playback functions
    void startProcessingSound();
    void pauseProcessingSound();
    void resumeProcessingSound();
    void stopProcessingSound();

    ~ParametricEQ();

};

#endif // PARAMETRIC_EQ_HH
