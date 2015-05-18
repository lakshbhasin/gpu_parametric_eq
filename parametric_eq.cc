/**
 * This is where we control the front-end to the parametric equalizer
 * program. To run the parametric equalizer, pass in the following
 * arguments:
 *
 *      ./parametric_eq <wav_file> <samp_per_buf> <threads_per_block>
 *                      <max_num_blocks>
 *
 * Where "samp_per_buf" is the number of samples per channel in each
 * buffer, which must be even.
 */

/* Standard includes */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

/* CUDA-related includes */
#include "parametric_eq_cuda.cuh"
#include <cuda_runtime.h>
#include <cufft.h>

/* Boost-related includes */
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>


/* SFML includes */
#include <SFML/Audio.hpp>

/* Other classes */
#include "WavData.hh"


using std::cerr;
using std::cout;
using std::endl;


/* Global "constants" */

const float PI = 3.14159265358979;

/* The number of filters used in this parametric EQ. */
const int16_t NUM_FILTERS = 1;

/* The number of threads per block. Initialized later. */
uint16_t THREADS_PER_BLOCK = 0;

/* The number of blocks to use. Initialized later. */
uint16_t NUM_BLOCKS = 0;

/* The number of samples per buffer. Initialized later. */
uint32_t NUM_BUF_SAMPLES = 0;

/* 
 * The amount of time it takes to play a full buffer of audio, in
 * microseconds. 
 */
uint64_t BUF_TIME_MU_S = 0;

/* The WavData object corresponding to this song. Initialized later. */
WavData *song;

/* 
 * TODO: add any other constants as necessary (e.g. initial values of
 * filters' properties when GUIs are added).
 */


/* Global variables */

/* 
 * An io_service object that keeps our code running until all processing
 * and playing are done.
 */
boost::asio::io_service *io = NULL;

/* The buffer we'll use to load sound into. */
sf::SoundBuffer *buffer = NULL;

/* The sound that we play while buffering. */
sf::Sound *bufferSound = NULL;

/*
 * A boolean used to track whether we've completed the first process call
 * to processAudio(). This will signal the first playback.
 */
bool doneWithFirstProcessCall = false;

/* An array of Filters, to keep on the host's side. */
Filter *hostFilters = NULL;

/* Device-side array of filters. */
Filter *devFilters = NULL;

/* The number of samples we've currently processed. */
uint32_t samplesProcessed = 0;

/* The number of samples we've currently played. */
uint32_t samplesPlayed = 0;

/*
 * An array of host "input" audio buffers. This will have numChannels *
 * NUM_BUF_SAMPLES entries, with the first NUM_BUF_SAMPLES corresponding to
 * channel 0, the next NUM_BUF_SAMPLES corresponding to channel 1, etc (in
 * order to work well with cuFFT). This array will be pinned for fast access.
 *
 * Note that floats are used for this, to conform with cuFFT.
 */
float *hostInputAudioBuf = NULL;

/*
 * An array of host "clipped output" audio buffers. This is laid out in the
 * same way as hostInputAudioBuf, with contiguous data storage for
 * channels.
 *
 * This array will also be pinned for fast access.
 */
int16_t *hostClippedAudioBuf = NULL;

/*
 * An array of host "output" audio buffers. This will also have
 * NUM_CHANNELS * NUM_BUF_SAMPLES entries, except the channels' samples
 * will be interleaved (e.g. the first entry will be from channel 0, the
 * second entry will be from channel 1, etc). This layout allows for easy
 * playback via SFML.
 *
 * Note that 16-bit integers are used in order to conform to SFML.
 */
int16_t *hostOutputAudioBuf = NULL;


/*
 * An array of CUDA streams. Each stream will process one channel's data.
 *
 */
cudaStream_t *streams = NULL;

/*
 * An array of forward and inverse FFT plans. Each plan will handle one
 * channel's data.
 */
cufftHandle *forwardPlans = NULL;
cufftHandle *inversePlans = NULL;

/* 
 * The device-side transfer function. This will be floor(NUM_BUF_SAMPLES/2)
 * + 1 in length (since our FFT only includes positive frequencies).
 */
cufftComplex *devTransferFunc = NULL;

/*
 * The device-side input audio buffer. This is laid out in the same way as
 * hostInputAudioBuf.
 */
float *devInputAudioBuf = NULL;

/*
 * The device-side FFT'd version of the input audio buffer. This will have
 * numChannels * ( floor(NUM_BUF_SAMPLES/2) + 1 ) entries, since each
 * channel's FFT will have floor(NUM_BUF_SAMPLES/2) + 1 elements. Each
 * channel's data will be laid out contiguously (i.e. not interleaved),
 * like in devInputAudioBuf.
 */
cufftComplex *devFFTAudioBuf = NULL;

/*
 * The device-side unclipped version of the post-processed data. This will
 * be laid out in the same way as devInputAudioBuf, with each channel's
 * data stored contiguously.
 */
float *devUnclippedAudioBuf = NULL;

/*
 * The device-side clipped version of the post-processed data. This will be
 * laid out in the same way as devInputAudioBuf, with each channel's data
 * stored contiguously. In order to play back on the host, interleaving (of
 * channels' samples) will have to be carried out while copying data back
 * to the host.
 */
int16_t *devClippedAudioBuf = NULL;


/* Additional structs */

/* This is used to pass arguments to interleaveCallback. */
typedef struct channelInfo
{
    /* The channel number */
    uint16_t channel;

    /* The number of samples to process per channel. */
    uint32_t samplesToProcess;
} channelInfo;


/* Timing events. */

/*
 * Used to check if a channel had finished interleaving its data back
 * into the host output buffer (where audio gets played from). The number
 * of entries in this array will equal numChannels.
 */
cudaEvent_t *finishedInterleaving;


/* Function prototypes */

void checkKernErr(const char *file, int line);
void usage(const char *progName);

void interleaveCallback(cudaStream_t stream, cudaError_t status,
                        void *userData);
void processAudio(const boost::system::error_code &e,
                  boost::asio::deadline_timer *processTimer);
void playAudio(const boost::system::error_code &e,
               boost::asio::deadline_timer *playTimer);


/* Function implementations */

/** 
 * Macro and helper function to check for errors with CUDA runtime API
 * calls. Modified from http://stackoverflow.com/questions/14038589/
 * what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 *
 */
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
    {
        cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file <<
             " " << line << endl;
        exit(code);
    }
}


/**
 * Helper function to turn a cufftResult into a char *.
 *
 */
static const char * cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

        default:
            return "<not included>";
    }
}


/**
 * Macro and helper function to see if a cuFFT call had an error (and if
 * so, where).
 *
 */
#define gpuFFTChk(ans) { gpuFFTHandle((ans), __FILE__, __LINE__); }
void gpuFFTHandle(cufftResult errval, const char *file, int line)
{
    if (errval != CUFFT_SUCCESS)
    {
        cerr << "Failed FFT call, error: " << cudaGetErrorEnum(errval) 
            << " " << file << " " << line << endl;
        exit(errval);
    }
}


/**
 * Macro and helper function to check if the last kernel call had an error.
 * If so, an error message is printed and execution terminates.
 *
 */
#define checkCUDAKernelError() { checkKernErr(__FILE__, __LINE__); }
void checkKernErr(const char *file, int line)
{   
    cudaError err = cudaGetLastError();
    
    if (cudaSuccess != err)
    {
        cerr << "CUDA kernel error: " << cudaGetErrorString(err) << 
            " " << file << " " << line << endl;
        exit(EXIT_FAILURE);
    }
}


/**
 * This function is called when "main" is not passed a valid set of
 * arguments. A usage statement is printed, and the program exits.
 *
 */
void usage(const char *progName)
{
    // The required arguments are: a WAV file's name, the number of samples
    // used for each buffer, the number of threads per block, and the
    // maximum number of blocks.
    cerr << "Usage: " << progName << " <wav_file> <samp_per_buf> " <<
         "<threads_per_block> <max_num_blocks>" << endl;
    cerr << "\nAll integer arguments must be positive." << endl;
    
    exit(EXIT_FAILURE);
}


/**
 * A callback function that gets signalled whenever a channel's data has
 * bene processed. That channel's data will be moved from
 * hostClippedAudioBuf to hostOutputAudioBuf via interleaving.
 */
void interleaveCallback(cudaStream_t stream, cudaError_t status,
                        void *userData)
{
    // Unpack the channelInfo struct.
    channelInfo *info = static_cast<channelInfo *>(userData);
    uint16_t ch = info->channel;
    uint32_t samplesToProcess = info->samplesToProcess;
    uint16_t numChannels = song->numChannels;

#ifndef NDEBUG
    boost::posix_time::ptime now = 
        boost::posix_time::microsec_clock::local_time();
    cout << "End processing channel " << ch << ": " << now << endl;

    if (ch == numChannels - 1)
    {
        cout << endl;
    }

#endif

    // Copy the samplesToProcess samples from hostClippedAudioBuf, starting
    // at index ch * NUM_BUF_SAMPLES (to only look at this channel's data).
    // These will be copied into hostOutputAudioBuf, which requires
    // interleaving.
    uint32_t startIndex = ch * NUM_BUF_SAMPLES;
    
    for (uint32_t i = 0; i < samplesToProcess; i++)
    {
        hostOutputAudioBuf[i * numChannels + ch] = 
            hostClippedAudioBuf[startIndex + i];
    }
}


/**
 * This function is called periodically to start processing the next audio
 * buffer in the song (by making the appropriate GPU calls, etc). As long
 * as there are still samples left, we keep re-calling this function.
 *
 */
void processAudio(const boost::system::error_code &e,
                  boost::asio::deadline_timer *processTimer)
{
    std::chrono::steady_clock::time_point start = 
        std::chrono::steady_clock::now();

#ifndef NDEBUG
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    cout << "Started processing: " << now << endl;
#endif

    // Only process if we're not done processing the whole song.
    if (samplesProcessed < song->numSamplesPerChannel)
    {
        // The number of samples to process in this buffer (used to avoid
        // going out of bounds on the last buffer). Note that this is a
        // per-channel count as usual.
        uint32_t samplesToProcess = std::min(song->numSamplesPerChannel
                                             - samplesProcessed,
                                             NUM_BUF_SAMPLES);

        // Check that the previous processing run is complete, for each
        // channel.
        uint16_t numChannels = song->numChannels;

        for (int i = 0; i < numChannels; i++)
        {
            gpuErrChk( cudaEventSynchronize(finishedInterleaving[i]) );
        }

        // Our current position in the song, after having processed
        // "samplesProcessed" samples per channel.
        int16_t *rawData = song->data + numChannels * samplesProcessed;
    
        size_t numEntriesPerFFT = floor(NUM_BUF_SAMPLES/2) + 1;
        
        // Each stream will handle one channel (on the GPU side).
        for (int ch = 0; ch < numChannels; ch ++)
        {
            // Move data into host's pinned audio input float buffers, while
            // un-interleaving it as well.
            for (uint32_t i = 0; i < samplesToProcess; i ++)
            {
                // Store each channel's data contiguously.
                hostInputAudioBuf[i + ch * NUM_BUF_SAMPLES] =
                    (float) rawData[i * numChannels + ch];
            }

            // After each channel's data has been loaded, copy that data
            // asynchronously to devInputAudioBuf (into the appropriate
            // location).
            gpuErrChk( cudaMemcpyAsync(
                                &devInputAudioBuf[ch * NUM_BUF_SAMPLES],
                                &hostInputAudioBuf[ch * NUM_BUF_SAMPLES],
                                samplesToProcess * sizeof(float),
                                cudaMemcpyHostToDevice,
                                streams[ch]) );

            // Set up the forward R->C FFT on this channel's data in
            // devInputAudioBuf. The number of points is samplesToProcess.
            gpuFFTChk( cufftPlan1d(&forwardPlans[ch], samplesToProcess,
                                   CUFFT_R2C, 1) );
            gpuFFTChk( cufftSetStream(forwardPlans[ch], streams[ch]) );

            // Schedule the forward R->C FFT on this channel's data. Store
            // the result in the corresponding location in
            // devFFTAudioBuf.
            
            gpuFFTChk( cufftExecR2C(forwardPlans[ch],
                                &devInputAudioBuf[ch * NUM_BUF_SAMPLES],
                                &devFFTAudioBuf[ch * numEntriesPerFFT]) );

            // Process this channel's buffer. This involves pointwise
            // multiplication by the transfer function, as well as dividing
            // by the number of samples per buffer (because of cuFFT's
            // scaling properties).
            //
            // Note that NUM_BUF_SAMPLES is passed in since the FFT's
            // length is unchanged.
            cudaCallProcessBufKernel(NUM_BLOCKS, THREADS_PER_BLOCK,
                                     streams[ch],
                                     &devFFTAudioBuf[ch * numEntriesPerFFT],
                                     devTransferFunc, NUM_BUF_SAMPLES);
            checkCUDAKernelError();
            
            // Set up an inverse C->R FFT on this channel's data. 
            gpuFFTChk( cufftPlan1d(&inversePlans[ch], samplesToProcess,
                                   CUFFT_C2R, 1) );
            gpuFFTChk( cufftSetStream(inversePlans[ch], streams[ch]) );

            // Schedule the inverse C->R FFT. Store the result in the
            // corresponding location in devUnclippedAudioBuf.
            gpuFFTChk( cufftExecC2R(inversePlans[ch],
                            &devFFTAudioBuf[ch * numEntriesPerFFT],
                            &devUnclippedAudioBuf[ch * NUM_BUF_SAMPLES]));

            // Carry out clipping on this channel's output buffer, and
            // store the clipped result in the appropriate location in
            // devClippedAudioBuf.
            cudaCallClippingKernel(NUM_BLOCKS, THREADS_PER_BLOCK,
                                streams[ch],
                                &devUnclippedAudioBuf[ch * NUM_BUF_SAMPLES],
                                samplesToProcess,
                                &devClippedAudioBuf[ch * NUM_BUF_SAMPLES]);
            checkCUDAKernelError();

            // Copy the contiguous section of this channel's data in
            // devClippedAudioBuf into hostClippedAudioBuf.
            gpuErrChk( cudaMemcpyAsync(
                        &hostClippedAudioBuf[ch * NUM_BUF_SAMPLES],
                        &devClippedAudioBuf[ch * NUM_BUF_SAMPLES],
                        samplesToProcess * sizeof(int16_t),
                        cudaMemcpyDeviceToHost,
                        streams[ch]) );

            // Schedule a callback that will take this channel's data
            // (stored contiguously in hostClippedAudioBuf), and interleave
            // it into host memory (hostOutputAudioBuf) so that SFML can
            // play it back properly later on.
            channelInfo *chInfo = new channelInfo;
            chInfo->channel = ch;
            chInfo->samplesToProcess = samplesToProcess;
            
            gpuErrChk( cudaStreamAddCallback(streams[ch],
                                             interleaveCallback, chInfo,
                                             0) );
            
            // Record when the interleaving finishes for this channel.
            gpuErrChk( cudaEventRecord(finishedInterleaving[ch],
                                       streams[ch]) );
        }
        
        samplesProcessed += samplesToProcess;

        // Measure time taken to run the above processing code, so we
        // know when to next call the timer.
        std::chrono::steady_clock::time_point end = 
            std::chrono::steady_clock::now();

        uint64_t processTimeMuS = std::chrono::duration_cast
            <std::chrono::microseconds>(end - start).count();

        // Subtract off processing time from waiting time, unless we're on
        // the first call to processAudio() (in which case it'll take a
        // while for the CPU and caches to warm up). 
        uint64_t muSecTillNextCall;
        
        if (!doneWithFirstProcessCall)
        {
            muSecTillNextCall = BUF_TIME_MU_S;
        }
        else
        {
            // The "-40" is just an estimate for how many microseconds some
            // of the following calls will take.
            // TODO: change "-40" stuff?
            muSecTillNextCall = BUF_TIME_MU_S - processTimeMuS - 40;
        }
        
#ifndef NDEBUG
        now = boost::posix_time::microsec_clock::local_time();
        cout << "Set up GPU: " << now << endl;
        cout << "Next call: " << muSecTillNextCall << endl;
#endif
        
        // Set up another callback timer to this processing function, for
        // muSecTillNextCall from now.
        processTimer->expires_from_now(
                    boost::posix_time::microseconds(muSecTillNextCall));
        processTimer->async_wait(boost::bind(processAudio,
                    boost::asio::placeholders::error, processTimer));

        // If this is the first processAudio() call, set up a call to the
        // playAudio() function, 0.5 * BUF_TIME_MU_S microseconds from now
        // (delay arbitrarily chosen). The playAudio() function will then
        // just call itself repeatedly.
        //
        // We carry out the first call to playAudio() here, because there's
        // a certain warm-up time before we can start processing audio very
        // fast. So we shouldn't do this in main().
        if (!doneWithFirstProcessCall)
        { 
            doneWithFirstProcessCall = true;
            
            // TODO: change?
            uint64_t muSecTillNextPlayCall = 0.5 * BUF_TIME_MU_S;

            boost::asio::deadline_timer *playTimer = new
                boost::asio::deadline_timer(*io,
                        boost::posix_time::microseconds(muSecTillNextPlayCall));
            
            playTimer->async_wait(boost::bind(playAudio,
                                     boost::asio::placeholders::error,
                                     playTimer));
        }
    }
}


/**
 * This function is called periodically to start playing the current audio
 * buffer (stored in hostOutputAudioBuf). As long as there are still
 * samples left to play, we keep re-calling this function.
 *
 */
void playAudio(const boost::system::error_code &e,
               boost::asio::deadline_timer *playTimer)
{
    // Record that we just started preparing to play.
    std::chrono::steady_clock::time_point start = 
        std::chrono::steady_clock::now();

#ifndef NDEBUG
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    cout << "Started preparing to play: " << now << endl;
#endif

    // Only play if we're not done playing the whole song.
    if (samplesPlayed < song->numSamplesPerChannel)
    {
        // The number of samples to play in this buffer (used to avoid
        // going out of bounds on the last buffer). Note that this is a
        // per-channel count as usual.
        uint32_t samplesToPlay = std::min(song->numSamplesPerChannel
                                          - samplesPlayed,
                                          NUM_BUF_SAMPLES);

        // Synchronize all streams to make sure they're done interleaving into
        // the host's output audio buffer.
        uint16_t numChannels = song->numChannels;

        for (int i = 0; i < numChannels; i++)
        {
            gpuErrChk( cudaEventSynchronize(finishedInterleaving[i]) );
        }

        if (!buffer->loadFromSamples(hostOutputAudioBuf,
                                     samplesToPlay * numChannels,
                                     numChannels,
                                     song->samplingRate))
        {
            cerr << "Failed to load samples in hostOutputAudioBuf." << endl;
            exit(EXIT_FAILURE);
        }
        
        bufferSound->setBuffer(*buffer);
        bufferSound->play();

#ifndef NDEBUG
        now = boost::posix_time::microsec_clock::local_time();
        cout << "Started playing: " << now << endl;
#endif

        samplesPlayed += samplesToPlay;

        // Measure time taken to run the above playing code, so we know
        // when to next call the timer.
        std::chrono::steady_clock::time_point end = 
            std::chrono::steady_clock::now();

        uint64_t playTimeMuS = std::chrono::duration_cast
            <std::chrono::microseconds>(end - start).count();

        // Subtract off playing time from waiting time. The "40" is just a
        // good estimate of how long the following operations (up to the
        // async_wait call) take, in microseconds.
        // TODO: change "-40" stuff?
        uint64_t muSecTillNextCall = BUF_TIME_MU_S - playTimeMuS - 40;
        
        if (playTimeMuS >= BUF_TIME_MU_S)
        {
            cerr << "Initiating playback is taking too long." << endl;
            muSecTillNextCall = BUF_TIME_MU_S;
        }

#ifndef NDEBUG
        cout << "Next playing call: " << muSecTillNextCall << endl;
        now = boost::posix_time::microsec_clock::local_time();
        cout << "End setting up playing: " << now << "\n" << endl;
#endif

        // Set up another callback timer to this processing function,
        // for muSecTillNextCall from now.
        playTimer->expires_from_now(
                    boost::posix_time::microseconds(muSecTillNextCall));
        playTimer->async_wait(boost::bind(playAudio,
                    boost::asio::placeholders::error, playTimer));
    }

}


int main(int argc, char *argv[])
{
    /* Argument parsing */

    // There should be 4 arguments.
    if (argc != 5)
    {
        usage(argv[0]);
    }

    // Unpack the remaining arguments and check that they're valid.
    char *fileName = argv[1];
    NUM_BUF_SAMPLES = atoi(argv[2]);
    THREADS_PER_BLOCK = atoi(argv[3]);
    uint16_t maxBlocks = atoi(argv[4]);
    
    if (NUM_BUF_SAMPLES <= 0 || THREADS_PER_BLOCK <= 0 || maxBlocks <= 0)
    {
        usage(argv[0]);
    }


    /* Reading in the song's data, other setup. */

    song = new WavData(/* verbose */ true);
    song->loadData(fileName);
    
    // If NUM_BUF_SAMPLES exceeds the total number of samples per channel,
    // then we have to decrease it.
    uint32_t numSamplesPerChannel = song->numSamplesPerChannel;

    if (NUM_BUF_SAMPLES > numSamplesPerChannel)
    {
        NUM_BUF_SAMPLES = numSamplesPerChannel;
        cerr << "The number of samples per buffer was greater than the "
             << "number of samples each channel\nhas, so it was set to "
             << NUM_BUF_SAMPLES << " instead.\n" << endl;
    }

    // Change the number of blocks in case it's too large, since the user
    // actually passed in a "max" number of blocks. We only really need to
    // spread out NUM_BUF_SAMPLES over all the blocks, with one thread per
    // sample.
    NUM_BLOCKS = std::min(maxBlocks, 
                          (uint16_t) ceil(((float) NUM_BUF_SAMPLES) / 
                                          THREADS_PER_BLOCK));
    
    // The amount of time that each buffer length corresponds to, in
    // microseconds.
    BUF_TIME_MU_S = (uint64_t) (((float) NUM_BUF_SAMPLES) /
                                song->samplingRate * 1.0e6);

    // Set up the bufferSound and buffer.
    buffer = new sf::SoundBuffer;
    bufferSound = new sf::Sound;
    

    /* Setting up parametric EQ's filters. */

    // TODO: add suport for filters to change over time.
    // TODO: add support for a GUI that controls these filters.
    size_t filterArrSize = NUM_FILTERS * sizeof(Filter);

    hostFilters = (Filter *) malloc(filterArrSize);
    gpuErrChk( cudaMalloc((void **) &devFilters, filterArrSize) );

    // Test a band-pass boost filter.
    float freq = 300.0;         // Hz
    float bandwidth = 100.0;    // Hz
    float gain = 20.0;           // dB (must be positive)

    // Allocate space for the filter's properties on device.
    BandBoostCutProp *devBandBCProp;
    gpuErrChk( cudaMalloc((void **) &devBandBCProp,
                          sizeof(BandBoostCutProp)) );

    BandBoostCutProp *hostBandBCProp = (BandBoostCutProp *) 
        malloc(sizeof(BandBoostCutProp));
    hostBandBCProp->omegaNought = 2.0 * PI * freq;
    hostBandBCProp->Q = freq/bandwidth;
    hostBandBCProp->K = std::pow(10.0, gain/20.0);
    
    gpuErrChk( cudaMemcpy(devBandBCProp, hostBandBCProp,
                          sizeof(BandBoostCutProp),
                          cudaMemcpyHostToDevice) );
    
    free(hostBandBCProp);
    
    // Make the host filter point to the device-side properties.
    hostFilters[0].type = FT_BAND_BOOST;
    hostFilters[0].bandBCProp = devBandBCProp;
    
    // Copy over filters to device
    // TODO: make this async?
    gpuErrChk( cudaMemcpy((void **) devFilters, hostFilters,
                          filterArrSize, cudaMemcpyHostToDevice) );
    

    /* Setting up other data storage on the host and device. */
    
    /* Host storage */

    // The array of host "input" audio buffers will have numChannels *
    // NUM_BUF_SAMPLES entries. Channels' data will be stored contiguously,
    // not interleaved. Note that this array is pinned for fast access.
    uint16_t numChannels = song->numChannels;
    gpuErrChk( cudaMallocHost((void **) &hostInputAudioBuf,
                              numChannels * NUM_BUF_SAMPLES *
                              sizeof(float)) );
 
    // The array of host "clipped output" audio buffers. This will have the
    // same number of elements as hostInputAudioBuf, with channels stored
    // contiguously. We'll also pin this array for fast memory transfers.
    gpuErrChk( cudaMallocHost((void **) &hostClippedAudioBuf,
                              numChannels * NUM_BUF_SAMPLES *
                              sizeof(int16_t)) );


    // The array of host "output" audio buffers. This will also have the
    // same number of elements as hostClippedAudioBuf, except the channels'
    // samples will be interleaved (e.g. the first entry will be sample 0
    // from channel 0, the second entry will be sample 0 from channel 1,
    // etc).
    hostOutputAudioBuf = (int16_t *) malloc(numChannels * NUM_BUF_SAMPLES *
                                            sizeof(int16_t));
    

    /* Device-related storage */
    
    // An array of CUDA streams, with one stream per channel.
    streams = (cudaStream_t *) malloc(numChannels * sizeof(cudaStream_t));

    for (int i = 0; i < numChannels; i++)
    {
        gpuErrChk( cudaStreamCreate(&streams[i]) );
    }

    // The device-side transfer function. This will have 
    // floor(NUM_BUF_SAMPLES/2) + 1 entries (since we only include positive
    // frequencies in our FFT).
    size_t fftSize = (floor(NUM_BUF_SAMPLES/2) + 1) * sizeof(cufftComplex);
    gpuErrChk( cudaMalloc((void **) &devTransferFunc, fftSize) );
    
    // The device-side input audio buffer. This is laid out in the same way
    // as hostInputAudioBuf.
    gpuErrChk( cudaMalloc((void **) &devInputAudioBuf,
                          numChannels * NUM_BUF_SAMPLES * sizeof(float)) );

    // A device-side FFT'd version of the input audio buffer. This will
    // have numChannels * ( floor(NUM_BUF_SAMPLES/2) + 1 ) entries in
    // total. Each channel's data will be laid out contiguously (i.e. not
    // interleaved).
    gpuErrChk( cudaMalloc((void **) &devFFTAudioBuf,
                          numChannels * fftSize) );

    // The device-side unclipped version of the post-processed data. This
    // will be laid out in the same way as devInputAudioBuf, with each
    // channel's data stored contiguously.
    gpuErrChk( cudaMalloc((void **) &devUnclippedAudioBuf,
                          numChannels * NUM_BUF_SAMPLES * sizeof(float)) );

    // The device-side clipped version of the post-processed data. This
    // will also have each channel's data stored contiguously.
    gpuErrChk( cudaMalloc((void **) &devClippedAudioBuf,
                          numChannels * NUM_BUF_SAMPLES * 
                          sizeof(int16_t)) );
    
    /* Set up timing events. */
    
    finishedInterleaving = (cudaEvent_t *) malloc(numChannels *
                                                  sizeof(cudaEvent_t));

    for (int i = 0; i < numChannels; i++)
    {
        gpuErrChk( cudaEventCreate(&finishedInterleaving[i]) );
    }
    
    
    /* Set up the transfer function on device. */
    
    // TODO: add support for this to be updated as the user changes things.
    cudaCallFilterSetupKernel(NUM_BLOCKS, THREADS_PER_BLOCK,
                              /* stream */ 0, devFilters,
                              NUM_FILTERS, devTransferFunc,
                              song->samplingRate,
                              NUM_BUF_SAMPLES);
    checkCUDAKernelError();

    // Synchronize across the whole device.
    gpuErrChk( cudaDeviceSynchronize() );


    /* Set up forward and inverse cuFFT plan arrays. */
    forwardPlans = (cufftHandle *) malloc(sizeof(cufftHandle) * 
                                          numChannels);
    inversePlans = (cufftHandle *) malloc(sizeof(cufftHandle) *
                                          numChannels);

    
    /* 
     * Set up a timer to call the "processAudio" function periodically,
     * with a waiting time of BUF_TIME_MU_S microseconds. Call the function
     * immediately for now, though.
     */

    io = new boost::asio::io_service;

    boost::asio::deadline_timer processTimer(*io, 
            boost::posix_time::microseconds(100.0));
    processTimer.async_wait(boost::bind(processAudio,
                                        boost::asio::placeholders::error,
                                        &processTimer));

    /* Keep running until both processing and playing are done. */
    io->run();
    
    
    /* Free memory. */

#ifndef NDEBUG
    cout << "Freeing memory on host and device." << endl;
#endif

    // Synchronize all GPU streams, destroy them, and free their array.
    for (int i = 0; i < numChannels; i++)
    {
        gpuErrChk( cudaStreamSynchronize(streams[i]) );
        gpuErrChk( cudaStreamDestroy(streams[i]) );
    }

    free(streams);

    // Free host-side memory.
    delete song;
    delete io;
    delete bufferSound;
    delete buffer;

    gpuErrChk( cudaFreeHost(hostInputAudioBuf) );
    gpuErrChk( cudaFreeHost(hostClippedAudioBuf) );
    free(hostOutputAudioBuf);

    // Free cuFFT plans.
    if (forwardPlans != NULL)
    {
        free(forwardPlans);
    }

    if (inversePlans != NULL)
    {
        free(inversePlans);
    }


    // Free device-side memory.
    gpuErrChk( cudaFree(devTransferFunc) );
    gpuErrChk( cudaFree(devInputAudioBuf) );
    gpuErrChk( cudaFree(devFFTAudioBuf) );
    gpuErrChk( cudaFree(devUnclippedAudioBuf) );
    gpuErrChk( cudaFree(devClippedAudioBuf) );
    

    return EXIT_SUCCESS;
}


