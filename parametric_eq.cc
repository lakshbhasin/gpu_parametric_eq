#include "parametric_eq.hh"

/* Internal structs */

/* This is used to pass arguments to interleaveCallback. */
typedef struct channelParamEQInfo
{
    /* A pointer to the Parametric EQ of interest. */
    ParametricEQ *paramEQ;

    /* The channel number */
    uint16_t channel;

    /* The number of samples to process per channel. */
    uint32_t samplesToProcess;
} channelParamEQInfo;


/* TODO: move these error-check functions to some other namespace... */

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
void gpuFFTHandle(cufftResult errval, const char *file,
                                int line)
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
 * This constructor just takes a number of filters and the current
 * (initial) values of those filters. It then initializes the host filters
 * and signals that the filters should be copied over to the device side.
 *
 * Note that we will make the elements in "hostFilters" point to
 * device-side property structs, since this makes it much easier to
 * cudaMemcpy the data over to the GPU.
 */
ParametricEQ::ParametricEQ(uint16_t numFilters, const Filter *filters) :
    numFilters(numFilters), threadsPerBlock(0), numBlocks(0),
    numBufSamples(0), bufTimeMuS(0)
{
    // Allocate space for the filters on both the host and device side.
    size_t filterArrSize = numFilters * sizeof(Filter);
    
    hostFilters = (Filter *) malloc(filterArrSize);
    gpuErrChk( cudaMalloc((void **) &devFilters, filterArrSize) );
    
    setFilters(filters);

    // Note: Any remaining variables or arrays will be initialized when the
    // song has been chosen and the user has decided to start processing.
}


/**
 * A setter function that sets the "hostFilters" array appropriately, and
 * signals that we need to copy this array over to "devFilters".
 *
 * Note that we will make the elements in "hostFilters" point to
 * device-side property structs, since this makes it much easier to
 * cudaMemcpy the data over to the GPU.
 *
 * Precondition: hostFilters is not NULL.
 */
void ParametricEQ::setFilters(const Filter *filters)
{
    if (hostFilters == NULL)
    {
        throw std::logic_error("The host-side Filter array was NULL.");
    }

    // Free old filters' properties (i.e. not hostFilters and devFilters
    // themselves, just the device-side property structs that those Filters
    // point to). This should only be done if we've actually initialized
    // those filters' properties.
    if (initializedDevFilterProps)
    {
        freeFilterProperties();
    }

    // Set the new filter properties.
    for (uint16_t i = 0; i < numFilters; i ++)
    {
        Filter thisFilter = filters[i];        
        hostFilters[i].type = thisFilter.type;

        // A pointer to this filter's property struct, which will be stored
        // **on device**.
        void *devFiltProp;

        switch(thisFilter.type)
        {
            case FT_BAND_BOOST:
            case FT_BAND_CUT:
            {
                BandBoostCutProp *bandBCProp = thisFilter.bandBCProp;
                
                // Allocate space for this filter's properties on device.
                gpuErrChk( cudaMalloc((void **) &devFiltProp,
                                      sizeof(BandBoostCutProp)) );
                
                // Copy over the data in bandBCProp
                gpuErrChk( cudaMemcpy(devFiltProp, bandBCProp,
                                      sizeof(BandBoostCutProp),
                                      cudaMemcpyHostToDevice) );
                
                // Make the host filter point to the device-side
                // properties.
                hostFilters[i].bandBCProp = (BandBoostCutProp *)
                    devFiltProp;

                break;
            }

            case FT_HIGH_SHELF:
            case FT_LOW_SHELF:
            {
                ShelvingProp *shelvingProp = thisFilter.shelvingProp;
                
                // Allocate space for this filter's properties on device.
                gpuErrChk( cudaMalloc((void **) &devFiltProp,
                                      sizeof(ShelvingProp)) );
                
                // Copy over the data in shelvingProp
                gpuErrChk( cudaMemcpy(devFiltProp, shelvingProp,
                                      sizeof(ShelvingProp),
                                      cudaMemcpyHostToDevice) );
                
                // Make the host filter point to the device-side
                // properties.
                hostFilters[i].shelvingProp = (ShelvingProp *)
                    devFiltProp;
                
                break;
            }
            
            default:
                throw std::invalid_argument("Invalid filter type: " +
                        std::to_string(thisFilter.type));
        }

    }

    // We can now free the device-side filter properties later, if we want.
    initializedDevFilterProps = true;

    // Signal that we want to copy over the new filters and set up the new
    // transfer function. This boolean will be checked in processAudio().
    hostFiltersChanged = true;
}


/**
 * This function sets the "song" to the song at the specified file name, by
 * loading that file's data and metadata into a WavData object.
 *
 * Precondition: The parametric EQ cannot be processing while this happens.
 */
void ParametricEQ::setSong(const char *fileName)
{
    if (processing)
    {
        throw std::logic_error("Cannot change song while Parametric EQ is "
                "still processing.");
    }

    // Delete any old song
    if (song != NULL)
    {
        delete song;
        song = NULL;
    }

    // Read in the new song's data.
    song = new WavData(/* verbose */ true);
    song->loadData(fileName);
}


/**
 * This function just sets numBufSamples (which must be positive). It also
 * requires a new set of filters, since changing the buffer size changes
 * the transfer function.
 *
 * Precondition: The parametric EQ must be paused (but it could be in the
 * middle of a song) or done processing.
 */
void ParametricEQ::setNumBufSamples(uint32_t numBufSamp, Filter *filters)
{
    if (!paused && processing)
    {
        throw std::logic_error("Cannot change numBufSamples while "
                "Parametric EQ is not paused and still processing.");
    }
    
    if (numBufSamp <= 0)
    {
        throw std::invalid_argument("numBufSamples must be positive.");
    }

    if (filters == NULL)
    {
        throw std::invalid_argument("filters cannot be NULL.");
    }

    numBufSamples = numBufSamp;

    // Signal a transfer function update
    setFilters(filters);
}


/**
 * This function just sets the number of threads per block to use (which
 * must be positive).
 *
 * Precondition: The parametric EQ must be paused (but it could be in the
 * middle of a song) or done processing.
 */
void ParametricEQ::setThreadsPerBlock(uint16_t tPerBlock)
{
    if (!paused && processing)
    {
        throw std::logic_error("Cannot change threadsPerBlock while "
                "Parametric EQ is not paused and still processing.");
    }
    
    if (tPerBlock <= 0)
    {
        throw std::invalid_argument("threadsPerBlock must be positive.");
    }

    threadsPerBlock = tPerBlock;
}


/**
 * This function just sets the number of blocks to use, based on the given
 * (user-specified) max number of blocks (which must be positive).
 *
 * Precondition: The parametric EQ must be paused (but it could be in the
 * middle of a song) or done processing.
 */
void ParametricEQ::setMaxBlocks(uint16_t maxBlocks)
{
    if (!paused && processing)
    {
        throw std::logic_error("Cannot change maxBlocks while Parametric "
                "EQ is not paused and still processing.");
    }
    
    if (maxBlocks <= 0)
    {
        throw std::invalid_argument("maxBlocks must be positive.");
    }

    // Change the number of blocks in case it's too large, since the user
    // actually passed in a "max" number of blocks. We only really need to
    // spread out numBufSamples over all the blocks, with one thread per
    // sample.
    numBlocks = std::min(maxBlocks, 
                         (uint16_t) ceil(((float) numBufSamples) / 
                                         threadsPerBlock));
}

EQStream * ParametricEQ::getSoundStream()
{
    return soundStream;
}

int16_t * ParametricEQ::getHostOutputAudioBuf()
{
    return hostOutputAudioBuf;
}


int16_t * ParametricEQ::getHostClippedAudioBuf()
{
    return hostClippedAudioBuf;
}


WavData * ParametricEQ::getSong()
{
    return song;
}

Filter * ParametricEQ::getCurrentFilter()
{
    return hostFilters;
}

uint32_t ParametricEQ::getNumBufSamples()
{
    return numBufSamples;
}

/**
 * Helper function to return the played time (in seconds) of our song, as a
 * float.
 */
float ParametricEQ::getPlayedTime()
{
    return (float) samplesPlayed / song->samplingRate;
}

/**
 * A callback function that gets signalled whenever a channel's data has
 * bene processed. That channel's data will be moved from
 * hostClippedAudioBuf to hostOutputAudioBuf via interleaving.
 *
 * Note that "userData" is of type channelParamEQInfo, and thus includes
 * information about the ParametricEQ object (which is needed since this
 * function is static) and the channel.
 */
void ParametricEQ::interleaveCallback(cudaStream_t stream,
                                      cudaError_t status, 
                                      void *userData)
{
    // Unpack the channelParamEQInfo struct.
    channelParamEQInfo *info = static_cast<channelParamEQInfo *>(userData);

    ParametricEQ *thisEQ = info->paramEQ;
    uint16_t ch = info->channel;
    uint32_t samplesToProcess = info->samplesToProcess;
    uint16_t numChannels = (thisEQ->getSong())->numChannels;

    // Copy the samplesToProcess samples from hostClippedAudioBuf, starting
    // at index ch * numBufSamples (to only look at this channel's data).
    // These will be copied into hostOutputAudioBuf, which requires
    // interleaving.
    uint32_t startIndex = ch * thisEQ->getNumBufSamples();
    
    int16_t *hostOutputAudioBuf = thisEQ->getHostOutputAudioBuf();
    int16_t *hostClippedAudioBuf = thisEQ->getHostClippedAudioBuf();
    
    for (uint32_t i = 0; i < samplesToProcess; i++)
    {
        hostOutputAudioBuf[i * numChannels + ch] = 
            hostClippedAudioBuf[startIndex + i];
    }


#ifndef NDEBUG
    boost::posix_time::ptime now = 
        boost::posix_time::microsec_clock::local_time();
    cout << "End processing channel " << ch << ": " << now << endl;

    if (ch == numChannels - 1)
    {
        cout << endl;
    }
#endif

    delete info;
}


/**
 * This function is called periodically to start processing the next audio
 * buffer in the song (by making the appropriate GPU calls, etc). As long
 * as there are still samples left, we keep re-calling this function.
 *
 */
void ParametricEQ::processAudio(const boost::system::error_code &e,
                                boost::asio::deadline_timer *processTimer)
{

#ifndef NDEBUG
    boost::posix_time::ptime now = 
        boost::posix_time::microsec_clock::local_time();
    cout << "Started processing: " << now << endl;
#endif

    // If we've been signalled to pause by the user, just indicate that we
    // got the signal and return.
    if (paused)
    {
        processAudioPaused = true;
        return;
    }

    // Only process if we're not done processing the whole song.
    if (samplesProcessed < song->numSamplesPerChannel)
    {
        // The number of samples to process in this buffer (used to avoid
        // going out of bounds on the last buffer). Note that this is a
        // per-channel count as usual.
        uint32_t samplesToProcess = std::min(song->numSamplesPerChannel
                                             - samplesProcessed,
                                             numBufSamples);

        // Check that the previous processing run is complete, for each
        // channel.
        uint16_t numChannels = song->numChannels;
        for (int i = 0; i < numChannels; i++)
        {
            gpuErrChk( cudaEventSynchronize(finishedInterleaving[i]) );
        }

        // Check if the transfer function changed. If so, we need to copy
        // over the new filters and set up the new transfer function.
        if (hostFiltersChanged)
        {
            // Copy over filters to device while blocking the host.
            gpuErrChk( cudaMemcpy((void **) devFilters, hostFilters,
                                  numFilters * sizeof(Filter),
                                  cudaMemcpyHostToDevice) ); 
            
            // Set up the transfer function on the device side.
            cudaCallFilterSetupKernel(numBlocks, threadsPerBlock,
                                      /* stream */ 0, devFilters,
                                      numFilters, devTransferFunc,
                                      song->samplingRate,
                                      numBufSamples);
            checkCUDAKernelError();

            // Synchronize across the whole device to make sure the
            // transfer function isn't changing later on.
            // TODO: check how slow this is.
            gpuErrChk( cudaDeviceSynchronize() ); 

            hostFiltersChanged = false;
        }

        // Our current position in the song, after having processed
        // "samplesProcessed" samples per channel.
        int16_t *rawData = song->data + numChannels * samplesProcessed;
    
        size_t numEntriesPerFFT = floor(numBufSamples/2) + 1;
        
        // Each stream will handle one channel (on the GPU side).
        for (int ch = 0; ch < numChannels; ch ++)
        {
            // Move data into host's pinned audio input float buffers, while
            // un-interleaving it as well.
            for (uint32_t i = 0; i < samplesToProcess; i ++)
            {
                // Store each channel's data contiguously.
                hostInputAudioBuf[i + ch * numBufSamples] =
                    (float) rawData[i * numChannels + ch];
            }

            // After each channel's data has been loaded, copy that data
            // asynchronously to devInputAudioBuf (into the appropriate
            // location).
            gpuErrChk( cudaMemcpyAsync(
                                &devInputAudioBuf[ch * numBufSamples],
                                &hostInputAudioBuf[ch * numBufSamples],
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
                                &devInputAudioBuf[ch * numBufSamples],
                                &devFFTAudioBuf[ch * numEntriesPerFFT]) );
            
            gpuFFTChk( cufftDestroy(forwardPlans[ch]) );

            // Process this channel's buffer. This involves pointwise
            // multiplication by the transfer function, as well as dividing
            // by the number of samples per buffer (because of cuFFT's
            // scaling properties).
            //
            // Note that numBufSamples is passed in since the FFT's
            // length is unchanged.
            cudaCallProcessBufKernel(numBlocks, threadsPerBlock,
                                     streams[ch],
                                     &devFFTAudioBuf[ch * numEntriesPerFFT],
                                     devTransferFunc, numBufSamples);
            checkCUDAKernelError();
            
            // Set up an inverse C->R FFT on this channel's data. 
            gpuFFTChk( cufftPlan1d(&inversePlans[ch], samplesToProcess,
                                   CUFFT_C2R, 1) );
            gpuFFTChk( cufftSetStream(inversePlans[ch], streams[ch]) );

            // Schedule the inverse C->R FFT. Store the result in the
            // corresponding location in devUnclippedAudioBuf.
            gpuFFTChk( cufftExecC2R(inversePlans[ch],
                            &devFFTAudioBuf[ch * numEntriesPerFFT],
                            &devUnclippedAudioBuf[ch * numBufSamples]));

            gpuFFTChk( cufftDestroy(inversePlans[ch]) );

            // Carry out clipping on this channel's output buffer, and
            // store the clipped result in the appropriate location in
            // devClippedAudioBuf.
            cudaCallClippingKernel(numBlocks, threadsPerBlock,
                                streams[ch],
                                &devUnclippedAudioBuf[ch * numBufSamples],
                                samplesToProcess,
                                &devClippedAudioBuf[ch * numBufSamples]);
            checkCUDAKernelError();

            // Copy the contiguous section of this channel's data in
            // devClippedAudioBuf into hostClippedAudioBuf.
            gpuErrChk( cudaMemcpyAsync(
                        &hostClippedAudioBuf[ch * numBufSamples],
                        &devClippedAudioBuf[ch * numBufSamples],
                        samplesToProcess * sizeof(int16_t),
                        cudaMemcpyDeviceToHost,
                        streams[ch]) );

            // Schedule a callback that will take this channel's data
            // (stored contiguously in hostClippedAudioBuf), and interleave
            // it into host memory (hostOutputAudioBuf) so that SFML can
            // play it back properly later on.
            channelParamEQInfo *chParamEQInfo = new channelParamEQInfo;
            chParamEQInfo->paramEQ = this;
            chParamEQInfo->channel = ch;
            chParamEQInfo->samplesToProcess = samplesToProcess;
            
            gpuErrChk( cudaStreamAddCallback(streams[ch],
                                             interleaveCallback,
                                             chParamEQInfo,
                                             0) );
            
            // Record when the interleaving finishes for this channel.
            gpuErrChk( cudaEventRecord(finishedInterleaving[ch],
                                       streams[ch]) );
        }
        
        samplesProcessed += samplesToProcess;

#ifndef NDEBUG
        now = boost::posix_time::microsec_clock::local_time();
        cout << "Set up GPU: " << now << endl;
#endif


        // If this is the first call to processAudio(), then it would've
        // taken a while for the CPU and caches to warm up. So the next
        // call will just be bufTimeMuS microseconds from now.
        if (!doneWithFirstProcessCall)
        {
            processTimer->expires_from_now(
                    boost::posix_time::microseconds(bufTimeMuS));
            processTimer->async_wait(boost::bind(&ParametricEQ::processAudio,
                        this, boost::asio::placeholders::error,
                        processTimer));

#ifndef NDEBUG
            cout << "Next call: " << bufTimeMuS << " microseconds"
                 << " from now." << endl;
#endif
        }
        else
        {
            // Otherwise, we'll assume that the processing is going
            // properly and just schedule the next call bufTimeMuS
            // microseconds after the previous one.
            processTimer->expires_at(processTimer->expires_at() + 
                    boost::posix_time::microseconds(bufTimeMuS));
            processTimer->async_wait(boost::bind(&ParametricEQ::processAudio,
                        this, boost::asio::placeholders::error,
                        processTimer));
            
#ifndef NDEBUG
            cout << "Next processing call: " << processTimer->expires_at() 
                 << endl;
#endif
        }

        // If this is the first processAudio() call, set up a call to the
        // playAudio() function, 0.5 * bufTimeMuS microseconds from now
        // (delay arbitrarily chosen). The playAudio() function will then
        // just call itself repeatedly.
        //
        // We carry out the first call to playAudio() here, because there's
        // a certain warm-up time before we can start processing audio very
        // fast.
        if (!doneWithFirstProcessCall)
        { 
            doneWithFirstProcessCall = true;
            
            uint64_t muSecTillNextPlayCall = 0.5 * bufTimeMuS;

            boost::asio::deadline_timer *playTimer = new
                boost::asio::deadline_timer(*io,
                        boost::posix_time::microseconds(muSecTillNextPlayCall));
            
            playTimer->async_wait(boost::bind(&ParametricEQ::playAudio,
                        this, boost::asio::placeholders::error, playTimer));
        }
    }
    else
    {
        // If we're done processing the whole song, signal that we've
        // "paused" processing.
        processAudioPaused = true;
    }

}


/**
 * This function is called periodically to start playing the current audio
 * buffer (stored in hostOutputAudioBuf). As long as there are still
 * samples left to play, we keep re-calling this function.
 *
 */
void ParametricEQ::playAudio(const boost::system::error_code &e,
                             boost::asio::deadline_timer *playTimer)
{
#ifndef NDEBUG
    boost::posix_time::ptime now = 
        boost::posix_time::microsec_clock::local_time();
    cout << "Started preparing to play: " << now << endl;
#endif

    // If we've been signalled to pause by the user, just indicate that we
    // got the signal and return.
    if (paused)
    {
        playAudioPaused = true;
        return;
    }

    // Only play if we're not done playing the whole song.
    if (samplesPlayed < song->numSamplesPerChannel)
    {
        // The number of samples to play in this buffer (used to avoid
        // going out of bounds on the last buffer). Note that this is a
        // per-channel count as usual.
        uint32_t samplesToPlay = std::min(song->numSamplesPerChannel
                                          - samplesPlayed,
                                          numBufSamples);

        // Synchronize all streams to make sure they're done interleaving into
        // the host's output audio buffer.
        uint16_t numChannels = song->numChannels;

        for (int i = 0; i < numChannels; i++)
        {
            gpuErrChk( cudaEventSynchronize(finishedInterleaving[i]) );
        }

        // Load the samples into a SoundBuffer. 
        if (!buffer->loadFromSamples(hostOutputAudioBuf,
                                     samplesToPlay * numChannels,
                                     numChannels,
                                     song->samplingRate))
        {
            cerr << "Failed to load samples in hostOutputAudioBuf." << endl;
            exit(EXIT_FAILURE);
        }

        // If this is the first call to playAudio(), we'll load data into
        // the stream and start playing the stream (which occurs on a
        // separate thread).
        if (firstPlayCall)
        {
            firstPlayCall = false;

            soundStream->addBufferData(*buffer);
            soundStream->play();
        }
        else
        {
            // Otherwise, just add the buffer data (since the stream is
            // already playing).
            soundStream->addBufferData(*buffer);
        }

#ifndef NDEBUG
        now = boost::posix_time::microsec_clock::local_time();
        cout << "Started playing: " << now << endl;
#endif

        samplesPlayed += samplesToPlay;

        // Schedule the next call exactly bufTimeMuS microseconds after
        // this timer went off.
        playTimer->expires_at(playTimer->expires_at() + 
                boost::posix_time::microseconds(bufTimeMuS));
        playTimer->async_wait(boost::bind(&ParametricEQ::playAudio, this,
                    boost::asio::placeholders::error, playTimer));
        
#ifndef NDEBUG
        cout << "Next playing call: " << playTimer->expires_at() << endl;
        now = boost::posix_time::microsec_clock::local_time();
        cout << "End setting up playing: " << now << "\n" << endl;
#endif
        
    }
    else
    {
        // If we're done playing the whole song now, signal that we've
        // "paused" playing.
        playAudioPaused = true;

        // Free the playTimer since it was allocated in a different scope.
        delete playTimer;
    }

}


/**
 * This function is called when the "process" button is hit for the first
 * time. It allocates space for all the required arrays (on both the host
 * and device sides), and then starts off a timer that calls 
 * processAudio().
 *
 * Note that this function cleans up after itself, and calls
 * stopProcessingSound() afterwards.
 *
 */
void ParametricEQ::startProcessingSound()
{
    if (song == NULL)
    {
        throw std::logic_error("Song was not initialized before "
                "startProcessingSound() was called.");
    }
    
    /* Host storage */
    
    // If numBufSamples exceeds the total number of samples per channel,
    // then we have to decrease it.
    uint32_t numSamplesPerChannel = song->numSamplesPerChannel;

    if (numBufSamples > numSamplesPerChannel)
    {
        numBufSamples = numSamplesPerChannel;
        cerr << "The number of samples per buffer was greater than the "
             << "number of samples each channel\nhas, so it was set to "
             << numBufSamples << " instead.\n" << endl;
    }
    
    // The amount of time that each buffer length corresponds to, in
    // microseconds.
    bufTimeMuS = (uint64_t) (((float) numBufSamples) /
                             song->samplingRate * 1.0e6);

    // Set up the buffer and sound stream.
    buffer = new sf::SoundBuffer;
    soundStream = new EQStream(song->samplingRate, song->numChannels,
                               numBufSamples, 
                               song->numSamplesPerChannel * 
                               song->numChannels);
                               
    // Set internal state variables.
    paused = false;
    processAudioPaused = false;
    playAudioPaused = false;
    firstPlayCall = true;
    doneWithFirstProcessCall = false;
    samplesPlayed = 0;
    samplesProcessed = 0;

    // The array of host "input" audio buffers will have numChannels *
    // numBufSamples entries. Channels' data will be stored contiguously,
    // not interleaved. Note that this array is pinned for fast access.
    uint16_t numChannels = song->numChannels;
    gpuErrChk( cudaMallocHost((void **) &hostInputAudioBuf,
                              numChannels * numBufSamples *
                              sizeof(float)) );
 
    // The array of host "clipped output" audio buffers. This will have the
    // same number of elements as hostInputAudioBuf, with channels stored
    // contiguously. We'll also pin this array for fast memory transfers.
    gpuErrChk( cudaMallocHost((void **) &hostClippedAudioBuf,
                              numChannels * numBufSamples *
                              sizeof(int16_t)) );


    // The array of host "output" audio buffers. This will also have the
    // same number of elements as hostClippedAudioBuf, except the channels'
    // samples will be interleaved (e.g. the first entry will be sample 0
    // from channel 0, the second entry will be sample 0 from channel 1,
    // etc).
    hostOutputAudioBuf = (int16_t *) malloc(numChannels * numBufSamples *
                                            sizeof(int16_t));
    
    /* Device-related storage */
    
    // An array of CUDA streams, with one stream per channel.
    streams = (cudaStream_t *) malloc(numChannels * sizeof(cudaStream_t));

    for (int i = 0; i < numChannels; i++)
    {
        gpuErrChk( cudaStreamCreate(&streams[i]) );
    }

    // The device-side transfer function. This will have 
    // floor(numBufSamples/2) + 1 entries (since we only include positive
    // frequencies in our FFT).
    size_t fftSize = (floor(numBufSamples/2) + 1) * sizeof(cufftComplex);
    gpuErrChk( cudaMalloc((void **) &devTransferFunc, fftSize) );
    
    // The device-side input audio buffer. This is laid out in the same way
    // as hostInputAudioBuf.
    gpuErrChk( cudaMalloc((void **) &devInputAudioBuf,
                          numChannels * numBufSamples * sizeof(float)) );

    // A device-side FFT'd version of the input audio buffer. This will
    // have numChannels * ( floor(numBufSamples/2) + 1 ) entries in
    // total. Each channel's data will be laid out contiguously (i.e. not
    // interleaved).
    gpuErrChk( cudaMalloc((void **) &devFFTAudioBuf,
                          numChannels * fftSize) );

    // The device-side unclipped version of the post-processed data. This
    // will be laid out in the same way as devInputAudioBuf, with each
    // channel's data stored contiguously.
    gpuErrChk( cudaMalloc((void **) &devUnclippedAudioBuf,
                          numChannels * numBufSamples * sizeof(float)) );

    // The device-side clipped version of the post-processed data. This
    // will also have each channel's data stored contiguously.
    gpuErrChk( cudaMalloc((void **) &devClippedAudioBuf,
                          numChannels * numBufSamples * 
                          sizeof(int16_t)) );
    
    /* Set up timing events. */
    
    finishedInterleaving = (cudaEvent_t *) malloc(numChannels *
                                                  sizeof(cudaEvent_t));

    for (int i = 0; i < numChannels; i++)
    {
        gpuErrChk( cudaEventCreate(&finishedInterleaving[i]) );
    }

    /* Set up forward and inverse cuFFT plan arrays. */
    forwardPlans = (cufftHandle *) malloc(sizeof(cufftHandle) * 
                                          numChannels);
    inversePlans = (cufftHandle *) malloc(sizeof(cufftHandle) *
                                          numChannels);
    
    /* 
     * Set up a timer to call the "processAudio" function periodically,
     * with a waiting time of bufTimeMuS microseconds. Call the function
     * immediately for now, though.
     */
    io = new boost::asio::io_service;
    
    boost::asio::deadline_timer processTimer(*io, 
            boost::posix_time::microseconds(100.0));
    processTimer.async_wait(boost::bind(&ParametricEQ::processAudio, this,
                                        boost::asio::placeholders::error,
                                        &processTimer));

    /* Keep running until both processing and playing are done. */
    processing = true;
    io->run();
    processing = false;

    // Wait until the EQStream is done too.
    while (soundStream->getStatus() == EQStream::Playing)
    {
        // Sleep 1 ms at a time.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

#ifndef NDEBUG
    cout << "EQStream finished playing." << endl;
#endif

    /* Clean up. */
    stopProcessingSound();
}


/**
 * This function is called when we want to stop a song (which happens if we
 * reach the end of the song, or if a "stop" button is pressed). This
 * requires freeing a lot of the input data (which was initialized in
 * startProcessingSound()) and resetting some variables.
 *
 */
void ParametricEQ::stopProcessingSound()
{
    stopProcessingMutex.lock();

    // Pause processAudio() and playAudio(), if they're not already done.
    if (processing)
    {
        pauseProcessingSound();
    }

    // Reset some internal variables.
    samplesPlayed = 0;
    samplesProcessed = 0;
    doneWithFirstProcessCall = false;
    firstPlayCall = true;
    processing = false;
    paused = false;

    freeAllBackendMemory();

    stopProcessingMutex.unlock();
}


/**
 * This function is called when a song pauses playing. It just signals to
 * processAudio() and playAudio() that we want to stop after the next
 * buffer completes.
 */
void ParametricEQ::pauseProcessingSound()
{
    if (paused)
    {
        throw std::logic_error("Parametric EQ was already paused.");
    }

    paused = true;

    // Wait until processAudio() and playAudio() acknowledge that they've
    // paused.
    while (!processAudioPaused || !playAudioPaused)
    {
        // Sleep 1 ms at a time.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Then pause the sound stream too.
    if (soundStream != NULL)
    {
        soundStream->pause();
    }
}


/**
 * This function is called to resume a song after it was paused. This
 * involves restarting the relevant timers.
 *
 * Note that this function only returns after all audio processing is done,
 * or after the user signals a pause/stop.
 */
void ParametricEQ::resumeProcessingSound()
{
    // TODO: check if "paused" boolean was true before this call was made.
    // NOTE: User might have changed transfer functions, samples per buf,
    // threads per block, or max number of blocks. If so, we need to reset
    // all of the memory that was allocated in startProcessingSound().
    throw std::runtime_error("NYI");
}


/**
 * A helper function to free all of the memory associated with the
 * **property structs** of the filters' arrays. Note that each element in
 * "hostFilters" actually stores pointers to device-side property structs,
 * so we need to use cudaFree to actually free those device-side property
 * structs.
 *
 * Note that this function does not free hostFilters and devFilters
 * themselves. We also assume hostFilters is not NULL.
 */
void ParametricEQ::freeFilterProperties()
{
    if (hostFilters == NULL)
    {
        throw std::logic_error("The host-side Filter array was NULL.");
    }

    // Free the host and device-side filters arrays. We don't free the
    // properties on the host side since they just point to device-side
    // memory.
    for (uint16_t i = 0; i < numFilters; i++)
    {
        Filter thisFilter = hostFilters[i];

        // The properties on the host side point to device-side structs, so
        // we use cudaFree to free them. We'll use the filter type to
        // determine the kind of struct to free.
        switch(thisFilter.type)
        {
            case FT_BAND_BOOST:
            case FT_BAND_CUT:
                // Free the BandBoostCutProp on the device side.
                gpuErrChk( cudaFree(thisFilter.bandBCProp) );
                break;

            case FT_HIGH_SHELF:
            case FT_LOW_SHELF:
                // Free the ShelvingProp on the device side.
                gpuErrChk( cudaFree(thisFilter.shelvingProp) );
                break;

            default:
                throw std::invalid_argument("Invalid filter type: " +
                        std::to_string(thisFilter.type));
        }
    }
    
    // Note: we don't free the actual arrays.
}


/**
 * A helper function that frees all of the allocated host-side and
 * device-side memory that's used **on the back end**, and sets various
 * pointers to NULL. Note that this only frees memory that was initialized
 * in startProcessingAudio(). Notably, this memory does **not** include
 * "song", "hostFilters", or "devFilters".
 *
 * Note: this function can be called twice in a row without any issue,
 * since it checks if things have been freed beforehand. Also, this
 * function is thread-safe since it uses a mutex.
 */
void ParametricEQ::freeAllBackendMemory()
{
    // Acquire a mutex lock.
    freeBackendMemMutex.lock();

#ifndef NDEBUG
    cout << "Freeing back-end memory on host and device." << endl;
#endif

    // Check if the streams haven't been freed before.
    if (streams != NULL)
    {
        // Synchronize all GPU streams, destroy them, and free their
        // array.
        for (uint16_t i = 0; i < song->numChannels; i++)
        {
            gpuErrChk( cudaStreamSynchronize(streams[i]) );
            gpuErrChk( cudaStreamDestroy(streams[i]) );
        }

        free(streams);
        streams = NULL;
    }

    // Check if the events haven't been freed before.
    if (finishedInterleaving != NULL)
    {
        // Destroy events and free their array.
        for (uint16_t i = 0; i < song->numChannels; i++)
        {
            gpuErrChk( cudaEventDestroy(finishedInterleaving[i]) );
        }

        free(finishedInterleaving);
        finishedInterleaving = NULL;
    }

    // Free host-side memory, if it hasn't already been freed.
    if (io != NULL)
    {
        delete io;
        io = NULL;
    }

    if (buffer != NULL)
    {
        delete buffer;
        buffer = NULL;
    }

    if (hostInputAudioBuf != NULL)
    {
        gpuErrChk( cudaFreeHost(hostInputAudioBuf) );
        hostInputAudioBuf = NULL;
    }
    
    if (hostClippedAudioBuf != NULL)
    {
        gpuErrChk( cudaFreeHost(hostClippedAudioBuf) );
        hostClippedAudioBuf = NULL;
    }

    if (hostOutputAudioBuf != NULL)
    {
        free(hostOutputAudioBuf);
        hostOutputAudioBuf = NULL;
    }

    if (soundStream != NULL)
    {
        // Signal the sound stream to stop first, before deleting it.
        soundStream->signalStop();
        delete soundStream;
        soundStream = NULL;
    }
   
    // Free cuFFT plans.
    if (forwardPlans != NULL)
    {
        free(forwardPlans);
        forwardPlans = NULL;
    }

    if (inversePlans != NULL)
    {
        free(inversePlans);
        inversePlans = NULL;
    }

    // Free device-side memory, if it hasn't already been freed.
    if (devTransferFunc != NULL)
    {
        gpuErrChk( cudaFree(devTransferFunc) );
        devTransferFunc = NULL;
    }

    if (devInputAudioBuf != NULL)
    {
        gpuErrChk( cudaFree(devInputAudioBuf) );
        devInputAudioBuf = NULL;
    }

    if (devFFTAudioBuf != NULL)
    {
        gpuErrChk( cudaFree(devFFTAudioBuf) );
        devFFTAudioBuf = NULL;
    }

    if (devUnclippedAudioBuf != NULL)
    {
        gpuErrChk( cudaFree(devUnclippedAudioBuf) );
        devUnclippedAudioBuf = NULL;
    }

    if (devClippedAudioBuf != NULL)
    {
        gpuErrChk( cudaFree(devClippedAudioBuf) );
        devClippedAudioBuf = NULL;
    }

    // Free the mutex.
    freeBackendMemMutex.unlock();
}


/* Destructor */
ParametricEQ::~ParametricEQ()
{
    // Free back-end memory first.
    freeAllBackendMemory();

    // Free the song.
    if (song != NULL)
    {
        delete song;
        song = NULL;
    }

    // Free the host-side Filter array, and the device-side filter
    // properties.
    if (hostFilters != NULL)
    {
        freeFilterProperties();    
        free(hostFilters);
        hostFilters = NULL;
    }
    
    // Free the device-side Filter array.
    if (devFilters != NULL)
    {
        gpuErrChk( cudaFree(devFilters) );
        devFilters = NULL;
    }

}
