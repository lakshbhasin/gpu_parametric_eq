#include "eq_stream.hh"


/**
 * A constructor for this EQStream. This just takes various song data
 * parameters and initializes our vector of "processedSamples" (which has
 * "totalNumSamples" elements, where this is the total number of samples in
 * the song). We also initialize the base class here.
 */
EQStream::EQStream(uint32_t samplingRate, uint16_t numChannels,
                   uint32_t numBufSampPerChannel, uint64_t totalNumSamples) :
    samplingRate(samplingRate), numChannels(numChannels),
    numBufSampPerChannel(numBufSampPerChannel), 
    currentSample(0), signalToStop(false), 
    processedSamples(totalNumSamples, 0), samplesPlayed(0)
{
    // Initialize the base class.
    initialize(numChannels, samplingRate);
}


/**
 * This function adds the given buffer's data to the end of our vector of
 * processed samples.
 */
void EQStream::addBufferData(const sf::SoundBuffer &buffer)
{
    // Prevent concurrent access of processedSamples.
    processedSamplesMutex.lock();
   
    assert(currentSample + buffer.getSampleCount() <=
           processedSamples.size());

    // Copy over data from the buffer into processedSamples[currentSample]
    memcpy(&processedSamples[currentSample], buffer.getSamples(),
           buffer.getSampleCount() * sizeof(int16_t));
    
    // Increase currentSample so that the next buffer can be processed.
    currentSample += buffer.getSampleCount();

    processedSamplesMutex.unlock();
}


/**
 * A helper function that used to indicate that we want to stop this
 * SoundStream.
 */
void EQStream::signalStop()
{
    signalToStop = true;
}


/**
 * This function sets the properties of the next "Chunk" to play. If
 * everything is alright and there are more samples to play, this function
 * returns true. If playback is finished, it returns false.
 */
bool EQStream::onGetData(sf::SoundStream::Chunk &data)
{
    // If we've reached the end of the song, then immediately return false.
    if (samplesPlayed >= processedSamples.size())
    {
        data.samples = NULL;
        data.sampleCount = 0;
        return false;
    }

    // Wait until currentSample is > samplesPlayed, since this tells us
    // that we're ready to load the next set of samples.
    //
    // Alternatively, if this stream is signalled to stop, then we should
    // leave this loop and return false.
    while (currentSample <= samplesPlayed && !signalToStop)
    {
        // cout << currentSample << " " << samplesPlayed << " " <<
        //    processedSamples.size() << endl;
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    // Check if we exited the above loop because we were signalled to stop.
    if (signalToStop)
    {
        signalToStop = false;
        data.samples = NULL;
        data.sampleCount = 0;
        return false;
    }

    // Prevent concurrent access of processedSamples.
    processedSamplesMutex.lock();

    // Otherwise, make data.samples point to the beginning of the next set
    // of audio samples to play.
    data.samples = &processedSamples[samplesPlayed];

    // Check if there's enough space for a full buffer of sound.
    if (samplesPlayed + numBufSampPerChannel * numChannels <=
            processedSamples.size())
    {
        // In this case, we can stream an entire buffer's worth of sound.
        data.sampleCount = numBufSampPerChannel * numChannels;
        samplesPlayed += numBufSampPerChannel * numChannels;
    }
    else
    {
        // In this case, we can only stream the remaining samples (not a
        // full buffer).
        data.sampleCount = processedSamples.size() - samplesPlayed;
        samplesPlayed = processedSamples.size();
    }

    processedSamplesMutex.unlock();

    return true;
}


/**
 * This function changes the current playing position in processedSamples,
 * and is called whenever the setPlayingOffset() public function is called.
 * The given "timeOffset" specifies the part of the stream to seek to,
 * relative to the beginning of the song.
 *
 * This seeking behavior is **not** currently supported. 
 */
void EQStream::onSeek(sf::Time timeOffset)
{
    // Convert the time offset into a sample number.
    uint64_t sampleToSeekTo = (uint64_t) 
        (timeOffset.asSeconds() * samplingRate * numChannels);

    // The sample to seek to must be <= currentSample (i.e. how much of the
    // sound we've currently processed). We cannot seek forwards since we
    // haven't processed that data yet.
    assert(sampleToSeekTo <= currentSample);
 
    // Don't actually seek. It seems as if setPlayingOffset() is called
    // after our stream starts playing, which is problematic since it moves
    // the stream back and makes things inconsistent. We can deal with this
    // by just not seeking.

    // currentSample = sampleToSeekTo;
    // samplesPlayed = sampleToSeekTo;
}


EQStream::~EQStream()
{
    // No dynamically-allocated resources to free at the moment.
}
