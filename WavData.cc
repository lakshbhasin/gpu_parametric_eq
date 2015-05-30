#include "WavData.hh"

WavData::WavData(const bool verbose) : 
    verbose(verbose)
{
    data = NULL;
    actualSize = 0;
    numSamplesPerChannel = 0;
    numChannels = 0;
    samplingRate = 0;
    bitsPerSample = 0;
}

void WavData::loadData(const char *fname)
{
    FILE* wavFile = fopen(fname, "rb");

    if (wavFile)
    {
        // Each char is 1 byte.
        char tag[5];

        // Each of these is 4 bytes.
        uint32_t overallSize, sampRate, realSize;
        uint32_t avgBytesSec, formatLength;

        // Each of these is 2 bytes.
        uint16_t ch, blockAlign, bPerSample;
        int16_t formatTag;

        fread(tag, sizeof(char), 4, wavFile);
        
        // Set last character of 4-character tag to NULL char.
        tag[4] = '\0';
        
        if (!strcmp(tag, "RIFF"))
        {
            // Read in the size of overall file.
            fread(&overallSize, sizeof(uint32_t), 1, wavFile);

            // Make sure file type is WAVE.
            fread(tag, sizeof(char), 4, wavFile);
            tag[4] = '\0';

            if (!strcmp(tag, "WAVE"))
            {
                if (verbose)
                {
                    cout << "Overall size: " << overallSize << " bytes"
                         << endl;
                }

                // Header info "fmt "
                fread(tag, sizeof(char), 4, wavFile);
                tag[4] = '\0';

                fread(&formatLength, sizeof(uint16_t), 2, wavFile);
                fread(&formatTag, sizeof(int16_t), 1, wavFile);

                fread(&ch, sizeof(uint16_t), 1, wavFile);
                
                if (verbose)
                {
                    cout << "Number of channels: " << ch << endl;
                }
                
                numChannels = ch;
                
                fread(&sampRate, sizeof(uint32_t), 1, wavFile);
                
                if (verbose)
                {
                    cout << "Sampling Rate: " << sampRate << " Hz" << endl;
                }
                
                samplingRate = sampRate;

                // (Sample Rate * Bits Per Sample * Channels) / 8
                fread(&avgBytesSec, sizeof(uint16_t), 2, wavFile);

                // (Bits Per Sample * Channels) / 8
                fread(&blockAlign, sizeof(uint16_t), 1, wavFile);
                fread(&bPerSample, sizeof(uint16_t), 1, wavFile);

                // The bits per sample must (currently) be 16!
                if (bPerSample != 16)
                {
                    cerr << "Error: bits per sample was " << bPerSample << 
                        ", not" << " 16! Only 16-bit PCM audio can " <<
                        "currently be\nprocessed." << endl;
                    exit(EXIT_FAILURE);
                }
                
                if (verbose)
                {
                    cout << "Bits per sample: " << bPerSample << endl;
                }
                
                bitsPerSample = bPerSample;
                
                // Header for "data"
                fread(tag, sizeof(char), 4, wavFile);
                tag[4] = '\0';

                fread(&realSize, sizeof(uint32_t), 1, wavFile);
                
                // TODO: figure out why the above read is an issue?
                realSize = overallSize - 36;
                
                if (verbose)
                {
                    cout << "Actual data size: " << realSize << " bytes"
                         << endl;
                }

                actualSize = realSize;

                // The number of samples per channel is given by taking
                // realSize, dividing by the number of channels, and then
                // dividing by the number of bytes per sample.
                numSamplesPerChannel = realSize / 
                    (numChannels * bitsPerSample / 8);

                if (verbose)
                {
                    cout << "Samples per channel: " << numSamplesPerChannel
                        << endl;
                }

                // Set up actual data of 16-bit int array.
                uint32_t numShorts = realSize / sizeof(int16_t);
                data = (int16_t *) malloc(realSize);
                fread(data, sizeof(int16_t), numShorts, wavFile);
                
                if (verbose)
                {
                    cout << "Loaded WAV file into WavData object.\n" << endl;
                }
            }
            else
            {
                cerr << "Error: RIFF file but not a WAVE file!" << endl;
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            // Not a RIFF file. Throw error.
            cerr << "Error: not a RIFF-WAV file!" << endl;
            exit(EXIT_FAILURE);
        }
    }
    // Close the file.
    fclose(wavFile);
}

float WavData::duration()
{
    float duration = (float) actualSize /
        (samplingRate * numChannels * bitsPerSample / 8);
    return duration;
}

WavData::~WavData()
{
    if (data != NULL)
    {
        free(data);
    }
}
