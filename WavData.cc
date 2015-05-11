#include "WavData.hh"

WavData::WavData()
{
    data = NULL;
    actualSize = 0;
    numChannels = 0;
    frequency = 0;
    resolution = 0;
}

void WavData::loadData(const char *fname)
{
    FILE* wavFile = fopen(fname, "rb");

    if (wavFile)
    {
        // Each char is 1 byte.
        char tag[5];
        // Each unsigned short or short is 2 bytes.
        unsigned short overallSize, formatLength, ch;
        unsigned short freq, avgBytesSec, realSize;
        unsigned short blockAlign, bitsPerSample;
        short formatTag;
        fread(tag, sizeof(char), 4, wavFile);
        // Sometimes there would be random trailing characters. Need
        // to clean out.
        tag[4] = '\0';
        if (!strcmp(tag, "RIFF"))
        {
            // Read in the size of overall file.
            fread(&overallSize, sizeof(unsigned short), 2, wavFile);
            // Make sure file type is WAVE.
            fread(tag, sizeof(char), 4, wavFile);
            tag[4] = '\0';
            if (!strcmp(tag, "WAVE"))
            {
                cout << "Overall size: " << overallSize << " bytes" << endl;
                // Header info "fmt "
                fread(tag, sizeof(char), 4, wavFile);
                tag[4] = '\0';
                fread(&formatLength, sizeof(unsigned short), 2, wavFile);
                fread(&formatTag, sizeof(short), 1, wavFile);
                fread(&ch, sizeof(unsigned short), 1, wavFile);
                cout << "Number of channels: " << ch << endl;
                numChannels = ch;
                fread(&freq, sizeof(unsigned short), 2, wavFile);
                cout << "Frequency: " << freq << " Hz" << endl;
                frequency = freq;

                // (Sample Rate * Bits Per Sample * Channels) / 8
                fread(&avgBytesSec, sizeof(unsigned short), 2, wavFile);

                // (Bits Per Sample * Channels) / 8
                fread(&blockAlign, sizeof(unsigned short), 1, wavFile);
                fread(&bitsPerSample, sizeof(unsigned short), 1, wavFile);
                cout << "Bits per sample: " << bitsPerSample << endl;
                resolution = bitsPerSample;
                // Header for "data"
                fread(tag, sizeof(char), 4, wavFile);
                tag[4] = '\0';

                fread(&realSize, sizeof(unsigned short), 2, wavFile);
                cout << "Actual data size: " << realSize << " bytes" << endl;
                actualSize = realSize;

                // Set up actual data of short array.
                unsigned short numShorts = realSize / sizeof(short);
                data = (short*)malloc(realSize);
                fread(data, sizeof(short), numShorts, wavFile);
                cout << "Loaded WAV file into WavData object." << endl;
            }
            else
            {
                cout << "Error: RIFF file but not a WAVE file!" << endl;
                exit(1);
            }
        }
        else
        {
            // Not a RIFF file. Throw error.
            cout << "Error: not a RIFF-WAV file!" << endl;
            exit(1);
        }
    }
    // Close the file.
    fclose(wavFile);
}

WavData::~WavData()
{
    free(data);
}
