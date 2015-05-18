#include "WavData.hh"

/* Test out loading WAV file into WavData */
int main()
{
    // Initialize a WavData object.
    WavData song(/* verbose */ true);
    song.loadData("data/sample17sec.wav");

    cout << "\n\nAfter loading, testing:" << endl;
    cout << "Num of channels is: " << song.numChannels << endl;
    cout << "Size of data is: " << song.actualSize << endl;
    return 0;
}

