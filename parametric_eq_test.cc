/**
 * A simple command-line test of the parametric EQ. This just carries out a
 * band-pass boost filter, whose properties are specified in main().
 *
 * Usage:
 *      ./parametric_eq_test <wav_file> <samp_per_buf> 
 *                           <threads_per_block> <max_num_blocks>
 *
 * Where "samp_per_buf" is the number of samples per channel in each
 * buffer.
 *
 */

#include "parametric_eq.hh"
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>


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


int main(int argc, char *argv[])
{
    /* Argument parsing */

    // There should be 4 arguments.
    if (argc != 5)
    {
        usage(argv[0]);
    }

    // Test a single band-pass boost filter.
    Filter *filters = (Filter *) malloc(1 * sizeof(Filter));

    float freq = 300.0;          // Hz
    float bandwidth = 100.0;     // Hz
    float gain = 15.0;           // dB (must be positive)

    BandBoostCutProp *bandBCProp = (BandBoostCutProp *)
        malloc(sizeof(BandBoostCutProp));
    bandBCProp->omegaNought = 2.0 * M_PI * freq;
    bandBCProp->Q = freq/bandwidth;
    bandBCProp->K = std::pow(10.0, gain/20.0);

    filters[0].type = FT_BAND_BOOST;
    filters[0].bandBCProp = bandBCProp;

    // Construct the Parametric EQ
    ParametricEQ *thisEQ = new ParametricEQ(1, filters);

    // Load the given song's data, and set parameters related to the number
    // of samples per buffer and GPU.
    char *fileName = argv[1];
    uint32_t numBufSamples = (uint32_t) atoi(argv[2]);
    uint16_t threadsPerBlock = (uint16_t) atoi(argv[3]);
    uint16_t maxBlocks = (uint16_t) atoi(argv[4]);

    thisEQ->setSong(fileName);
    thisEQ->setNumBufSamples(numBufSamples);
    thisEQ->setThreadsPerBlock(threadsPerBlock);
    thisEQ->setMaxBlocks(maxBlocks);

    // Start processing sound.
    thisEQ->startProcessingSound();

    // Free memory
    delete thisEQ;

    free(bandBCProp);
    free(filters);

}
