/**
 * Parametric equalizer back-end.
 */

#ifndef PARAMETRIC_EQ_CUDA_CUH
#define PARAMETRIC_EQ_CUDA_CUH

#include <cufft.h>
#include <stdio.h>
#include <stdint.h>


/**
 * This struct represents the properties of a band-pass boost or cut
 * filter. See p. 278 of the following link for more information:
 * http://www.thatcorp.com/datashts/AES13-041_Control_of_DSP_Parametric_EQs.pdf
 *
 * Note: both band-pass boost and cut filters have properties with the same
 * names, but the transfer functions of cuts are the reciprocals of those
 * of boosts. We distinguish between boost and cut filters by using their
 * FilterTypes.
 */
typedef struct BandBoostCutProp
{
    /* 
     * omegaNought = 2 pi * f_0, where f_0 is the central frequency of the
     * filter in Hz.
     */
    float omegaNought;

    /* Q = f_0/BW, where BW is the bandwidth of the filter in Hz. */
    float Q;

    /* 
     * K = 10^(G/20), where G (positive) is the desired boost or cut in dB
     * (depending on whether this is a boost or cut filter).
     */
    float K;
} BandBoostProp;


/**
 * This struct represents the properties of a high-shelving or low-shelving
 * filter. The transfer function equation for this filter is completely
 * real, and is given by the following function for low-shelf filters:
 *
 *      H(s) = 1 + (K - 1) * {1 - tanh( (|s| - Omega_0) / Omega_BW ) } / 2
 *
 * Where K = 10^(G/20) (where G here can be negative), Omega_0 = 2 * pi *
 * f_0, and Omega_BW = 2 * pi * BW. The transfer function for high-shelf
 * filters is identical, but the argument to the tanh is negated.
 *
 * Note that this is mostly a heuristic filter that Laksh came up with.
 * There's no guarantees that it has nice properties about roll-offs etc.
 * However, there weren't very many nicely parametrizable filters to use
 * for low-shelving and high-shelving filters.
 */
typedef struct ShelvingProp
{
    /*
     * omegaNought = 2 * pi * f_0, where f_0 is the central frequency of
     * the filter in Hz.
     */
    float omegaNought;

    /* 
     * omegaBW = 2 * pi * BW, where BW is the bandwidth of the filter in
     * Hz. (This is technically the "width" over which the filter's
     * passband transitions to the stopband; it's not a "bandwidth" in the
     * ordinary sense.
     */
    float omegaBW;

    /*
     * K = 10^(G/20), where G (can be positive or negative) is the desired
     * gain in dB.
     */
    float K;
} ShelvingProp;


/**
 * This enumeration specifies all the different kinds of filters allowed in
 * this parametric equalizer. It is used to "tag" filters.
 */
typedef enum FilterType
{
    FT_BAND_BOOST,      /* Band-pass boost filter. */
    FT_BAND_CUT,        /* Band-pass cut filter. */
    FT_HIGH_SHELF,      /* High-shelving filter. */
    FT_LOW_SHELF        /* Low-shelving filter. */
} FilterType;


/**
 * A struct that represents a generic filter, including its type and a
 * pointer to a properties struct (representing that filter's specific
 * properties).
 */
typedef struct Filter
{
    /* The type of this filter. */
    FilterType type;

    /*
     * A union of all the different possible properties that this filter
     * could have. These are pointers to structs with more information. The
     * specific property struct to use is indicated by the type tag.
     */
    union 
    {
        BandBoostCutProp *bandBCProp;    /* FT_BAND_BOOST, FT_BAND_CUT */
        ShelvingProp *shelvingProp;      /* FT_HIGH_SHELF, FT_LOW_SHELF */
    };

} Filter;


/**
 * This function calls a kernel that will take an array of filters
 * (representing the parametric equalizer) and construct a new transfer
 * function at "transferFunc" of length floor(bufSamples/2) + 1 (the number
 * of samples in each buffer divided by two, since we are only considering
 * positive frequencies in the FFT since we have real signals). This 
 * kernel is used to update the filter data stored on the GPU, before
 * the next buffer of data is passed into the equalizer.
 *
 */
void cudaCallFilterSetupKernel(const unsigned int blocks,
                               const unsigned int threadsPerBlock,
                               const cudaStream_t stream,
                               const Filter *filters,
                               const unsigned int numFilters,
                               cufftComplex *transferFunc,
                               const unsigned int samplingRate,
                               const unsigned int bufSamples);


/**
 * This function calls a kernel that will process a given buffer of audio
 * after it has been FFT'd. The input audio buffer will only contain a
 * *single* channel's samples post-FFT.
 *
 * This processing involves multiplication by a transfer function, as well
 * as downscaling by bufSamples (because of the properties of the FFT).
 * This is all carried out in place, using the input-output parameter
 * inOutAudioFFTBuf.
 * 
 */
void cudaCallProcessBufKernel(const unsigned int blocks,
                              const unsigned int threadsPerBlock,
                              const cudaStream_t stream,
                              cufftComplex *inOutAudioFFTBuf,
                              const cufftComplex *transferFunc,
                              const unsigned int bufSamples);


/**
 * This function calls a kernel that clips all values in a given array of
 * floats, so that they can be stored as an array of signed 16-bit shorts.
 * Note that the input array contains bufSamples entries (i.e. it only
 * contains the data needed for one channel).
 *
 */
void cudaCallClippingKernel(const unsigned int blocks,
                            const unsigned int threadsPerBlock,
                            const cudaStream_t stream,
                            const float *inAudioBuf,
                            const unsigned int bufSamples,
                            int16_t *outAudioBuf);



#endif  // PARAMETRIC_EQ_CUDA_CUH
