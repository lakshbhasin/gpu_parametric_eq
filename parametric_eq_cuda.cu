/**
 * Parametric equalizer back-end GPU code.
 */

#include <cuda_runtime.h>
#include <cufft.h>

#include "parametric_eq_cuda.cuh"

const float PI = 3.14159265358979;


/**
 * This kernel takes an array of Filters, and creates the appropriate
 * output transfer function in the frequency domain. This just involves a
 * superposition of the transfer functions described by each filter. The
 * resulting transfer function will be floor(bufSamples/2) + 1 in length,
 * since we only care about the positive frequencies in our FFT (since
 * we're FFTing real signals).
 *
 */
__global__
void cudaFilterSetupKernel(const Filter *filters,
                           const unsigned int numFilters,
                           cufftComplex *transferFunc,
                           const unsigned int samplingRate,
                           const unsigned int bufSamples)
{
    // This is the index in "transferFunc" to which we're initially writing
    // on this thread.
    unsigned int transFuncInd = blockIdx.x * blockDim.x + threadIdx.x;

    // The resolution of this transfer function will be
    // samplingRate/bufSamples (in Hz), as is standard in FFTs.
    const float transFuncRes = ((float) samplingRate) / bufSamples;

    // The length of the resulting transfer function.
    const unsigned int transFuncLength = (bufSamples / 2) + 1;

    // Each thread might have to write to multiple indices of the transfer
    // function array.
    while (transFuncInd < transFuncLength)
    {
        // The frequency of this element is just the index times the
        // resolution, but we multiply by "2 pi sqrt(-1)" to get an
        // imaginary angular frequency (as desired for s-plane transfer
        // functions).
        cufftComplex s;
        s.x = 0.0;
        s.y = (double) 2.0 * PI * transFuncInd * transFuncRes;

        // The output to store in the appropriate index of the transfer
        // function.
        cufftComplex output;
        output.x = 0.0;
        output.y = 0.0;
        
        // Iterate through all of the filters and superimpose their
        // transfer functions.
        for (int i = 0; i < numFilters; i++)
        {
            Filter thisFilter = filters[i];
            FilterType thisFilterType = thisFilter.type;
            
            cufftComplex sSq;
            double omegaNought, Q, K, omegaNoughtOvQ, omegaNoughtSq;

            switch (thisFilterType)
            {
                case FT_BAND_BOOST:
                case FT_BAND_CUT:
                    // For boosts, use the transfer function: 
                    //
                    // H(s) = (s^2 + K * omegaNought/Q * s + omegaNought^2)
                    //        / (s^2 + omegaNought/Q * s + omegaNought^2)
                    // 
                    // And use the reciprocal of this for cuts.
                    
                    omegaNought = (double) thisFilter.bandBCProp->omegaNought;
                    Q = (double) thisFilter.bandBCProp->Q;
                    K = (double) thisFilter.bandBCProp->K;

                    // Do some precomputation
                    sSq = cuCmulf(s, s);
                    omegaNoughtOvQ = omegaNought / Q;
                    omegaNoughtSq = omegaNought * omegaNought;

                    // The numerator and denominator of the above H(s) for
                    // boosts.
                    cufftComplex numerBoost;
                    cufftComplex denomBoost;

                    numerBoost.x = sSq.x + K * omegaNoughtOvQ * s.x +
                                   omegaNoughtSq;
                    numerBoost.y = sSq.y + K * omegaNoughtOvQ * s.y;

                    denomBoost.x = sSq.x + omegaNoughtOvQ * s.x +
                                   omegaNoughtSq;
                    denomBoost.y = sSq.y + omegaNoughtOvQ * s.y;

                    // If this is a boost, then just add numerBoost /
                    // denomBoost to the output element. Otherwise, if it's
                    // a cut, add the reciprocal of this.
                    cufftComplex quot;
                    
                    if (thisFilterType == FT_BAND_BOOST)
                    {
                        quot = cuCdivf(numerBoost, denomBoost);
                    }
                    else
                    {
                        quot = cuCdivf(denomBoost, numerBoost);
                    }

                    output.x += quot.x;
                    output.y += quot.y;
                    
                    break;
                
                default:
                    printf("Unknown filter type; exiting");
                    asm("trap;");
            }
        }

        // Write the "output" to global memory.
        transferFunc[transFuncInd] = output;
        
        // This thread might have to process multiple entries.
        transFuncInd += blockDim.x * gridDim.x;
    }
}


/**
 * This kernel takes an input-output FFT'd audio buffer, which is
 * floor(bufSamples/2) + 1 long (because we ignore the negative frequencies
 * in the FFT). We multiply each element in this buffer by the
 * corresponding element in the transfer function, which is assumed to be
 * of the same length. This multiplication is carried out in place.
 *
 * After multiplying, we also divide by bufSamples (since IFFT(FFT(x)) will
 * be bufSamples * x otherwise).
 *
 */
__global__
void cudaProcessBufKernel(cufftComplex *inOutAudioFFTBuf,
                          const cufftComplex *transferFunc,
                          const unsigned int bufSamples)
{
    // The index in the buffer that this thread is initially looking at.
    unsigned int bufInd = blockIdx.x * blockDim.x + threadIdx.x;

    // The FFT should be floor(bufSamples/2) + 1 elements long.
    const unsigned int fftLength = (bufSamples / 2) + 1;

    // Each thread might have to write to more than one entry in the FFT
    // buffer.
    while (bufInd < fftLength)
    {
        // Pointwise-multiply by the transfer function, and downscale.
        cufftComplex transferFuncValue = transferFunc[bufInd];
        cufftComplex newValue = inOutAudioFFTBuf[bufInd];

        newValue = cuCmulf(newValue, transferFuncValue);
        newValue.x /= bufSamples;
        newValue.y /= bufSamples;

        // Write to global memory.
        inOutAudioFFTBuf[bufInd] = newValue;
        
        // Process the next entry in the FFT buffer.
        bufInd += blockDim.x * gridDim.x;
    }
}


/**
 * This kernel takes an audio buffer of floats, and clips all of its values
 * so that they can be stored in an array of signed 16-bit shorts. 
 * 
 * It is assumed that both the input and output audio buffers are
 * "bufSamples" long.
 *
 */
__global__
void cudaClippingKernel(const float *inAudioBuf,
                        const unsigned int bufSamples,
                        int16_t *outAudioBuf)
{
    // The index in the input buffer we're initially dealing with.
    unsigned int bufInd = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread might have to write to multiple entries.
    while (bufInd < bufSamples)
    {
        float input = inAudioBuf[bufInd];

        // Use fminf and fmaxf for clamping, since these are probably
        // optimized (and allow for less divergence and better pipelining).
        float output = fminf(input, 32767.0 - 0.001);
        output = fmaxf(output, -32768.0 + 0.001);

        outAudioBuf[bufInd] = (int16_t) output;
    
        // Process the next entry in the input buffer.
        bufInd += blockDim.x * gridDim.x;
    }

}


void cudaCallFilterSetupKernel(const unsigned int blocks,
                               const unsigned int threadsPerBlock,
                               const cudaStream_t stream,
                               const Filter *filters,
                               const unsigned int numFilters,
                               cufftComplex *transferFunc,
                               const unsigned int samplingRate,
                               const unsigned int bufSamples)
{
    int shmemBytes = 0;

    cudaFilterSetupKernel<<<blocks, threadsPerBlock, shmemBytes, stream>>>
        (filters, numFilters, transferFunc, samplingRate, bufSamples);
}
                              

void cudaCallProcessBufKernel(const unsigned int blocks,
                              const unsigned int threadsPerBlock,
                              const cudaStream_t stream,
                              cufftComplex *inOutAudioFFTBuf,
                              const cufftComplex *transferFunc,
                              const unsigned int bufSamples)
{
    int shmemBytes = 0;

    cudaProcessBufKernel<<<blocks, threadsPerBlock, shmemBytes, stream>>>
        (inOutAudioFFTBuf, transferFunc, bufSamples);
}


void cudaCallClippingKernel(const unsigned int blocks,
                            const unsigned int threadsPerBlock,
                            const cudaStream_t stream,
                            const float *inAudioBuf,
                            const unsigned int bufSamples,
                            int16_t *outAudioBuf)
{
    int shmemBytes = 0;

    cudaClippingKernel<<<blocks, threadsPerBlock, shmemBytes, stream>>>
        (inAudioBuf, bufSamples, outAudioBuf);
}
