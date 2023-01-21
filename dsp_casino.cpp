
#include "DSP/dsp_samples_std.hpp"
#include "DSP/dsp_fftw.hpp"
#include "Core/core_sndfile.hpp"

//#include "DSP/dsp_adsr.hpp"
//#include "DSP/dsp_audiofft.hpp"
//#include "DSP/dsp_decimators.hpp"
//#include "DSP/dsp_envelope_detector.hpp"
//#include "DSP/dsp_fftconvolver.hpp"
//#include "DSP/dsp_fftwpp.hpp"
//#include "DSP/dsp_gist.hpp"
//#include "DSP/dsp_ir_convolution.hpp"
//#include "DSP/dsp_kissfft_fastfir.hpp"
//#include "DSP/dsp_minfft.hpp"
//#include "DSP/dsp_pitch_detection.hpp"
//#include "DSP/dsp_polyblep.hpp"
//#include "DSP/dsp_samplerate_calculus.hpp"
//#include "DSP/dsp_samples_dsp.hpp"
//#include "DSP/dsp_samples_eigen.hpp"
//#include "DSP/dsp_simple_resampler.hpp"
//#include "DSP/dsp_spectrum.hpp"

/*
    ////////////////////////////////////////////////////////////////
    // FFTW Convolution
    ////////////////////////////////////////////////////////////////    
    sample_vector<float> convolution(sample_vector<float> x, sample_vector<float> y) {
        int M = x.size();
        int N = y.size();       
        float in_a[M+N-1];
        complex_vector<float> out_a(M+N-1);
        float in_b[M+N-1];
        complex_vector<float> out_b(M+N-1);
        complex_vector<float> in_rev(M+N-1);
        sample_vector<float> out_rev(M+N-1);

        // Plans for forward FFTs
        fftwf_plan plan_fwd_a = fftwf_plan_dft_r2c_1d (M+N-1, in_a, reinterpret_cast<fftwf_complex*>(&out_a), FFTW_MEASURE);
        fftwf_plan plan_fwd_b = fftwf_plan_dft_r2c_1d (M+N-1, in_b, reinterpret_cast<fftwf_complex*>(&out_b), FFTW_MEASURE);

        // Plan for reverse FFT
        fftwf_plan plan_rev = fftwf_plan_dft_c2r_1d (M+N-1,reinterpret_cast<fftwf_complex*>(&in_rev), out_rev.data(), FFTW_MEASURE);

        // Prepare padded input data
        std::memcpy(in_a, x.data(), sizeof(float) * M);
        std::memset(in_a + M, 0, sizeof(float) * (N-1));
        std::memset(in_b, 0, sizeof(float) * (M-1));
        std::memcpy(in_b + (M-1), y.data(), sizeof(float) * N);
        
        // Calculate the forward FFTs
        fftwf_execute(plan_fwd_a);
        fftwf_execute(plan_fwd_b);

        // Multiply in frequency domain
        for( int idx = 0; idx < M+N-1; idx++ ) {
            in_rev[idx] = out_a[idx] * out_b[idx];
        }

        // Calculate the backward FFT
        fftwf_execute(plan_rev);

        // Clean up
        fftwf_destroy_plan(plan_fwd_a);
        fftwf_destroy_plan(plan_fwd_b);
        fftwf_destroy_plan(plan_rev);

        return out_rev;
    }

    void blockconvolve(sample_vector<float> h, sample_vector<float> x, sample_vector<float>& y, sample_vector<float> & ytemp)    
    {
        int i;
        int M = h.size();
        int L = x.size();
        y = convolution(h,x);      
        for (i=0; i<M; i++) {
            y[i] += ytemp[i];                     /* add tail of previous block */
            ytemp[i] = y[i+L];                    /* update tail for next call */
        }        
    }


    ////////////////////////////////////////////////////////////////
    // FFTW Deconvolution
    ////////////////////////////////////////////////////////////////
    sample_vector<float> deconvolution(sample_vector<float> & xin, sample_vector<float> & yout)
    {
        int M = xin.size();
        int N = yout.size();
        float x[M] = {0,1,0,0};
        float y[N] = {0,0,1,0,0,0,0,0};
        float in_a[M+N-1];
        std::complex<float> out_a[M+N-1];
        float in_b[M+N-1];
        std::complex<float> out_b[M+N-1];
        std::complex<float> in_rev[M+N-1];
        sample_vector<float> out_rev(M+N-1);

        // Plans for forward FFTs
        fftwf_plan plan_fwd_a = fftwf_plan_dft_r2c_1d (M+N-1, in_a, reinterpret_cast<fftwf_complex*>(&out_a), FFTW_MEASURE);
        fftwf_plan plan_fwd_b = fftwf_plan_dft_r2c_1d (M+N-1, in_b, reinterpret_cast<fftwf_complex*>(&out_b), FFTW_MEASURE);

        // Plan for reverse FFT
        fftwf_plan plan_rev = fftwf_plan_dft_c2r_1d (M+N-1,reinterpret_cast<fftwf_complex*>(&in_rev[0]), out_rev.data(), FFTW_MEASURE);

        // Prepare padded input data
        std::memcpy(in_a, xin.data(), sizeof(float) * M);
        std::memset(in_a + M, 0, sizeof(float) * (N-1));
        std::memset(in_b, 0, sizeof(float) * (M-1));
        std::memcpy(in_b + (M-1), yout.data(), sizeof(float) * N);
        
        // Calculate the forward FFTs
        fftwf_execute(plan_fwd_a);
        fftwf_execute(plan_fwd_b);

        // Multiply in frequency domain
        for( int idx = 0; idx < M+N-1; idx++ ) {
            in_rev[idx] = out_a[idx] / out_b[idx];
        }

        // Calculate the backward FFT
        fftwf_execute(plan_rev);

        // Clean up
        fftwf_destroy_plan(plan_fwd_a);
        fftwf_destroy_plan(plan_fwd_b);
        fftwf_destroy_plan(plan_rev);

        return out_rev;
    }

    ////////////////////////////////////////////////////////////////
    // FFTW Cross Correlation
    ////////////////////////////////////////////////////////////////
    sample_vector<float> xcorrelation(sample_vector<float> & xin, sample_vector<float> & yout)
    {
        int M = xin.size();
        int N = yout.size();        
        float in_a[M+N-1];
        std::complex<float> out_a[M+N-1];
        float in_b[M+N-1];
        std::complex<float> out_b[M+N-1];
        std::complex<float> in_rev[M+N-1];
        sample_vector<float> out_rev(M+N-1);

        // Plans for forward FFTs
        fftwf_plan plan_fwd_a = fftwf_plan_dft_r2c_1d (M+N-1, in_a, reinterpret_cast<fftwf_complex*>(&out_a), FFTW_MEASURE);
        fftwf_plan plan_fwd_b = fftwf_plan_dft_r2c_1d (M+N-1, in_b, reinterpret_cast<fftwf_complex*>(&out_b), FFTW_MEASURE);

        // Plan for reverse FFT
        fftwf_plan plan_rev = fftwf_plan_dft_c2r_1d (M+N-1,reinterpret_cast<fftwf_complex*>(&in_rev[0]), out_rev.data(), FFTW_MEASURE);

        // Prepare padded input data
        std::memcpy(in_a, xin.data(), sizeof(float) * M);
        std::memset(in_a + M, 0, sizeof(float) * (N-1));
        std::memset(in_b, 0, sizeof(float) * (M-1));
        std::memcpy(in_b + (M-1), yout.data(), sizeof(float) * N);
        
        // Calculate the forward FFTs
        fftwf_execute(plan_fwd_a);
        fftwf_execute(plan_fwd_b);

            // Multiply in frequency domain
        for( int idx = 0; idx < M+N-1; idx++ ) {
            in_rev[idx] = std::conj(out_a[idx]) * out_b[idx]/(float)(M+N-1);
        }

        // Calculate the backward FFT
        fftwf_execute(plan_rev);

        // Clean up
        fftwf_destroy_plan(plan_fwd_a);
        fftwf_destroy_plan(plan_fwd_b);
        fftwf_destroy_plan(plan_rev);

        return out_rev;
    }

    

    ////////////////////////////////////////////////////////////////
    // FFTW Resampler
    ////////////////////////////////////////////////////////////////

    struct FFTResampler
    {
        int inFrameSize;
        int inWindowSize;
        int inSampleRate;
        float *inWindowing;
        fftwf_plan inPlan;
        int outFrameSize;
        int outWindowSize;
        int outSampleRate;
        float *outWindowing;
        fftwf_plan outPlan;
        float *inFifo;
        float *synthesisMem;
        fftwf_complex *samples;
        int pos;
        

        FFTResampler(size_t inSampleRate, size_t outSampleRate, size_t nFFT)
        {
            
            pos = 0;
            if (outSampleRate < inSampleRate) {
                nFFT = nFFT * inSampleRate * 128 / outSampleRate;
            }
            else {
                nFFT = nFFT * outSampleRate * 128 / inSampleRate;
            }
            nFFT += (nFFT % 2);

            inFrameSize = nFFT;
            inWindowSize = nFFT * 2;
            inSampleRate = inSampleRate;
            outSampleRate = outSampleRate;
            outFrameSize = inFrameSize * outSampleRate / inSampleRate;
            outWindowSize = outFrameSize * 2;        

            outWindowing = (float *) fftwf_alloc_real(outFrameSize);
            inFifo = (float *) fftwf_alloc_real(std::max(inWindowSize, outWindowSize));
            samples = (fftwf_complex *) fftwf_alloc_complex(std::max(inWindowSize, outWindowSize));
            inWindowing = (float *) fftwf_alloc_real(inFrameSize);
            synthesisMem = (float *) fftwf_alloc_real(outFrameSize);
                    
            inPlan = fftwf_plan_dft_r2c_1d(inWindowSize,inFifo,samples,FFTW_ESTIMATE);        
            outPlan = fftwf_plan_dft_c2r_1d(outWindowSize,samples,synthesisMem,FFTW_ESTIMATE);
            
            if ((inFifo == NULL) || (inPlan == NULL) || (outPlan == NULL)
                || (samples == NULL)
                || (synthesisMem == NULL) || (inWindowing == NULL) || (outWindowing == NULL)
                ) {
                    assert(1==0);
            }
            float norm = 1.0f / inWindowSize;
            for (size_t i = 0; i < inFrameSize; i++) {
                double t = std::sin(.5 * M_PI * (i + .5) / inFrameSize);
                inWindowing[i] = (float) std::sin(.5 * M_PI * t * t) * norm;
            }
            for (size_t i = 0; i < outFrameSize; i++) {
                double t = std::sin(.5 * M_PI * (i + .5) / outFrameSize);
                outWindowing[i] = (float) std::sin(.5 * M_PI * t * t);
            }    
        }
        
        ~FFTResampler()
        {   
            if (inFifo) {
                free(inFifo);
                inFifo = NULL;
            }

            if (inPlan) {
                fftwf_destroy_plan(inPlan);
                inPlan = NULL;
            }

            if (outPlan) {
                fftwf_destroy_plan(outPlan);
                outPlan = NULL;
            }

            if (samples) {
                fftw_free(samples);
                samples = NULL;
            }

            if (synthesisMem) {
                fftw_free(synthesisMem);
                synthesisMem = NULL;
            }

            if (inWindowing) {
                fftw_free(inWindowing);
                inWindowing = NULL;
            }

            if (outWindowing) {
                fftw_free(outWindowing);
                outWindowing = NULL;
            }    
        }

        void reset()
        {        
            pos = 0;
        }

        

        int Process(const float *input, float *output)
        {
            if ((input == NULL) || (output == NULL)) {
                return -1;
            }
            float *inFifo = inFifo;
            float *synthesis_mem = synthesisMem;
            for (size_t i = 0; i < inFrameSize; i++) {
                inFifo[i] *= inWindowing[i];
                inFifo[inWindowSize - 1 - i] = input[inFrameSize - 1 - i] * inWindowing[i];
            }
            fftwf_execute(inPlan);
            if (outWindowSize < inWindowSize) {
                int half_output = (outWindowSize / 2);
                int diff_size = inWindowSize - outWindowSize;
                memset(samples + half_output, 0, diff_size * sizeof(fftw_complex));
            }
            else if (outWindowSize > inWindowSize) {
                int half_input = inWindowSize / 2;
                int diff_size = outWindowSize - inWindowSize;
                memmove(samples + half_input + diff_size, samples + half_input,
                        half_input * sizeof(fftw_complex));
                memset(samples + half_input, 0, diff_size * sizeof(fftw_complex));
            }
            fftwf_execute(outPlan);
            for (size_t i = 0; i < outFrameSize; i++) {
                output[i] = inFifo[i] * outWindowing[i] + synthesis_mem[i];
                inFifo[outWindowSize - 1 - i] *= outWindowing[i];
            }
            memcpy(synthesis_mem, inFifo + outFrameSize, outFrameSize * sizeof(float));
            memcpy(inFifo, input, inFrameSize * sizeof(float));
            if (pos == 0) {
                pos++;
                return 0;
            }
            pos++;
            return 1;
        }
    };
*/

using namespace Casino;


template<typename T>
std::ostream& operator << (std::ostream& o, const std::vector<T> & v)
{
    for(auto i : v) o << i << ",";
    return o;
}
    
void p1()
{
    std::vector<std::complex<float>> v(16),r(16),x(16);

    for(size_t i = 0; i < 16; i++) v[i] = std::complex<float>(i,0);
    FFTPlanComplexFloat fftPlan(16);
    fft(fftPlan,v.data(),r.data());
    ifft(fftPlan,r.data(),x.data());
    for(size_t i = 0; i < x.size(); i++) 
        std::cout << x[i] << ",";
    std::cout << std::endl;
}
void p2()
{
    std::vector<std::complex<double>> v(16),r(16),x(16);

    for(size_t i = 0; i < 16; i++) v[i] = std::complex<double>(i,0);
    FFTPlanComplexDouble fftPlan(16);
    fft(fftPlan,v.data(),r.data());
    ifft(fftPlan,r.data(),x.data());
    for(size_t i = 0; i < x.size(); i++) 
        std::cout << x[i] << ",";
    std::cout << std::endl;    
}

template<typename T>
std::ostream& operator << (std::ostream & o, const sample_vector<T> & v)
{
    for(auto i : v) o << i << ",";
    o << std::endl;
    return o;
}


void convtest()
{

    SndFileReaderDouble halls("01 Halls 1 Large Hall.wav");
    SndFileReaderDouble baby("AcGtr.wav");      
        
    std::vector<double> x1(halls.size()),f2(baby.size());
    
    halls.read(x1.size(),x1.data());
    baby.read(f2.size(),f2.data());
    
    std::vector<double> f1(halls.size()/2);
    for(size_t i = 0; i < halls.size()/2; i++) f1[i] = x1[i*2];
    
    FFTConvolutionDouble convolver(f1.size(),f1.data(),f2.size());
    std::vector<double> out(f1.size() + f2.size() -1);
    convolver.ProcessBlock(f2.size(),f2.data(),out.data());
    SndFileWriterFloat output("test.wav",0x10006,baby.channels(),baby.samplerate());
    std::vector<float> temp(out.size());
    
    for(size_t i = 0; i < temp.size(); i++) temp[i] = out[i];
    output.write(temp.size(),temp.data());

}

void vec1()
{
    sample_vector<float> v(10),b(10),c;
    int i = 0;
    std::generate(v.begin(),v.end(),[&i](){return i++; });
    //std::for_each(v.begin(),v.end(),[](float &x){ x = x*x; });    
    i = 10;
    std::generate(b.begin(),b.end(),[&i](){return i+=10; });
    c = v + b;
    std::cout << c << std::endl;
}
void interleaving()
{
    sample_vector<float> v(20),r;
    sample_vector<sample_vector<float>> m;
    int i = 0;
    std::generate(v.begin(),v.end(),[&i](){ return i=!i; });
    std::cout << v << std::endl;
    m = deinterleave(10,2,v);
    std::cout << m[0] << std::endl;
    std::cout << m[1] << std::endl;
    r = interleave(10,2,m);
    std::cout << r << std::endl;
}

int main()
{
    sample_vector<float> v(20),a(10),b(10),r;
    sample_vector<sample_vector<float>> m;
    int i = 0;
    std::generate(v.begin(),v.end(),[&i](){ return i=!i; });
    std::cout << v << std::endl;
    split_stereo(v,a,b);
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    swap(a,b);
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    i = 0;
    std::generate(v.begin(),v.end(),[&i](){ return i++; });
    cshift(v,-25);    
    std::cout << v << std::endl;
    std::cout << mean(v) << std::endl;
    std::cout << sum(v) << std::endl;
    std::cout << min(v) << std::endl;
    std::cout << max(v) << std::endl;
    i = 0;
    std::generate(v.begin(),v.end(),[&i](){ return i++; });
    shift(v);
    std::cout << v << std::endl;

}