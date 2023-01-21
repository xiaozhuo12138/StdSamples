
#include "DSP/dsp_samples_std.hpp"
#include "DSP/dsp_fftw.hpp"

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

using namespace Casino;

/*
void xcorr(fftw_complex * signala, fftw_complex * signalb, fftw_complex * result, int N)
{
    fftw_complex * signala_ext = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (2 * N - 1));
    fftw_complex * signalb_ext = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (2 * N - 1));
    fftw_complex * out_shifted = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (2 * N - 1));
    fftw_complex * outa = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (2 * N - 1));
    fftw_complex * outb = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (2 * N - 1));
    fftw_complex * out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (2 * N - 1));

    fftw_plan pa = fftw_plan_dft_1d(2 * N - 1, signala_ext, outa, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pb = fftw_plan_dft_1d(2 * N - 1, signalb_ext, outb, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan px = fftw_plan_dft_1d(2 * N - 1, out, result, FFTW_BACKWARD, FFTW_ESTIMATE);

    //zeropadding
    memset (signala_ext, 0, sizeof(fftw_complex) * (N - 1));
    memcpy (signala_ext + (N - 1), signala, sizeof(fftw_complex) * N);
    memcpy (signalb_ext, signalb, sizeof(fftw_complex) * N);
    memset (signalb_ext + N, 0, sizeof(fftw_complex) * (N - 1));

    fftw_execute(pa);
    fftw_execute(pb);

    fftw_complex scale = 1.0/(2 * N -1);
    for (int i = 0; i < 2 * N - 1; i++)
        out[i] = outa[i] * conj(outb[i]) * scale;

    fftw_execute(px);

    fftw_destroy_plan(pa);
    fftw_destroy_plan(pb);
    fftw_destroy_plan(px);

    fftw_free(signala_ext);
    fftw_free(signalb_ext);
    fftw_free(out_shifted);
    fftw_free(out);
    fftw_free(outa);
    fftw_free(outb);

    fftw_cleanup();

    return;
}
*/
/* xcorr
// We want to calculate the crosscorrelation between x and y like:
    // xcorr(x,y) = ifft(conj(fft([x 0]) .* fft([0 y])))
    // peak in result should move left<->right as we shift y left<->right

    int M = 4;
    int N = 8;
    float x[M] = {0,1,0,0};
    float y[N] = {0,0,1,0,0,0,0,0};
    float in_a[M+N-1];
    std::complex<float> out_a[M+N-1];
    float in_b[M+N-1];
    std::complex<float> out_b[M+N-1];
    std::complex<float> in_rev[M+N-1];
    float out_rev[M+N-1];

    // Plans for forward FFTs
    fftwf_plan plan_fwd_a = fftwf_plan_dft_r2c_1d (M+N-1, in_a,
        reinterpret_cast<fftwf_complex*>(&out_a), FFTW_MEASURE);
    fftwf_plan plan_fwd_b = fftwf_plan_dft_r2c_1d (M+N-1, in_b,
        reinterpret_cast<fftwf_complex*>(&out_b), FFTW_MEASURE);

    // Plan for reverse FFT
    fftwf_plan plan_rev = fftwf_plan_dft_c2r_1d (M+N-1,
        reinterpret_cast<fftwf_complex*>(&in_rev), out_rev, FFTW_MEASURE);

    // Prepare padded input data
    std::memcpy(in_a, x, sizeof(float) * M);
    std::memset(in_a + M, 0, sizeof(float) * (N-1));
    std::memset(in_b, 0, sizeof(float) * (M-1));
    std::memcpy(in_b + (M-1), y, sizeof(float) * N);

     for( int idx = 0; idx < M+N-1; idx++ ) {
        std::cout << in_a[idx] << " ";
    }
    std::cout << std::endl;
    for( int idx = 0; idx < M+N-1; idx++ ) {
        std::cout << in_b[idx] << " ";
    }
    std::cout << std::endl;

    // Calculate the forward FFTs
    fftwf_execute(plan_fwd_a);
    fftwf_execute(plan_fwd_b);

    // Multiply in frequency domain
    for( int idx = 0; idx < M+N-1; idx++ ) {
        in_rev[idx] = std::conj(out_a[idx]) * out_b[idx]/(float)(M+N-1);
    }

    // Calculate the backward FFT
    fftwf_execute(plan_rev);

    for( int idx = 0; idx < M+N-1; idx++ ) {
        std::cout << out_rev[idx] << " ";
    }
    std::cout << std::endl;

    // Clean up
    fftwf_destroy_plan(plan_fwd_a);
    fftwf_destroy_plan(plan_fwd_b);
    fftwf_destroy_plan(plan_rev);

    return 0;
*/

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

int main()
{
    sample_vector<float> r(100);
    r.fill(1.0);
    std::cout << r << std::endl;
}