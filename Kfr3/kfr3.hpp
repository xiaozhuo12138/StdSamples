#pragma once

#include <vector>
#include <cmath>
#include <complex>

// will be huge problem with c++
#define KFR_NO_C_COMPLEX_TYPES
#include <kfr/capi.h>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>

#include "kfr3_samples.hpp"
#include "kfr3_capi.hpp"

namespace kfr3
{
    template<typename T>
    sample_vector<T> dcremove(const sample_vector<T> & input, T cutoff) {
         kfr::univector<T> out(input.size());
        auto x = kfr::dcremove(input,cutoff);
        x.apply(input,out);
        return make_univec(out);
    }

    template<typename T>
    sample_vector<T> 
    cheby1_lowpass(int order, T w, T cutoff, T sample_rate) {
        kfr::zpk<T> filt = kfr::iir_lowpass(kfr::chebyshev1<T>(order,w),cutoff,sample_rate);        
        sample_vector<kfr::biquad_params<T>> bqs = kfr::to_sos<T>(filt);
        sample_vector<T> r(bqs.size()*6);
        size_t x = 0;
        for(size_t i = 0; i < bqs.size(); i)
        {            
            r[x++] = bqs[i].b0;
            r[x++] = bqs[i].b1;
            r[x++] = bqs[i].b2;
            r[x++] = bqs[i].a0;
            r[x++] = bqs[i].a1;
            r[x++] = bqs[i].a2;
        }            
        return r;
    }

    template<typename T>
    sample_vector<T> 
    cheby1_highpass(int order, T w, T cutoff, T sample_rate) {
        kfr::zpk<T> filt = kfr::iir_highpass(kfr::chebyshev1<T>(order,w),cutoff,sample_rate);        
        sample_vector<kfr::biquad_params<T>> bqs = kfr::to_sos<T>(filt);
        sample_vector<T> r(bqs.size()*6);
        size_t x = 0;
        for(size_t i = 0; i < bqs.size(); i)
        {            
            r[x++] = bqs[i].b0;
            r[x++] = bqs[i].b1;
            r[x++] = bqs[i].b2;
            r[x++] = bqs[i].a0;
            r[x++] = bqs[i].a1;
            r[x++] = bqs[i].a2;
        }            
        return r;
    }

    template<typename T>
    sample_vector<T> 
    cheby1_bandstop(int order, T w, T locutoff, T hicutoff, T sample_rate) {
        kfr::zpk<T> filt = kfr::iir_bandstop(kfr::chebyshev1<T>(order,w),locutoff,hicutoff,sample_rate);        
        sample_vector<kfr::biquad_params<T>> bqs = kfr::to_sos<T>(filt);
        sample_vector<T> r(bqs.size()*6);
        size_t x = 0;
        for(size_t i = 0; i < bqs.size(); i)
        {            
            r[x++] = bqs[i].b0;
            r[x++] = bqs[i].b1;
            r[x++] = bqs[i].b2;
            r[x++] = bqs[i].a0;
            r[x++] = bqs[i].a1;
            r[x++] = bqs[i].a2;
        }            
        return r;
    }

    template<typename T>
    sample_vector<T> 
    cheby1_bandpass(int order, T w, T locutoff, T hicutoff, T sample_rate) {
        kfr::zpk<T> filt = kfr::iir_bandpass(kfr::chebyshev1<T>(order,w),locutoff,hicutoff,sample_rate);        
        sample_vector<kfr::biquad_params<T>> bqs = kfr::to_sos<T>(filt);
        sample_vector<T> r(bqs.size()*6);
        size_t x = 0;
        for(size_t i = 0; i < bqs.size(); i)
        {            
            r[x++] = bqs[i].b0;
            r[x++] = bqs[i].b1;
            r[x++] = bqs[i].b2;
            r[x++] = bqs[i].a0;
            r[x++] = bqs[i].a1;
            r[x++] = bqs[i].a2;
        }            
        return r;
    }

    template<typename T>
    struct Resampler 
    {
        kfr::samplerate_converter<T> * sc;

        using quality = kfr::sample_rate_conversion_quality;

        Resampler(quality q, int64_t interp_factor, int64_t decimation_factor, T scale=1.0, T cutoff=0.5)
        {
            sc = new kfr::samplerate_converter<T>(q,interp_factor,decimation_factor,scale,cutoff);
        }
        ~Resampler()
        {
            if(sc) delete sc;
        }

        size_t ProcessBlock(size_t n, T * in, T * out)
        {
            kfr::univector<T> I(n);
            kfr::univector<T> O(n);
            memcpy(I.data(),in,n*sizeof(T));
            memcpy(O.data(),out,n*sizeof(T));
            size_t r = sc->process(O,I);
            memcpy(out,O.data(),n*sizeof(T));
            return r;
        }
        size_t process(sample_vector<T> & in, sample_vector<T> & out)
        {
            return sc->process(in,out);
        }
    };

    template<typename T>
    size_t resample(Resampler<T>& r, sample_vector<T> & in, sample_vector<T> & out)
    {
        return r.process(in,out);
    }      
    template<typename T>
    size_t resample(Resampler<T> & r, size_t n, T * in, T * out)
    {
        return r.ProcessBlock(n,in,out);
    }  

}