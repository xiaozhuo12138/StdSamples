#pragma once

// Dsp Filters
// DspFilters
#pragma once
#include "DspFilters/Elliptic.h"

namespace Filters::IIR::Elliptic
{

    struct LowPassFilter : public FilterProcessor
    {
        Dsp::Elliptic::LowPass<32> prototype;
        BiquadTypeIICascade biquads;
        size_t order;
        DspFloatType fc,sr,ripple,rolloff;
        LowPassFilter(size_t Order, DspFloatType Fc, DspFloatType ripple, DspFloatType rolloff, DspFloatType Fs)
        {            
            order = Order;
            sr    = Fs;
            this->ripple = ripple;
            this->rolloff = rolloff;
            setCutoff(Fc);            
        }
        void setCutoff(DspFloatType Fc) {
            BiquadSOS sos;            
            fc = Fc;            
            int total = order/2;
            if(order == 1) { total++; }
            sos.resize(total);
            prototype.setup(order,sr,fc,ripple,rolloff);
            for(size_t i = 0; i < total; i++) {
                Dsp::Cascade::Stage s = prototype[i];
                sos[i].z[0] = s.m_b0;
                sos[i].z[1] = s.m_b1;
                sos[i].z[2] = s.m_b2;
                sos[i].p[0] = s.m_a1;
                sos[i].p[1] = s.m_a2;
                sos[i].p[2] = 0;
            }
            biquads.setCoefficients(sos);
        }
        void setQ(DspFloatType q) {
            // not used yet
        }
        enum {
            PORT_ORDER,
            PORT_CUTOFF,
        };
        void setPort(int port, DspFloatType v) {
            switch(port) {
                case PORT_ORDER: order = (int)v; break;
                case PORT_CUTOFF: fc = v; break;
            }
            setCutoff(fc);
        }
        DspFloatType Tick(DspFloatType I, DspFloatType A=1, DspFloatType X=0, DspFloatType Y=0) {
            return biquads.Tick(I,A,X,Y);
        }
    };
    struct HighPassFilter : public FilterProcessor
    {
        Dsp::Elliptic::HighPass<32> prototype;
        BiquadTypeIICascade biquads;
        size_t order;
        DspFloatType fc,sr,ripple,rolloff;
        HighPassFilter(size_t Order, DspFloatType Fc, DspFloatType ripple, DspFloatType rolloff, DspFloatType Fs)
        {            
            order = Order;
            sr    = Fs;
            this->ripple = ripple;
            this->rolloff = rolloff;
            setCutoff(Fc);                        
        }
        void setCutoff(DspFloatType Fc) {
            BiquadSOS sos;            
            fc = Fc;            
            int total = order/2;
            if(order == 1) { total++; }
            sos.resize(total);
		prototype.setup(order,sr,fc,ripple,rolloff);
            for(size_t i = 0; i < total; i++) {
                Dsp::Cascade::Stage s = prototype[i];
                sos[i].z[0] = s.m_b0;
                sos[i].z[1] = s.m_b1;
                sos[i].z[2] = s.m_b2;
                sos[i].p[0] = s.m_a1;
                sos[i].p[1] = s.m_a2;
                sos[i].p[2] = 0;
            }
            biquads.setCoefficients(sos);
        }
        void setQ(DspFloatType q) {
            // not used yet
        }
        enum {
            PORT_ORDER,
            PORT_CUTOFF,
        };
        void setPort(int port, DspFloatType v) {
            switch(port) {
                case PORT_ORDER: order = (int)v; break;
                case PORT_CUTOFF: fc = v; break;
            }
            setCutoff(fc);
        }        
        DspFloatType Tick(DspFloatType I, DspFloatType A=1, DspFloatType X=0, DspFloatType Y=0) {
            return biquads.Tick(I,A,X,Y);
        }
    };
   struct BandPassFilter : public FilterProcessor
    {
        Dsp::Elliptic::BandPass<32> prototype;
        BiquadTypeIICascade biquads;
        size_t order;
        DspFloatType fc,sr,bw,rolloff,ripple;
        BandPassFilter(size_t Order, DspFloatType Fc, DspFloatType ripple, DspFloatType rolloff, DspFloatType BW, DspFloatType Fs)
        {            
            order = Order;
            sr    = Fs;
            bw    = BW;
            this->ripple = ripple;
            this->rolloff = rolloff;
            setCutoff(Fc);            
        }
        void setCutoff(DspFloatType Fc) {
            BiquadSOS sos;            
            fc = Fc;            
            int total = order;
            sos.resize(total);
            prototype.setup(order,sr,fc,bw,ripple,rolloff);
            for(size_t i = 0; i < total; i++) {
                Dsp::Cascade::Stage s = prototype[i];
                sos[i].z[0] = s.m_b0;
                sos[i].z[1] = s.m_b1;
                sos[i].z[2] = s.m_b2;
                sos[i].p[0] = s.m_a1;
                sos[i].p[1] = s.m_a2;
                sos[i].p[2] = 0;
            }
            biquads.setCoefficients(sos);
        }
        void setQ(DspFloatType q) {
            // not used yet
        }
        enum {
            PORT_ORDER,
            PORT_CUTOFF,
            PORT_BANDWIDTH,
        };
        void setPort(int port, DspFloatType v) {
            switch(port) {
                case PORT_ORDER: order = (int)v; break;
                case PORT_CUTOFF: fc = v; break;
                case PORT_BANDWIDTH: bw = v; break;
            }
            setCutoff(fc);
        }
        DspFloatType Tick(DspFloatType I, DspFloatType A=1, DspFloatType X=0, DspFloatType Y=0) {
            return biquads.Tick(I,A,X,Y);
        }
    };
   struct BandStopFilter : public FilterProcessor
    {
        Dsp::Elliptic::BandStop<32> prototype;
        BiquadTypeICascade biquads;
        size_t order;
        DspFloatType fc,sr,bw,rolloff,ripple;
        BandStopFilter(size_t Order, DspFloatType Fc, DspFloatType ripple, DspFloatType rolloff,DspFloatType BW, DspFloatType Fs)
        {            
            order = Order;
            sr    = Fs;
            bw    = BW;
            this->ripple = ripple;
            this->rolloff = rolloff;
            setCutoff(Fc);            
        }
        void setCutoff(DspFloatType Fc) {
            BiquadSOS sos;
            // this filter alises very badly above sr/4
            fc = Fc;            
            int total = order;
            sos.resize(total);
            prototype.setup(order,sr,fc,bw,ripple,rolloff);
            for(size_t i = 0; i < total; i++) {
                Dsp::Cascade::Stage s = prototype[i];
                sos[i].z[0] = s.m_b0;
                sos[i].z[1] = s.m_b1;
                sos[i].z[2] = s.m_b2;
                sos[i].p[0] = s.m_a1;
                sos[i].p[1] = s.m_a2;
                sos[i].p[2] = 0;
            }
            biquads.setCoefficients(sos);
        }
        void setQ(DspFloatType q) {
            // not used yet
        }
        enum {
            PORT_ORDER,
            PORT_CUTOFF,
            PORT_BANDWIDTH,
        };
        void setPort(int port, DspFloatType v) {
            switch(port) {
                case PORT_ORDER: order = (int)v; break;
                case PORT_CUTOFF: fc = v; break;
                case PORT_BANDWIDTH: bw = v; break;
            }
            setCutoff(fc);
        }
        DspFloatType Tick(DspFloatType I, DspFloatType A=1, DspFloatType X=0, DspFloatType Y=0) {
            return biquads.Tick(I,A,X,Y);
        }
    };        
}
