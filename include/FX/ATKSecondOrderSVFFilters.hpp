#pragma once

#include "ATK.hpp"

namespace Filters::AudioTK::EQ
{  
    struct SVFLowPassFilter: public ATKFilter
    {             
        ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFLowPassCoefficients<DspFloatType>> * filter;

        SVFLowPassFilter() : ATKFilter()
        {
            filter = new ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFLowPassCoefficients<DspFloatType>>(2);
        }
        ~SVFLowPassFilter() {
            if(filter) delete filter;
        }
        enum {
            PORT_CUTOFF,
            PORT_Q,             
        };
        void setPort(int port, DspFloatType v) {
            switch(port)
            {
                case PORT_CUTOFF: filter->set_cut_frequency(v); break;
                case PORT_Q: filter->set_Q(v); break;                
            }
        }
    };
    struct SVFBandPassFilter: public ATKFilter
    {
        ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFBandPassCoefficients<DspFloatType>> * filter;

        SVFBandPassFilter() : ATKFilter()
        {
            filter = new ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFBandPassCoefficients<DspFloatType>>(2);
        }
        ~SVFBandPassFilter() {
            if(filter) delete filter;
        }
        enum {
            PORT_CUTOFF,
            PORT_Q,             
        };
        void setPort(int port, DspFloatType v) {
            switch(port)
            {
                case PORT_CUTOFF: filter->set_cut_frequency(v); break;
                case PORT_Q: filter->set_Q(v); break;                
            }
        }
    };
    struct SVFHighPassFilter: public ATKFilter
    {
        ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFHighPassCoefficients<DspFloatType>> * filter;

        SVFHighPassFilter() : ATKFilter()
        {
            filter = new ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFHighPassCoefficients<DspFloatType>>(2);
        }
        ~SVFHighPassFilter() {
            if(filter) delete filter;
        }
        enum {
            PORT_CUTOFF,
            PORT_Q,             
        };
        void setPort(int port, DspFloatType v) {
            switch(port)
            {
                case PORT_CUTOFF: filter->set_cut_frequency(v); break;
                case PORT_Q: filter->set_Q(v); break;                
            }
        }
    };
    struct SVFNotchFilter: public ATKFilter
    {
        ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFNotchCoefficients<DspFloatType>> * filter;

        SVFNotchFilter() : ATKFilter()
        {
            filter = new ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFNotchCoefficients<DspFloatType>>(2);
        }
        ~SVFNotchFilter() {
            if(filter) delete filter;
        }
        enum {
            PORT_CUTOFF,
            PORT_Q,             
        };
        void setPort(int port, DspFloatType v) {
            switch(port)
            {
                case PORT_CUTOFF: filter->set_cut_frequency(v); break;
                case PORT_Q: filter->set_Q(v); break;                
            }
        }
    };
    struct SVFPeakFilter: public ATKFilter
    {
        ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFPeakCoefficients<DspFloatType>> * filter;

        SVFPeakFilter() : ATKFilter()
        {
            filter = new ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFPeakCoefficients<DspFloatType>>(2);
        }
        ~SVFPeakFilter() {
            if(filter) delete filter;
        }
        enum {
            PORT_CUTOFF,
            PORT_Q,             
        };
        void setPort(int port, DspFloatType v) {
            switch(port)
            {
                case PORT_CUTOFF: filter->set_cut_frequency(v); break;
                case PORT_Q: filter->set_Q(v); break;                
            }
        }
    };
    struct SVFBellFilter: public ATKFilter
    {
        ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFBellCoefficients<DspFloatType>> * filter;

        SVFBellFilter() : ATKFilter()
        {
            filter = new ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFBellCoefficients<DspFloatType>>(2);
        }
        ~SVFBellFilter() {
            if(filter) delete filter;
        }
        enum {
            PORT_CUTOFF,
            PORT_Q,             
            PORT_GAIN,
        };
        void setPort(int port, DspFloatType v) {
            switch(port)
            {
                case PORT_CUTOFF: filter->set_cut_frequency(v); break;
                case PORT_Q: filter->set_Q(v); break;                
                case PORT_GAIN: filter->set_gain(v); break;
            }
        }
    };

    struct SVFLowShelfFilter: public ATKFilter
    {
        ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFLowShelfCoefficients<DspFloatType>> * filter;

        SVFLowShelfFilter() : ATKFilter()
        {
            filter = new ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFLowShelfCoefficients<DspFloatType>>(2);
        }
        ~SVFLowShelfFilter() {
            if(filter) delete filter;
        }
        enum {
            PORT_CUTOFF,
            PORT_Q,             
            PORT_GAIN,
        };
        void setPort(int port, DspFloatType v) {
            switch(port)
            {
                case PORT_CUTOFF: filter->set_cut_frequency(v); break;
                case PORT_Q: filter->set_Q(v); break;                
                case PORT_GAIN: filter->set_gain(v); break;
            }
        }
    };

    struct SVFHighShelfFilter: public ATKFilter
    {
        ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFHighShelfCoefficients<DspFloatType>> * filter;

        SVFHighShelfFilter() : ATKFilter()
        {
            filter = new ATK::SecondOrderSVFFilter<ATK::SecondOrderSVFHighShelfCoefficients<DspFloatType>>(2);
        }
        ~SVFHighShelfFilter() {
            if(filter) delete filter;
        }
        enum {
            PORT_CUTOFF,
            PORT_Q,             
            PORT_GAIN,
        };
        void setPort(int port, DspFloatType v) {
            switch(port)
            {
                case PORT_CUTOFF: filter->set_cut_frequency(v); break;
                case PORT_Q: filter->set_Q(v); break;                
                case PORT_GAIN: filter->set_gain(v); break;
            }
        }
    };
}