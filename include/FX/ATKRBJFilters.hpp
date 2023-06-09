#pragma once

#include "ATK.hpp"

namespace Filters::AudioTK
{  
    struct RBJLowPassFilter: public ATKFilter
    {
        ATK::IIRFilter<ATK::RobertBristowJohnsonLowPassCoefficients<DspFloatType>> * filter;

        RBJLowPassFilter() : ATKFilter()
        {
            filter = new ATK::IIRFilter<ATK::RobertBristowJohnsonLowPassCoefficients<DspFloatType>>(2);
        }
        ~RBJLowPassFilter() {
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
    struct RBJHighPassFilter: public ATKFilter
    {
        ATK::IIRFilter<ATK::RobertBristowJohnsonHighPassCoefficients<DspFloatType>> * filter;

        RBJHighPassFilter() : ATKFilter()
        {
            filter = new ATK::IIRFilter<ATK::RobertBristowJohnsonHighPassCoefficients<DspFloatType>>(2);
        }
        ~RBJHighPassFilter() {
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
    struct RBJBandPassFilter: public ATKFilter
    {
        ATK::IIRFilter<ATK::RobertBristowJohnsonBandPassCoefficients<DspFloatType>> * filter;

        RBJBandPassFilter() : ATKFilter()
        {
            filter = new ATK::IIRFilter<ATK::RobertBristowJohnsonBandPassCoefficients<DspFloatType>>(2);
        }
        ~RBJBandPassFilter() {
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
    struct RBJBandPass2Filter: public ATKFilter
    {
        ATK::IIRFilter<ATK::RobertBristowJohnsonBandPass2Coefficients<DspFloatType>> * filter;

        RBJBandPass2Filter() : ATKFilter()
        {
            filter = new ATK::IIRFilter<ATK::RobertBristowJohnsonBandPass2Coefficients<DspFloatType>>(2);
        }
        ~RBJBandPass2Filter() {
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
    struct RBJBandStopFilter: public ATKFilter
    {
        ATK::IIRFilter<ATK::RobertBristowJohnsonBandStopCoefficients<DspFloatType>> * filter;

        RBJBandStopFilter() : ATKFilter()
        {
            filter = new ATK::IIRFilter<ATK::RobertBristowJohnsonBandStopCoefficients<DspFloatType>>(2);
        }
        ~RBJBandStopFilter() {
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
    struct RBJAllPassFilter: public ATKFilter
    {
        ATK::IIRFilter<ATK::RobertBristowJohnsonAllPassCoefficients<DspFloatType>> * filter;

        RBJAllPassFilter() : ATKFilter()
        {
            filter = new ATK::IIRFilter<ATK::RobertBristowJohnsonAllPassCoefficients<DspFloatType>>(2);
        }
        ~RBJAllPassFilter() {
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
    struct RBJBandPassPeakFilter: public ATKFilter
    {
        ATK::IIRFilter<ATK::RobertBristowJohnsonBandPassPeakCoefficients<DspFloatType>> * filter;

        RBJBandPassPeakFilter() : ATKFilter()
        {
            filter = new ATK::IIRFilter<ATK::RobertBristowJohnsonBandPassPeakCoefficients<DspFloatType>>(2);
        }
        ~RBJBandPassPeakFilter() {
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
    struct RBJLowShelfFilter: public ATKFilter
    {
        ATK::IIRFilter<ATK::RobertBristowJohnsonLowShelvingCoefficients<DspFloatType>> * filter;

        RBJLowShelfFilter() : ATKFilter()
        {
            filter = new ATK::IIRFilter<ATK::RobertBristowJohnsonLowShelvingCoefficients<DspFloatType>>(2);
        }
        ~RBJLowShelfFilter() {
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
    struct RBJHighShelfFilter: public ATKFilter
    {
        ATK::IIRFilter<ATK::RobertBristowJohnsonHighShelvingCoefficients<DspFloatType>> * filter;

        RBJHighShelfFilter() : ATKFilter()
        {
            filter = new ATK::IIRFilter<ATK::RobertBristowJohnsonHighShelvingCoefficients<DspFloatType>>(2);
        }
        ~RBJHighShelfFilter() {
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