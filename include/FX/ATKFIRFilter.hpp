#pragma once

#include "ATK.hpp"

namespace Filters::AudioTK::EQ
{  
    struct FIRFilter : public ATKFilter
    {
        ATK::FIRFilter<ATK::CustomFIRCoefficients<DspFloatType>> * filter;
        
        FIRFilter(const std::vector<DspFloatType> & taps) : ATKFilter()
        {
            filter = new ATK::FIRFilter<ATK::CustomFIRCoefficients<DspFloatType>>(2);            
            filter->set_coefficients_in(taps);
            this->setFilter(filter);
        }
        ~FIRFilter() {
            if(filter) delete filter;
        }
        void setCoefficients(std::vector<DspFloatType> & taps) 
        {            
            filter->set_coefficients_in(taps);
        }
    };
}