#pragma once

#define GAMMA_H_INC_ALL
#include "Gamma/Gamma.h"
#include "Gamma/Voices.h"

namespace Gamma::Filters
{
    struct AllPass2 : public FilterProcessorPlugin<gam::AllPass2>
    {
        AllPass2() : FilterProcessorPlugin<gam::AllPass2>()
        {

        }
        enum {
            PORT_WIDTH,
            PORT_ZERO,
            PORT_CUTOFF,
        };
        void setPort(int port, double v) {
            switch(port) {
                case PORT_WIDTH: this->width(v); break;
                case PORT_ZERO: this->zero(); break;
                case PORT_CUTOFF: this->freq(v); break;
            }
        }
        double Tick(double I, double A=1, double X=1, double Y=1) {
            return A*(*this)(I);
        }
    };
}