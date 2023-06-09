#pragma once

#include <cmath>

namespace FX::Filters
{
    class OnePole {
    public:
        OnePole() {a0 = 1.0; b1 = 0.0; z1 = 0.0;};
        OnePole(DspFloatType Fc) {z1 = 0.0; setFc(Fc);};
        ~OnePole() = default;

        void setFc(DspFloatType Fc);
        void setHighPass(DspFloatType Fc);
        DspFloatType process(DspFloatType in);

    protected:
        DspFloatType a0, b1, z1;
    };

    // low pass
    inline void OnePole::setFc(DspFloatType Fc) {
        b1 = exp(-2.0 * M_PI * Fc);
        a0 = 1.0 - b1;
    }

    inline void OnePole::setHighPass(DspFloatType Fc) {
        b1 = -exp(-2.0 * M_PI * (0.5 - Fc));
        a0 = 1.0 + b1;
    }

    inline DspFloatType OnePole::process(DspFloatType in) {
        return z1 = in * a0 + z1 * b1;
    }
}