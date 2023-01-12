#pragma once

#include "DaisySP.hpp"

namespace DaisySP::FX
{
    struct Phaser : public MonoFXProcessorPlugin<daisysp::Phaser>
    {
        Phaser() : MonoFXProcessorPlugin<daisysp::Phaser>()
        {
            this->Init(sampleRate);
        }
        enum {
            PORT_LFODEPTH,
            PORT_LFOFREQ,
            PORT_FREQ,
            PORT_FEEDBACK,
        };
        void setPort(int port, DspFloatType v) {
            switch(port) {
                case PORT_LFODEPTH: this->SetLfoDepth(v); break;
                case PORT_LFOFREQ: this->SetLfoFreq(v); break;
                case PORT_FREQ: this->SetFreq(v); break;
                case PORT_FEEDBACK: this->SetFeedback(v); break;
            }
        }
        DspFloatType Tick(DspFloatType I, DspFloatType A=1, DspFloatType X=1, DspFloatType Y=1) {
            return A*this->Process(I);
        }
        void ProcessBlock(size_t n, DspFloatType * in, DspFloatType * out) {
            std::vector<float> _in(n),_out(n);
            for(size_t i = 0; i < n; i++) _in[i] = in[i];
            for(size_t i = 0; i < n; i++) _out[i] = Tick(_in[i]);
            for(size_t i = 0; i < n; i++) out[i] = _out[i];
        }
    };
}