#pragma once

#include "DaisySP.hpp"

namespace DaisySP::Dynamics
{
    struct Compressor : public MonoFXProcessorPlugin<daisysp::Compressor>
    {
        Compressor() : MonoFXProcessorPlugin<daisysp::Compressor>()
        {
            this->Init(sampleRate);
        }
        enum {
            PORT_RATIO,
            PORT_THRESHOLD,
            PORT_ATTACK,
            PORT_RELEASE,
            PORT_MAKEUP,
            PORT_AUTOMAKEUP,
            PORT_GAIN,
        };
        void setPort(int port, DspFloatType v) {
            switch(port) {
                case PORT_RATIO: this->SetRatio(v); break;
                case PORT_THRESHOLD: this->SetThreshold(v); break;
                case PORT_ATTACK: this->SetAttack(v); break;
                case PORT_RELEASE: this->SetRelease(v); break;
                case PORT_MAKEUP: this->SetMakeup(v); break;
                case PORT_AUTOMAKEUP: this->AutoMakeup((bool)v); break;
            }
        }
        DspFloatType getPort(int port) {
            switch(port) {
                case PORT_GAIN: return this->GetGain();       
                case PORT_RATIO: return this->GetRatio();
                case PORT_THRESHOLD: return this->GetThreshold();
                case PORT_ATTACK: return this->GetAttack();
                case PORT_RELEASE: return this->GetRelease();
                case PORT_MAKEUP: return this->GetMakeup();
            }
            return 0.0;
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