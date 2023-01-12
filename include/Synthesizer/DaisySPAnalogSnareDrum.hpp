#pragma once

#include "DaisySP.hpp"

namespace DaisySP::Drums
{
    struct AnalogSnareDrum : public GeneratorProcessorPlugin<daisysp::AnalogSnareDrum>
    {        
        AnalogSnareDrum() : GeneratorProcessorPlugin<daisysp::AnalogSnareDrum>()
        {
            this->Init(sampleRate);
        }
        enum {
            PORT_TRIG,            
            PORT_SUSTAIN,
            PORT_ACCENT,
            PORT_FREQ,
            PORT_TONE,
            PORT_DECAY,
            PORT_SNAPPY,
        };
        void setPort(int port, DspFloatType v) {
            switch(port) {
                case PORT_TRIG: this->Trig(); break; 
                case PORT_SUSTAIN: this->SetSustain(v); break;
                case PORT_ACCENT: this->SetAccent(v); break;
                case PORT_FREQ: this->SetFreq(v); break;
                case PORT_TONE: this->SetTone(v); break;
                case PORT_DECAY: this->SetDecay(v); break;
                case PORT_SNAPPY: this->SetSnappy(v); break;
            }
        }
        DspFloatType Tick(DspFloatType I=0, DspFloatType A=1, DspFloatType X=1, DspFloatType Y=1 )
        {
            return A*this->Process(I);
        }
        
    };
}