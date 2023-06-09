#pragma once

#include <cmath>
#include <vector>

extern DspFloatType sampleRate;
extern DspFloatType invSampleRate;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// B-A-N-D-L-I-M-I-T-E-D O-S-C-I-L-L-A-T-O-R-S
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "FX/OnePole.hpp"
#include "FX/Filters.h"
#include "FX/PolyBLEP.hpp"
#include "FX/minBLEP.hpp"

namespace Oscillators
{
    //////////////////////////////////////////////
    // Old Blit it works
    //////////////////////////////////////////////
    struct BlitSaw : public OscillatorProcessor
    {
        //! Class constructor.
        BlitSaw( DspFloatType frequency = 220.0 ) : OscillatorProcessor()
        {
            nHarmonics_ = 0;
            offset = 0;
            reset();
            setFrequency( frequency );
            block.setFc(10.0f/sampleRate);
            gain = 1;
        }

        //! Class destructor.
        ~BlitSaw() = default;

        //! Resets the oscillator state and phase to 0.
        void reset()
        {
            phase_ = 0.0f;
            state_ = 0.0;
            y = 0.0;
        }

        //! Set the sawtooth oscillator rate in terms of a frequency in Hz.
        void setFrequency( DspFloatType frequency )
        {
            p_ = sampleRate / frequency;
            C2_ = 1 / p_;
            rate_ = M_PI * C2_;
            updateHarmonics();
        }

        void setHarmonics( unsigned int nHarmonics = 0 )
        {
            nHarmonics_ = nHarmonics;
            this->updateHarmonics();        
            state_ = -0.5 * a_;
        }

        
        //! Return the last computed output value.
        DspFloatType lastOut( void ) const { return y; };

        void setGain(DspFloatType g) {
            gain = g;
        }
        
        //! Compute and return one output sample.
        
        // blit = sin(m * phase) / (p * sin(phase));

        DspFloatType Tick( DspFloatType I=1, DspFloatType A=1, DspFloatType X=0, DspFloatType Y=0 )
        {     
            // I = index
            // X = FM
            // Y = PM
            DspFloatType tmp, denominator = sin( phase_ );
            if ( fabs(denominator) <= std::numeric_limits<DspFloatType>::epsilon() )
                tmp = a_;
            else {
                tmp =  sin( m_ * phase_ );
                tmp /= p_ * denominator;
            }

            tmp += state_ - C2_;
            state_ = tmp * 0.995;
            //phase_   = x;
            phase_ += rate_;
            if ( phase_ >= M_PI ) phase_ -= M_PI;

            DspFloatType out = tmp;
            y -= block.process(y);
            return y;
        }

        DspFloatType getPhase() { return phase_; }

        void setPhaseOffset(DspFloatType o) {
            phase_ = o;    
        }

        void updateHarmonics( void )
        {
            if ( nHarmonics_ <= 0 ) {
                unsigned int maxHarmonics = (unsigned int) floor( 0.5 * p_ );
                m_ = 2 * maxHarmonics + 1;
            }
            else
                m_ = 2 * nHarmonics_ + 1;

            a_ = m_ / p_;
        }

        FX::Filters::OnePole     block;    
        unsigned int nHarmonics_;
        unsigned int m_;
        DspFloatType rate_;
        DspFloatType phase_;
        DspFloatType offset;
        DspFloatType p_;
        DspFloatType C2_;
        DspFloatType a_;
        DspFloatType state_;
        DspFloatType y;
        DspFloatType gain;
    };


    ///////////////////////////////////////////////////////////////////////////////////////
    // square is made from subtracting out of phase sawtooth waves
    ///////////////////////////////////////////////////////////////////////////////////////
    struct BlitSquare : public OscillatorProcessor
    {
        FX::Filters::OnePole block;
        BlitSaw s1,s2;
        DspFloatType _out = 0;
        DspFloatType _duty = 0.5;

        BlitSquare() : OscillatorProcessor()
        {
            block.setFc(10.0f/sampleRate);
            _out = 0;
            _duty = 0.5;
            setFrequency(440.0f);
            s1.setGain(1);
            s2.setGain(1);
        }
        void setFrequency(DspFloatType f)
        {
            s1.setFrequency(f);
            s2.setFrequency(f);        
        }
        void setDuty(DspFloatType d)
        {
            _duty = d;
        }
        void reset() {
            s1.reset();
            s2.reset();
        }
        DspFloatType Tick(DspFloatType I=1, DspFloatType A=1, DspFloatType X=0, DspFloatType Y=0) {
            DspFloatType r1 = s1.Tick();        
            s2.setPhaseOffset(s1.getPhase() + _duty*M_PI);
            DspFloatType r2 = s2.Tick();                
            _out = r2-r1;
            DspFloatType x = _out;
            x -= block.process(x);
            return 4*x;
        }
    };

    ///////////////////////////////////////////////////////////////////////////////////////
    // triangle integrates the square
    ///////////////////////////////////////////////////////////////////////////////////////
    struct BlitTriangle : public OscillatorProcessor
    {
        FX::Filters::OnePole b1,b2;    
        BlitSquare s1;
        DspFloatType _out = 0;
        
        BlitTriangle() : OscillatorProcessor()
        {
            b1.setFc(10.0f/sampleRate);
            b2.setFc(10.0f/sampleRate);
            setFrequency(440);
        }
        void setFrequency(DspFloatType f)
        {        
            s1.setFrequency(f);                
        }
        void setDuty(DspFloatType d)
        {    
            s1.setDuty(d);
        }
        void reset() {
            s1.reset();
            _out = 0;
        }
        DspFloatType Tick(DspFloatType I=1, DspFloatType A=1, DspFloatType X=0, DspFloatType Y=0) {
            DspFloatType r1 = s1.Tick();                        
            // there's a tremendous amount of weird dc noise in this thing
            r1   -= b1.process(r1);
            // not really sure why it works but it does I think m_ = harmonic * harmonic like the fourier expansion
            _out += (r1/s1.s1.m_);                
            DspFloatType x = _out;        
            return 2*(x-b2.process(x));
        }
    };


    ///////////////////////////////////////////////////////////////////////////////////////
    // The new blit
    ///////////////////////////////////////////////////////////////////////////////////////
    DspFloatType BlitDSF(DspFloatType phase,DspFloatType m,DspFloatType p, DspFloatType a) 
    {
        DspFloatType tmp, denominator = sin( phase );
        if ( fabs(denominator) <= std::numeric_limits<DspFloatType>::epsilon() )
            tmp = a;
        else {
            tmp =  sin( m * phase );
            tmp /= p * denominator;
        }
        return tmp;
    }

    struct blitSaw : public OscillatorProcessor
    {
        FX::Filters::OnePole block;
        unsigned int nHarmonics_;
        unsigned int m_;
        DspFloatType rate_;
        DspFloatType phase_;
        DspFloatType offset;
        DspFloatType p_;
        DspFloatType C2_;
        DspFloatType a_;
        DspFloatType state_;
        DspFloatType y;
        
        blitSaw(DspFloatType sampleRate=44100.0f, DspFloatType frequency=440.0f)
        : OscillatorProcessor()
        {
            nHarmonics_ = 0;
            offset = 0;
            reset();
            setFrequency( frequency );
            block.setFc(10.0f/sampleRate);        
        }

        //! Resets the oscillator state and phase to 0.
        void reset()
        {
            phase_ = 0.0f;
            state_ = 0.0;
            y = 0.0;
        }

        //! Set the sawtooth oscillator rate in terms of a frequency in Hz.
        void setFrequency( DspFloatType frequency )
        {
            p_      = (sampleRate) / frequency;
            C2_     = 1 / p_;
            rate_   = M_PI * C2_;
            updateHarmonics();
        }

        void setHarmonics( unsigned int nHarmonics = 0 )
        {
            nHarmonics_ = nHarmonics;
            this->updateHarmonics();        
            state_ = -0.5 * a_;
        }

        
        //! Return the last computed output value.
        DspFloatType lastOut( void ) const  { 
            return y; 
        };
        
        
        // blit = sin(m * phase) / (p * sin(phase));
        DspFloatType Tick( DspFloatType I=1, DspFloatType A=1, DspFloatType X=0, DspFloatType Y=0 )
        {    
            DspFloatType tmp = BlitDSF(phase_,m_,p_,a_);        
            tmp += state_ - C2_;
            state_ = tmp * 0.995;        
            phase_ += rate_;
            if ( phase_ >= M_PI ) phase_ -= M_PI;
            y = clamp(tmp,-1,1);        
            y -= block.process(y);
            return 2*(0.8*y+0.47)-1;
        }

        DspFloatType getPhase() { 
            return phase_; 
        }

        void setPhaseOffset(DspFloatType o) {
            phase_ = o;    
        }

        void updateHarmonics( void )
        {
            if ( nHarmonics_ <= 0 ) {
                unsigned int maxHarmonics = (unsigned int) floor( 0.5 * p_ );
                m_ = 2 * maxHarmonics + 1;
            }
            else
                m_ = 2 * nHarmonics_ + 1;

            a_ = m_ / p_;
        }
    };

    struct blitSquare : public OscillatorProcessor
    {
        FX::Filters::OnePole block;
        unsigned int nHarmonics_;
        unsigned int m_;
        DspFloatType f;
        DspFloatType rate_;
        DspFloatType phase_;
        DspFloatType offset;
        DspFloatType p_;
        DspFloatType C2_;
        DspFloatType a_;
        DspFloatType state_;
        DspFloatType y;
        DspFloatType D;
        
        blitSquare(DspFloatType sampleRate=44100.0f, DspFloatType frequency=440.0f)
        : OscillatorProcessor()
        {
            nHarmonics_ = 0;
            offset = 0;
            reset();
            setFrequency( frequency );
            block.setFc(10.0f/sampleRate);        
            D = 0.5;
        }

        //! Resets the oscillator state and phase to 0.
        void reset()
        {
            phase_ = 0.0f;
            state_ = 0.0;
            y = 0.0;
        }

        //! Set the sawtooth oscillator rate in terms of a frequency in Hz.
        void setFrequency( DspFloatType frequency )
        {
            f       = frequency;
            p_      = (sampleRate) / frequency;
            C2_     = 1 / p_;
            rate_   = M_PI * C2_;
            updateHarmonics();
        }

        void setHarmonics( unsigned int nHarmonics = 0 )
        {
            nHarmonics_ = nHarmonics;
            this->updateHarmonics();        
            state_ = -0.5 * a_;
        }

        
        //! Return the last computed output value.
        DspFloatType lastOut( void ) const  { 
            return y; 
        };
        
        
        // blit = sin(m * phase) / (p * sin(phase));
        DspFloatType Tick( DspFloatType I=1, DspFloatType A=1, DspFloatType X=0, DspFloatType Y=0 )
        {    
            DspFloatType tmp = BlitDSF(phase_,m_,p_,a_);        
            DspFloatType tmp2= BlitDSF(phase_+D*M_PI,m_,p_,a_);
            tmp      = tmp - tmp2;
            //tmp     += state_ - C2_;        
            state_ += tmp * 0.995;
            phase_ += rate_;
            if ( phase_ >= 2*M_PI ) phase_ -= 2*M_PI;
            y = state_;
            y -= block.process(y);
            return 2*((y+D)*0.7+0.15)-1;
        }

        DspFloatType getPhase() { 
            return phase_; 
        }

        void setPhaseOffset(DspFloatType o) {
            phase_ = o;    
        }

        void updateHarmonics( void )
        {
            if ( nHarmonics_ <= 0 ) {
                unsigned int maxHarmonics = (unsigned int) floor( 0.5 * p_ );
                m_ = 2 * maxHarmonics + 1;
            }
            else
                m_ = 2 * nHarmonics_ + 1;

            a_ = m_ / p_;
        }
    };

    struct blitTriangle : public OscillatorProcessor
    {
        blitSquare sqr;
        FX::Filters::OnePole b1;

        DspFloatType triangle;
        blitTriangle(DspFloatType sampleRate=44100.0f, DspFloatType frequency=440.0f) : 
        OscillatorProcessor(),
        sqr(sampleRate,frequency)
        {
            b1.setFc(10.0f/sampleRate);
            triangle = 0;
        }
        void reset() {
            triangle = 0;
        }
        void setFrequency(DspFloatType f) {
            sqr.setFrequency(f);
        }
        DspFloatType Tick(DspFloatType I=1, DspFloatType A=1, DspFloatType X=1, DspFloatType Y=1)
        {
            DspFloatType x = sqr.Tick();
            DspFloatType a = 1.0 - 0.1*std::fmin(1,sqr.f/1000.0);
            triangle = a*triangle + x/sqr.p_;
            DspFloatType kaka = b1.process(triangle);
            triangle -= kaka;
            return 4*triangle;
        }
    };


    /////////////////////////////////////////////////////////////////////
    // Differential Parablic Wave
    /////////////////////////////////////////////////////////////////////
    struct DPWSaw : public OscillatorProcessor
    {
        DspFloatType freq,fs,inc;
        DspFloatType phase,lastPhase;
        DspFloatType lastValue,position;
        DspFloatType scaleFactor;

        DPWSaw() : OscillatorProcessor()
        {
            freq = 440.0f;
            fs   = sampleRate;
            inc  = freq/fs;
            lastValue = phase = lastPhase = position = 0.0f;
            scaleFactor = sampleRate / (4.0f * freq);
        }
        void setFrequency(DspFloatType f) {
            freq = f;
            inc  = f/fs;
            scaleFactor = sampleRate / (4.0f * freq);
        }
        DspFloatType Tick(DspFloatType I=1, DspFloatType A=1, DspFloatType X=1, DspFloatType Y=1)
        {                                    
            position += phase - lastPhase;
            lastPhase = phase;

            position = fmod(position, 1.0f);

            DspFloatType value = position * 2 - 1;
            value = value * value;
            
            DspFloatType out = scaleFactor * (value - lastValue);
            lastValue = value;

            phase = fmod(phase + inc,1.0f);
            return out;
        }   
    };

    struct DPWPulse
    {
        DspFloatType freq,fs,inc;
        DspFloatType phase,lastPhase;
        DspFloatType lastValueA,lastValueB,position;
        DspFloatType positionA,positionB;
        DspFloatType scaleFactor;

        DPWPulse()
        {
            freq = 440.0f;
            fs   = sampleRate;
            inc  = freq/fs;
            lastValueA = lastValueB = phase = lastPhase = position = 0.0f;
            positionA = 0.5f;
            positionB = 0.5f;
            scaleFactor = 0.5f * sampleRate /(4.0f * freq);    
            phase = 0.5;
        }
        void setFrequency(DspFloatType f) {
            freq = f;
            inc  = f/fs;
            scaleFactor = 0.5f * sampleRate /(4.0f * freq);    
        }
        void setDuty(DspFloatType d) {
            phase = clamp(d,0.01f,0.99f);
        }
        DspFloatType Tick(DspFloatType I=1, DspFloatType A=1, DspFloatType X=0, DspFloatType Y=0) {
                            
            positionB += phase - lastPhase;
            lastPhase = phase;

            positionA = fmod(positionA, 1.0f);
            positionB = fmod(positionB, 1.0f);

            DspFloatType valueA = positionA * 2.0f - 1.0f;
            DspFloatType valueB = positionB * 2.0f - 1.0f;
            valueA = valueA * valueA;
            valueB = valueB * valueB;
            DspFloatType out = ((valueA - lastValueA) -(valueB - lastValueB)) * scaleFactor;
            lastValueA = valueA;
            lastValueB = valueB;

            positionA += freq * invSampleRate;
            positionB += freq * invSampleRate;

            return out;        
        }
    };

    struct DPWTriangle
    {
        DspFloatType freq,fs,inc;
        DspFloatType phase,lastPhase;
        DspFloatType lastValue,position;    
        DspFloatType scaleFactor;

        DPWTriangle()
        {
            freq = 440.0f;
            fs   = sampleRate;
            inc  = freq/fs;
            lastValue = phase = lastPhase = position = 0.0f;
            position = 0.0f;        
            scaleFactor =  sampleRate / (2.0f * freq);
            phase = 0.5;
        }
        void setFrequency(DspFloatType f) {
            freq = f;
            inc  = f/fs;
            scaleFactor =  sampleRate / (2.0f * freq);
        }
        void setDuty(DspFloatType d) {
            phase = clamp(d,0.01f,0.99f);
        }
        DspFloatType Tick(DspFloatType I=1, DspFloatType A=1, DspFloatType X=1, DspFloatType Y=1)
        {        
            position += phase - lastPhase;
            lastPhase = phase;
            position = fmod(position, 1.0f);                
            DspFloatType out = std::abs(position - 0.5) * 4 - 1;                
            position += freq * invSampleRate;        
            return out;
        }
    };
}