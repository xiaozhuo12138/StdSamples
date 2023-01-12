#pragma once

namespace Filters::TransferFunctions
{   

    // H(s) transfer function bilinear/z
    struct TransferFunctionLaplace
    {
        std::vector<double> num, den;
        std::dequeue<double> dx,dy;

        TransferFunctionLaplace(size_t order) {
            num.resize(order);
            den.resize(order);
            for(size_t i = 0; i < order; i++)
            {
                dx.push(0.0);
                dy.push(0.0);
            }
        }
        void setNumS(int power, double val) {
            num[power] = val;
        }
        void setDenS(int power, double val) {
            den[power] = val;
        }
        double tick(double I)
        {
            double r = 0;
            for(size_t i = 0; i < num.size(); i++)
                r += num[i]*dx[i];
            for(size_t i = 0; i < den.size(); i++)
                r -= den[i]*dy[i];
            dx.push_front(I);
            dx.pop_back();
            dy.push_front(r);
            dy.pop_back();
            return r;
        }
    };

    // H(z) transfer function
    struct TransferFunctionDigital
    {
        std::vector<double> num, den;
    };


    struct Integrator
    {
        BiquadTypeI b;

        Integrator()
        {
            BiquadSection s;
            s.z[0] = 1;
            s.z[1] = 0;
            s.z[2] = 0;
            s.p[0] = 1;
            s.p[1] = 1;
            s.p[2] = 0;
            //            BiquadSection section = AnalogBiquadSection(s[0], fc, sampleRate);
            b = Filters::IIR::AnalogBiquadSection(s,sampleRate/2.0,sampleRate);
        }
        double Tick(double I, double A=1, double X=0, double Y=0)
        {
            return b.Tick(I,A,X,Y);
        }
    };
    //https://en.wikipedia.org/wiki/Differentiator
    struct Differentiator
    {
        BiquadTypeI b;

        Differentiator()
        {
            BiquadSection s;
            s.z[0] = 1;
            s.z[1] = 1;
            s.z[2] = 0;
            s.p[0] = 1;
            s.p[1] = 0;
            s.p[2] = 0;
            //            BiquadSection section = AnalogBiquadSection(s[0], fc, sampleRate);
            b = Filters::IIR::AnalogBiquadSection(s,sampleRate/2.0,sampleRate);
        }
        double Tick(double I, double A=1, double X=0, double Y=0)
        {
            return b.Tick(I,A,X,Y);
        }
    };

}