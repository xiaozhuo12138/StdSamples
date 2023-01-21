#pragma once
#include <iostream>
#include <vector>
#include <Undenormal.hpp>
#include <spuce/filters/iir_coeff.h>
#include <spuce/filters/butterworth_iir.h>
#include <spuce/filters/iir_df.h>

#define MAX_ORDER 64

typedef float DspFloatType;

struct FilterCoefficients
{
    DspFloatType a[2];
    DspFloatType b[3];
};

struct BiquadSection
{
    DspFloatType z[3];
    DspFloatType p[3];

    BiquadSection()
    {
        memset(z, 0, sizeof(z));
        memset(p, 0, sizeof(p));
    }
    BiquadSection(const FilterCoefficients &c)
    {
        z[0] = c.b[0];
        z[1] = c.b[1];
        z[2] = c.b[2];
        p[0] = c.a[0];
        p[1] = c.a[1];
    }
    BiquadSection(DspFloatType z1, DspFloatType z2, DspFloatType z3, DspFloatType p1, DspFloatType p2)
    {
        z[0] = z1;
        z[1] = z2;
        z[2] = z3;
        p[0] = p1;
        p[1] = p2;
    }
    BiquadSection(const BiquadSection &b)
    {
        memcpy(z, b.z, sizeof(z));
        memcpy(p, b.p, sizeof(p));
    }
    void setCoefficients(DspFloatType z1, DspFloatType z2, DspFloatType z3, DspFloatType p1, DspFloatType p2)
    {
        z[0] = z1;
        z[1] = z2;
        z[2] = z3;
        p[0] = p1;
        p[1] = p2;
    }
    void setCoefficients(DspFloatType n[3], DspFloatType d[2])
    {
        memcpy(z, n, sizeof(z));
        memcpy(p, d, sizeof(p));
    }
    void setCoefficients(const FilterCoefficients &c)
    {
        z[0] = c.b[0];
        z[1] = c.b[1];
        z[2] = c.b[2];
        p[0] = c.a[0];
        p[1] = c.a[1];
    }
    BiquadSection &operator=(const BiquadSection &b)
    {
        memcpy(z, b.z, sizeof(z));
        memcpy(p, b.p, sizeof(p));
        return *this;
    }

    void print()
    {
        std::cout << z[0] << " + " << z[1] << " z^-1 + " << z[2] << " z^-1\n";
        std::cout << "-------------------------------------------------------------\n";
        std::cout << " 1 + " << p[0] << +" z^-1 + " << p[1] << " z^-2\n";
    }
};


using BiquadSOS = std::vector<BiquadSection>;

struct BiquadTransposedTypeII 
{
    BiquadSection biquad;
    DspFloatType x, y, d1, d2;

    BiquadTransposedTypeII() 
    {
        x = y = 0;
        d1 = d2 = 0;
    }
    BiquadTransposedTypeII(const BiquadSection &b) : biquad(b)
    {
        x = y = 0;
        d1 = d2 = 0;
    }
    BiquadTransposedTypeII &operator=(const BiquadTransposedTypeII &b)
    {
        biquad = b.biquad;
        x = b.x;
        y = b.y;
        d1 = b.d1;
        d2 = b.d2;
        return *this;
    }
    void setCoefficients(const BiquadSection &b)
    {
        biquad = b;
    }        
    void setBiquad(const BiquadSection &b)
    {
        biquad = b;
    }

    // transposed is just flip - to +
    DspFloatType Tick(DspFloatType I, DspFloatType A = 1, DspFloatType X = 0, DspFloatType Y = 0)
    {
        Undenormal denormal;
        x = I;
        y = biquad.z[0] * x + d1;
        d1 = biquad.z[1] * x - biquad.p[0] * y + d2;
        d2 = biquad.z[2] * x - biquad.p[1] * y;
        return A * y;
    }
};

struct FilterBase
{
    
    std::vector<BiquadTransposedTypeII> biquads;
    size_t order;
    DspFloatType fc,sr,R,q,bw,g,ripple,rolloff,stop,pass;
    bool init = false;

    FilterBase() 
    {
    }
    FilterBase(const BiquadSOS &s)
    {
        setCoefficients(s);
    }
    void setCoefficients(const BiquadSOS &s)
    {        
        if(s.size() == 0) {
            init = false;
            return;
        }
        biquads.resize(s.size());
        for (size_t i = 0; i < s.size(); i++)
        {
            biquads[i].setCoefficients(s[i]);
        }
        init = true;
    }
    DspFloatType Tick(DspFloatType I, DspFloatType A = 1, DspFloatType X = 0, DspFloatType Y = 0)
    {
        if(!init) return 0;
        DspFloatType o = biquads[0].Tick(I, A, X, Y);
        for (size_t i = 1; i < biquads.size(); i++)
            o = biquads[i].Tick(o, A, X, Y);
        return A * o;
    }

    void ProcessBlock(size_t n, DspFloatType * in, DspFloatType * out) {
        for(size_t i = 0; i < n; i++) out[i] = Tick(in[i]);
    }
    void ProcessBlock(size_t n, DspFloatType * in, DspFloatType * out, DspFloatType * A) {
        for(size_t i = 0; i < n; i++) out[i] = Tick(in[i],A[i]);
    }
    void ProcessBlock(size_t n, DspFloatType * in, DspFloatType * out, DspFloatType * A, DspFloatType * X) {
        for(size_t i = 0; i < n; i++) out[i] = Tick(in[i],A[i],X[i]);
    }
    void ProcessBlock(size_t n, DspFloatType * in, DspFloatType * out, DspFloatType * A, DspFloatType * X, DspFloatType * Y) {
        for(size_t i = 0; i < n; i++) out[i] = Tick(in[i],A[i],X[i],Y[i]);
    }
};

void prewarp(DspFloatType *a0, DspFloatType *a1, DspFloatType *a2, DspFloatType fc, DspFloatType fs)
{
    DspFloatType wp, pi;

    pi = 4.0 * std::atan(1.0);
    wp = 2.0 * fs * std::tan(pi * fc / fs);

    *a2 = (*a2) / (wp * wp);
    *a1 = (*a1) / wp;
}
void prewarpR(DspFloatType *a0, DspFloatType *a1, DspFloatType *a2, DspFloatType fc, DspFloatType fs, DspFloatType R)
{
    DspFloatType wp, pi;

    pi = 4.0 * std::atan(1.0);
    wp = 2.0 * fs * std::tan(pi * fc / fs);

    *a2 = R * R * (*a2) / (wp * wp);
    *a1 = R * (*a1) / wp;
}
void prewarpQ(DspFloatType *a0, DspFloatType *a1, DspFloatType *a2, DspFloatType fc, DspFloatType fs, DspFloatType Q)
{
    DspFloatType wp, pi;

    pi = 4.0 * std::atan(1.0);
    wp = 2.0 * fs * std::tan(pi * fc / fs);

    *a2 = (*a2) / (Q * Q * wp * wp);
    *a1 = (*a1) / (Q * wp);
}
void prewarpRQ(DspFloatType *a0, DspFloatType *a1, DspFloatType *a2, DspFloatType fc, DspFloatType fs, DspFloatType R, DspFloatType Q)
{
    DspFloatType wp, pi;

    pi = 4.0 * std::atan(1.0);
    wp = 2.0 * fs * std::tan(pi * fc / fs);

    *a2 = R * R * (*a2) / (Q * Q * wp * wp);
    *a1 = R * (*a1) / (Q * wp);
}

void inversebilinear(
    DspFloatType z[3], DspFloatType p[3],
    DspFloatType k,    /* overall gain factor */
    DspFloatType fs,   /* sampling rate */
    DspFloatType *coef /* pointer to 4 iir coefficients */
)
{
    DspFloatType ad, bd;
    DspFloatType b0 = k;
    DspFloatType b1 = coef[0];
    DspFloatType b2 = coef[1];
    DspFloatType a0 = 1;
    DspFloatType a1 = coef[2];
    DspFloatType a2 = coef[3];

    ad = 1 / 4. * a2 / (fs * fs) + a1 / (2 * fs) + a0;
    bd = 1 / 4. * b2 / (fs * fs) + b1 / (2 * fs) + b0;

    z[0] = k * bd / ad;
    z[1] = b1 * bd;
    z[2] = b2 * bd;
    p[0] = 1;
    p[1] = a1 * ad;
    p[2] = a2 * ad;
}

void bilinear(
    DspFloatType a0, DspFloatType a1, DspFloatType a2, /* numerator coefficients */
    DspFloatType b0, DspFloatType b1, DspFloatType b2, /* denominator coefficients */
    DspFloatType *k,                       /* overall gain factor */
    DspFloatType fs,                       /* sampling rate */
    DspFloatType *coef                     /* pointer to 4 iir coefficients */
)
{
    DspFloatType ad, bd;

    /* alpha (Numerator in s-domain) */
    ad = 4. * a2 * fs * fs + 2. * a1 * fs + a0;
    /* beta (Denominator in s-domain) */
    bd = 4. * b2 * fs * fs + 2. * b1 * fs + b0;

    /* update gain constant for this section */
    *k *= ad / bd;

    // Q
    // Q *= 0.5/b1
    // b1 = (b1/Q)

    // radius
    // b1 = R
    // b2 = R*R

    /* Denominator */
    // K = 2 * fs
    // K already prewapred
    *coef++ = (2. * b0 - 8. * b2 * fs * fs) / bd;           /* beta1 */
    *coef++ = (4. * b2 * fs * fs - 2. * b1 * fs + b0) / bd; /* beta2 */

    /* Nominator */
    *coef++ = (2. * a0 - 8. * a2 * fs * fs) / ad;         /* alpha1 */
    *coef = (4. * a2 * fs * fs - 2. * a1 * fs + a0) / ad; /* alpha2 */
}


// convert analog setion to biquad type I
BiquadSection AnalogBiquadSection(const BiquadSection &section, DspFloatType fc, DspFloatType fs)
{
    BiquadSection ns = section;
    prewarp(&ns.z[0], &ns.z[1], &ns.z[2], fc, fs);
    prewarp(&ns.p[0], &ns.p[1], &ns.p[2], fc, fs);
    std::vector<DspFloatType> coeffs(4);
    DspFloatType k = 1;
    bilinear(ns.z[0], ns.z[1], ns.z[2], ns.p[0], ns.p[1], ns.p[2], &k, fs, coeffs.data());
    ns.z[0] = k;
    ns.z[1] = coeffs[2];
    ns.z[2] = coeffs[3];
    ns.p[0] = coeffs[0];
    ns.p[1] = coeffs[1];
    ns.p[2] = 0;
    return ns;
}
// H(s) => Bilinear/Z => H(z)
// convert analog sos to biquad cascade type I
BiquadSOS AnalogBiquadCascade(const BiquadSOS &sos, DspFloatType fc, DspFloatType fs)
{
    BiquadSOS nsos = sos;
    for (size_t i = 0; i < sos.size(); i++)
    {
        BiquadSection b = AnalogBiquadSection(sos[i], fc, fs);
        nsos[i] = b;
    }
    return nsos;
}

/*
void lp2lp(std::vector<std::complex<DspFloatType>> & poles,
            DspFloatType wc, DspFloatType & gain) {
        
    gain *= pow(wc,poles.size());
    for(size_t i = 0; i < poles.size(); i++)
        poles[i] = wc * poles[i];
        
}
void lp2hp(std::vector<std::complex<DspFloatType>> & zeros,
            std::vector<std::complex<DspFloatType>> & poles,
            DspFloatType wc, DspFloatType & gain)
{
    std::complex<DspFloatType> prodz(1.0,0.0),prodp(1.0,0.0);
    for(size_t i = 0; i < zeros.size(); i++) prodz *= -zeros[i];
    for(size_t i = 0; i < poles.size(); i++) prodp *= -poles[i];
    gain *= prodz.real() / prodp.real();
    for(size_t i = 0; i < poles.size(); i++)
        if(abs(poles[i])) poles[i] = std::complex<DspFloatType>(wc) / poles[i];
    zeros.resize(poles.size());
    for(size_t i = 0; i < zeros.size(); i++)
        zeros[i] = std::complex<DspFloatType>(0.0);
}               
void lp2bp(std::vector<std::complex<DspFloatType>> & zeros,
            std::vector<std::complex<DspFloatType>> & poles,
            DspFloatType wu, DspFloatType wl, DspFloatType & gain)
{
    DspFloatType wc = sqrt(wu*wl);
    DspFloatType bw = wu-wl;
    gain      *= pow(bw,poles.size()-zeros.size());
    std::vector<std::complex<DspFloatType>> temp;
    for(size_t i = 0; i < poles.size(); i++) 
    {
        if(abs(poles[i])) {
            std::complex<DspFloatType> first = DspFloatType(0.5) * poles[i] * bw;
            std::complex<DspFloatType> second= DspFloatType(0.5) * sqrt(bw*bw) * (poles[i]*poles[i]-DspFloatType(4.0)*wc*wc);
            temp.push_back(first + second);
        }
    }
    for(size_t i = 0; i < poles.size(); i++) {
        if(abs(poles[i])) {
            std::complex<DspFloatType> first = DspFloatType(0.5) * poles[i] * bw;
            std::complex<DspFloatType> second= DspFloatType(0.5) * sqrt(bw*bw) * (poles[i]*poles[i]-DspFloatType(4.0)*wc*wc);
            temp.push_back(first - second);
        }
    }
    zeros.resize(poles.size());
    for(size_t i = 0; i < zeros.size(); i++) {
        zeros[i] = std::complex<DspFloatType>(0);
    }
    size_t index = 0;
    poles.resize(temp.size());
    for(auto i = temp.begin(); i != temp.end(); i++) {
        poles[index] = *i;
        index++;
    }        
}       
void lp2bs(std::vector<std::complex<DspFloatType>> & zeros,
            std::vector<std::complex<DspFloatType>> & poles,
            DspFloatType wu, DspFloatType wl, DspFloatType & gain)
{ 
    DspFloatType bw = wu-wl;
    DspFloatType Wc = sqrt(wu*wl);
    Complex prodz(1.0,0.0);
    Complex prodp(1.0,0.0);
    for(size_t i = 0; i < zeros.size(); i++)
        prodz *= -zeros[i];
    for(size_t i = 0; i < poles.size(); i++)
        prodp *= -poles[i];
    gain *= prodz.real() / prodp.real();
    std::vector<Complex> ztmp;
    for(size_t i = 0; i < zeros.size(); i++) {
        ztmp.push_back(Complex(0.0,Wc));
        ztmp.push_back(Complex(0.0,-Wc));            
    }
    std::vector<Complex> ptmp;
    for(size_t i = 0; i < poles.size(); i++) {
        if(abs(poles[i])) {
            Complex term1 = DspFloatType(0.5) * bw / poles[i];
            Complex term2 = DspFloatType(0.5) * sqrt((bw*bw) / (poles[i]*poles[i]) - (DspFloatType(4)*Wc*Wc));
            ptmp.push_back(term1+term2);
        }
    }
    size_t index = 0;
    for(auto i = ztmp.begin(); i != ztmp.end(); i++) {
        zeros[index++] = *i;
    }
    index = 0;
    for(auto i = ptmp.begin(); i != ptmp.end(); i++) {
        poles[index++] = *i;
    }
} 
*/

#define FORI(n) for(size_t i = 0; i < n; i++) {
#define ENDI }

struct SpuceIIRFilter
{
    spuce::iir_coeff coef;
    spuce::iir_df<DspFloatType> filter;                
};

// trying to resonate it but not working yet
struct LowpassButterworth : public FilterBase
{
    SpuceIIRFilter filter;    
    DspFloatType fc,ff,fs;
    size_t order;
    
    LowpassButterworth(size_t O,DspFloatType cutoff, DspFloatType sr)
    {
        fs = sr;            
        order = O;
        setCutoff(cutoff);
    }
    void setCutoff(DspFloatType f)
    {            
        ff = f/fs;     
        // (s-p1)*(s-p2) = s^2 -sp1 -sp2 +p1p2   
        filter.coef = spuce::iir_coeff(order,spuce::filter_type::low);                 
        spuce::butterworth_iir(filter.coef,ff,4.0);        
        //filter.filter = spuce::iir_df<DspFloatType>(filter.coef);

        // can't get the poles out of it
        std::vector<std::complex<double>> poles,zeros;        
        std::cout << filter.coef.getN2() << std::endl;
        for(size_t i = 0; i < order/2; i++) {
            std::complex<double> x;
            x = filter.coef.get_pole(i);            
            poles.push_back(x);
        }
        FORI(poles.size()) std::cout<< poles[i] << std::endl; ENDI
        
        std::vector<BiquadSection> sos;
        for(size_t i = 0; i < order/2; i++) {
            std::complex<double> x1,x2;
            x1 = poles[i];
            x2 = poles[i] * std::complex<double>(1,-1);
            double p1 = abs(x1*x2);
            double p2 = abs(-x1-x2);
            std::cout << p1 << std::endl;
            std::cout << p2 << std::endl;
            BiquadSection s;
            s.p[0] = 1.0;
            s.p[1] = p2/p1;
            s.p[2] = 1/p1;            
            s.z[0] = 1/p1;
            s.z[1] = 0;
            s.z[2] = 0;
            sos.push_back(s);
        }
        
        
        sos = AnalogBiquadCascade(sos,f,fs);
        setCoefficients(sos);
    }
    DspFloatType Tick(DspFloatType I, DspFloatType A=1, DspFloatType X=0, DspFloatType Y=0)
    {
        return A * FilterBase::Tick(I);
    }
};
