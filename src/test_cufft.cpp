#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <complex>
#include <ccomplex>
#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cufftw.h>
//#include "viper.hpp"
// 1D = audio/signals
// 2D = images matrix
// 3D = vision 

#include <cuda/std/ccomplex>
//#include "cuda_complex.hpp"

template<typename T>
using complex = cuda::std::complex<T>;



template<typename T>
std::ostream& operator << (std::ostream& o,const std::vector<complex<T>> & v)
{
    for(size_t i = 0; i < v.size(); i++) o << "(" << v[i].real() << "," << v[i].imag()  << "),";
    o << std::endl;
    return o;
}

void Memcpy(void * dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{        
    cudaMemcpy(dst,src,count,kind);        
} 

fftw_complex* fftw_alloc_complex(size_t n) {
    fftw_complex * p = nullptr;
    cudaMalloc(&p,n * sizeof(fftw_complex));
    return p;
}
fftwf_complex* fftwf_alloc_complex(size_t n) {
    fftwf_complex * p = nullptr;
    cudaMalloc(&p,n * sizeof(fftwf_complex));
    return p;
}
double* fftw_alloc_real(size_t n) {
    double * p = nullptr;
    cudaMalloc(&p,n * sizeof(double));
    return p;
}
float* fftwf_alloc_real(size_t n) {
    float * p = nullptr;
    cudaMalloc(&p,n * sizeof(float));
    return p;
}



template<typename T>
struct Window 
{   
    using Vector = std::vector<T>;

    Vector window;

    Window(size_t i) { window.resize(i); }
    virtual ~Window() = default;

    T& operator[](size_t i) { return window[i]; }

    Vector operator * (const Vector& v) { return window * v; }
};
template<typename T>
struct ComplexWindow 
{   
    using Vector = std::vector<complex<T>>;
    Vector window;

    ComplexWindow(size_t i) { window.resize(i); }
    virtual ~ComplexWindow() = default;

    T& operator[](size_t i) { return window[i]; }

    Vector operator * (const Vector& v) { return window * v; }
};

template<typename T>
struct Rectangle: public Window<T>
{
    Rectangle(size_t i) : Window<T>(i) { 
        fill(this->window,1.0f);
        } 
};
template<typename T>
struct Hamming: public Window<T>
{
    Hamming(size_t n) : Window<T>(n) {            
        for(size_t i = 0; i < this->window.size(); i++)
        {
            this->window[i] = 0.54 - (0.46 * std::cos(2*M_PI*(double)i/(double)n));
        }        
    }
};
template<typename T>
struct Hanning: public Window<T>
{
    Hanning(size_t n) : Window<T>(n) {            
        for(size_t i = 0; i < this->window.size(); i++)
        {
            this->window[i] = 0.5*(1 - std::cos(2*M_PI*(double)i/(double)n));
        }        
    }
};
template<typename T>
struct Blackman: public Window<T>
{
    Blackman(size_t n) : Window<T>(n)    
    {            
        for(size_t i = 0; i < this->window.size(); i++)                    
            this->window[i] = 0.42 - (0.5* std::cos(2*M_PI*i/(n)) + (0.08*std::cos(4*M_PI*i/n)));        
    }
};
template<typename T>
struct BlackmanHarris: public Window<T>
{
    BlackmanHarris(size_t n) : Window<T>(n)    
    {            
        for(size_t i = 0; i < this->window.size(); i++)            
        {   
            double ci = (double) i / (double) n;
            this->window[i] = 0.35875 
                    - 0.48829*std::cos(2*M_PI*(ci))
                    + 0.14128*std::cos(4.0*M_PI*(ci)) 
                    - 0.01168*std::cos(6.0*M_PI*(ci));
        }
    }
};
template<typename T>
struct Gaussian: public Window<T>
{
    Gaussian(size_t i) : Window<T>(i)
    {
        T a,b,c=0.5;
        for(size_t n = 0; n < this->window.size(); n++)
        {
            a = ((double)n - c*(this->window.size()-1)/(std::sqrt(c)*this->window.size()-1));
            b = -c * std::sqrt(a);
            this->window(n) = std::exp(b);
        }
    }
};
template<typename T>
struct Welch: public Window<T>
{
    Welch(size_t n) : Window<T>(n)
    {
        for(size_t i = 0; i < this->window.size(); i++)
            this->window[i] = 1.0 - std::sqrt((2.0*(double)i-(double)this->window.size()-1)/((double)this->window.size()));        
    }
};
template<typename T>
struct Parzen: public Window<T>
{

    Parzen(size_t n) : Window<T>(n)
    {
        for(size_t i = 0; i < this->window.size(); i++)
            this->window[i] = 1.0 - std::abs((2.0*(double)i-this->window.size()-1)/(this->window.size()));        
    }    
};
template<typename T>
struct Tukey: public Window<T>
{
    Tukey(size_t num_samples, T alpha) : Window<T>(num_samples)
    {            
        T value = (-1*(num_samples/2)) + 1;
        double n2 = (double)num_samples / 2.0;
        for(size_t i = 0; i < this->window.size(); i++)
        {    
            if(value >= 0 && value <= (alpha * (n2))) 
                this->window[i] = 1.0; 
            else if(value <= 0 && (value >= (-1*alpha*(n2)))) 
                this->vector.vector[i] = 1.0;
            else 
                this->vector.vector[i] = 0.5 * (1 + std::cos(M_PI *(((2.0*value)/(alpha*(double)num_samples))-1)))        ;
            value = value + 1;
        }     
    }
};

struct FFTPlanFloat
{
    fftwf_complex * in;    
    fftwf_complex * out;    
    size_t size;
    fftwf_plan pf,pb;

    FFTPlanFloat(size_t n)
    {
        in = fftwf_alloc_complex(n);
        out= fftwf_alloc_complex(n);            
        size = n;    
        pf = fftwf_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        pb = fftwf_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);        
    }
    ~FFTPlanFloat()
    {
        cudaFree(in);
        cudaFree(out);
        fftwf_destroy_plan(pf);
        fftwf_destroy_plan(pb);
    }
    void forward() {
        fftwf_execute(pf);
    }
    void backward() {
        fftwf_execute(pb);
    }
};

struct FFTPlanDouble
{
    fftw_complex * in;    
    fftw_complex * out;    
    size_t size;
    fftw_plan pf,pb;

    FFTPlanDouble(size_t n)
    {
        in = fftw_alloc_complex(n);
        out= fftw_alloc_complex(n);            
        size = n;    
        pf = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        pb = fftw_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);        
    }
    ~FFTPlanDouble()
    {
        cudaFree(in);
        cudaFree(out);
        fftw_destroy_plan(pb);
        fftw_destroy_plan(pf);
    }
    void forward() {
        fftw_execute(pf);
    }
    void backward() {
        fftw_execute(pb);
    }
};

std::vector<complex<double>> fft(FFTPlanDouble & plan, std::vector<complex<double>> & in)
{    
    Memcpy(plan.in,in.data(),plan.size*sizeof(complex<double>),cudaMemcpyHostToDevice);
    plan.forward();
    std::vector<complex<double>> o(plan.size);
    Memcpy(o.data(),plan.out,plan.size*sizeof(complex<double>),cudaMemcpyDeviceToHost);
    for(size_t i = 0; i < plan.size; i++) {
        o[i].real(o[i].real() / (double)plan.size);
        o[i].imag(o[i].imag() / (double)plan.size);
    }       
    return o;
}
std::vector<complex<double>> ifft(FFTPlanDouble & plan, std::vector<complex<double>> & in)
{    
    Memcpy(plan.in,in.data(),plan.size*sizeof(complex<double>),cudaMemcpyHostToDevice);
    plan.backward();
    std::vector<complex<double>> o(plan.size);    
    Memcpy(o.data(),plan.out,plan.size*sizeof(complex<double>),cudaMemcpyDeviceToHost);
    return o;
}

std::vector<complex<float>> fft(FFTPlanFloat & plan, std::vector<complex<float>> & in)
{    
    Memcpy(plan.in,in.data(),plan.size*sizeof(complex<float>),cudaMemcpyHostToDevice);
    plan.forward();
    std::vector<complex<float>> o(plan.size);
    Memcpy(o.data(),plan.out,plan.size*sizeof(complex<float>),cudaMemcpyDeviceToHost);
    for(size_t i = 0; i < plan.size; i++) {
        o[i].real(o[i].real() / (float)plan.size);
        o[i].imag(o[i].imag() / (float)plan.size);
    }       
    return o;
}
std::vector<complex<float>> ifft(FFTPlanFloat & plan, std::vector<complex<float>> & in)
{    
    Memcpy(plan.in,in.data(),plan.size*sizeof(complex<float>),cudaMemcpyHostToDevice);
    plan.backward();
    std::vector<complex<float>> o(plan.size);    
    Memcpy(o.data(),plan.out,plan.size*sizeof(complex<float>),cudaMemcpyDeviceToHost);
    return o;
}

/*
convolution
crosscorrelation
autocorrelation
pearson coefficients
circular shift buffer
cublas multiply
fft resampler
stft

std::vector<float> conv(std::vector<float> x, std::vector<float> y)
{
    int size = x.size() + y.size() - 1;
    if(size % 2 != 0) size =  pow(2,log2(size)+1):size;
    R2CF forward(size);
    C2RF backward(size);
    int size1 = x.size % 2 != 0? pow(2,log2(x.size())+1):x.size();
    int size2 = y.size % 2 != 0? pow(2,log2(y.size())+1):y.size();
    std::vector<float> t1(log2(size1));
    std::vector<float> t2(log2(size2));
    forward.set_input(t1);
    t1 = forward.execute(t1);
    forward.set_input(t2);
    t2 = forward.execute(t2);
    std::vector<float> c = t1*t2;
    backward.set_input(c);
    c = backward.get_output();
    return c;
}
void blockconv(std::vector<float> h, std::vector<float> x, std::vector<float>& y, std::vector<float> & ytemp)    
{
    int i;
    int M = h.size();
    int L = x.size();
    y = conv(h,x);      
    for (i=0; i<M; i++) {
        y[i] += ytemp[i]; 
        ytemp[i] = y[i+L];
    }        
}

std::vector<float> deconv(std::vector<float> x, std::vector<float> y)
{
    int size = x.size() + y.size() - 1;
    if(size % 2 != 0) size =  pow(2,log2(size)+1):size;
    R2CF forward(size);
    C2RF backward(size);
    int size1 = x.size % 2 != 0? pow(2,log2(x.size())+1):x.size();
    int size2 = y.size % 2 != 0? pow(2,log2(y.size())+1):y.size();
    std::vector<float> t1(log2(size1));
    std::vector<float> t2(log2(size2));
    forward.set_input(t1);
    t1 = forward.execute(t1);
    forward.set_input(t2);
    t2 = forward.execute(t2);
    std::vector<float> c = t1/t2;
    backward.set_input(c);
    c = backward.get_output();
    return c;
}

std::vector<float> xcorr(std::vector<float> x, std::vector<float> y)
{
    int size = x.size() + y.size() - 1;
    if(size % 2 != 0) size =  pow(2,log2(size)+1):size;
    R2CF forward(size);
    C2RF backward(size);
    int size1 = x.size % 2 != 0? pow(2,log2(x.size())+1):x.size();
    int size2 = y.size % 2 != 0? pow(2,log2(y.size())+1):y.size();
    std::vector<float> t1(log2(size1));
    std::vector<float> t2(log2(size2));
    forward.set_input(t1);
    t1 = forward.execute(t1);
    forward.set_input(t2);
    t2 = forward.execute(t2);
    std::vector<float> c = conj(t1)*t2;
    backward.set_input(c);
    c = backward.get_output();
    return c;
}
*/

int main()
{
    std::vector<complex<double>> buffer(16),temp;
    for(size_t i= 0; i < 16; i++) buffer[i] = complex<double>(i,0);
    FFTPlanDouble plan(16);
    
    temp = fft(plan,buffer);
    std::cout << temp << std::endl;
    
    buffer = ifft(plan,temp);
    std::cout << buffer << std::endl;
}
