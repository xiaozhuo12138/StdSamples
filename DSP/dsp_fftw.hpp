
// ipp
// mkl
// kfr
// FFTW
// FFTWPP
// minfft
// tinyfft
// oourafft
// kissfft
// pffft
// pocketfft
// cufft
// vkfft

template<typename T>
using sample_vector = std::vector<T>;

template<typename T>
using complex_vector = std::vector<std::complex<T>>;

template<typename T>
struct Window : public sample_vector<T>
{   
    sample_vector<T> window;

    Window() = default;
    Window(size_t i) { window.resize(i); }
    virtual ~Window() = default;

    void rectangle(size_t n) {
        this->resize(n);
        std::fill(this-begin(),this->end(),(T)1.0);
    }
    void hamming(size_t n) {
        this->resize(n);
        #pragma omp simd
        for(size_t i = 0; i < this->size(); i++)
        {
            (*this)[i] = 0.54 - (0.46 * std::cos(2*M_PI*(double)i/(double)n));
        }        
    }
    void hanning(size_t n)
    {
        this->resize(n);
        #pragma omp simd
        for(size_t i = 0; i < this->size(); i++)
        {
            (*this)[i] = 0.5*(1 - std::cos(2*M_PI*(double)i/(double)n));
        }        
    }
    void blackman(size_t n)
    {        
        this->resize(n);
        #pragma omp simd
        for(size_t i = 0; i < this->size(); i++)                    
            (*this)[i] = 0.42 - (0.5* std::cos(2*M_PI*i/(n)) + (0.08*std::cos(4*M_PI*i/n)));        
    }
    void blackmanharris(size_t n)
    {
        this->resize(n);
        #pragma omp simd        
        for(size_t i = 0; i < this->size(); i++)            
        {   
            double ci = (double) i / (double) n;
            (*this)[i] = 0.35875 
                    - 0.48829*std::cos(2*M_PI*(ci))
                    + 0.14128*std::cos(4.0*M_PI*(ci)) 
                    - 0.01168*std::cos(6.0*M_PI*(ci));
        }
    }
    void gaussian(size_t n)
    {
        this->resize(n);
        T a,b,c=0.5;
        #pragma omp simd        
        for(size_t i = 0; i < this->size(); i++)
        {
            a = ((double)i - c*(this->size()-1)/(std::sqrt(c)*this->size()-1));
            b = -c * std::sqrt(a);
            (*this)[i] = std::exp(b);
        }    
    }
    void welch(size_t n)
    {
        this->resize(n);
        #pragma omp simd
        for(size_t i = 0; i < this->size(); i++)
            (*this)[i] = 1.0 - std::sqrt((2.0*(double)i-(double)this->size()-1)/((double)this->size()));        
    }
    void parzen(size_t n)
    {
        this->resize(n);
        #pragma omp simd
        for(size_t i = 0; i < this->size(); i++)
            (*this)[i] = 1.0 - std::abs((2.0*(double)i-this->size()-1)/(this->size()));        
    }
    void tukey(size_t num_samples)
    {
        this->resize(num_samples);
        T value = (-1*(num_samples/2)) + 1;
        double n2 = (double)num_samples / 2.0;
        #pragma omp simd
        for(size_t i = 0; i < this->size(); i++)
        {    
            if(value >= 0 && value <= (alpha * (n2))) 
                (*this)i] = 1.0; 
            else if(value <= 0 && (value >= (-1*alpha*(n2)))) 
                (*this)[i] = 1.0;
            else 
                (*this)[i] = 0.5 * (1 + std::cos(M_PI *(((2.0*value)/(alpha*(double)num_samples))-1)))        ;
            value = value + 1;
        }     
    }
};

struct FFTPlanComplexDouble
{
    fftw_complex *  x;    
    fftw_complex *  y;
    size_t          size;
    fftw_plan       pf,pb;

    FFTPlanComplexDouble(size_t n) 
    {
        x = fftw_alloc_complex(n);
        y = fftw_alloc_complex(n);        
        size = n;
        pf = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        pb = fftw_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    ~FFTPlanComplexDouble()
    {
        if(x) fftw_free(x);
        if(y) fftw_free(y);
        if(pf) fftw_destroy_plan(pf);
        if(pb) fftw_destroy_plan(pb);
    }
    void setInput(const std::vector<std::complex<double>> & input)
    {
        for(size_t i = 0; i < size; i++) {
            x[i][0] = input[i].real();
            x[i][1] = input[i].imag();
        }
    }
    std::vector<std::complex<double>> getOutput() {
        std::vector<std::complex<double>> r(size);
        for(size_t i = 0; i < size; i++)        
            r[i] = std::complex<double>(y[i][0],y[i][1]);
        return r;
    }
    void normalize()
    {
        for(size_t i = 0; i < size; i++) {
            y[i][0] /= (double)size;    
            y[i][1] /= (double)size;
        }
    }
};


struct FFTPlanComplexFloat
{
    fftwf_complex * in;    
    fftwf_complex * out;
    size_t size;
    fftwf_plan p;

    FFTPlanComplexFloat(size_t n)     
    {
        x = fftwf_alloc_complex(n);
        y = fftwf_alloc_complex(n);        
        size = n;
        pf = fftwf_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        pb = fftwf_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    ~FFTPlanComplexFloat()
    {
        if(x) fftwf_free(x);
        if(y) fftwf_free(y);
        if(pf) fftwf_destroy_plan(pf);
        if(pb) fftwf_destroy_plan(pb);
    }
    void setInput(const std::vector<std::complex<float>> & input)
    {
        for(size_t i = 0; i < size; i++) {
            x[i][0] = input[i].real();
            x[i][1] = input[i].imag();
        }
    }
    std::vector<std::complex<float>> getOutput() {
        std::vector<std::complex<float>> r(size);
        for(size_t i = 0; i < size; i++)        
            r[i] = std::complex<float>(y[i][0],y[i][1]);
        return r;
    }
    void normalize()
    {
        for(size_t i = 0; i < size; i++) {
            y[i][0] /= (float)size;    
            y[i][1] /= (float)size;
        }
    }
};

struct FFTPlanRealDouble
{
    double * x;
    fftw_complex * y;
    size_t size;
    fftw_plan p;

    FFTPlanRealDouble(size_t n)     
    {
        x = fftw_alloc_real(n);
        y = fftw_alloc_complex(n);        
        size = n;
        pf = fftw_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);
        pb = fftw_plan_dft_c2r_1d(n, in, out, FFTW_ESTIMATE);
    }
    ~FFTPlanRealDouble()
    {
        if(x) fftw_free(x);
        if(y) fftw_free(y);
        if(pf) fftw_destroy_plan(pf);
        if(pb) fftw_destroy_plan(pb);
    }
    void setReal(const std::vector<double> & input)
    {
        memcpy(x,input.data(),size*sizeof(double));
    }
    void setComplex(const std::vector<std::complex<double>> & input)
    {
        for(size_t i = 0; i < size; i++) {
            y[i][0] = input[i].real();
            y[i][1] = input[i].imag();
        }
    }
    std::vector<double> getReal() {
        std::vector<double> r(size);
        memcpy(r.data(),x,size*sizeof(double));
        return r;
    }
    std::vector<std::complex<double>> getComplex() {
        std::vector<std::complex<double>> r(size);
        for(size_t i = 0; i < size; i++)        
            r[i] = std::complex<double>(y[i][0],y[i][1]);
        return r;
    }
    void normalize()
    {
        for(size_t i = 0; i < size; i++) {
            y[i][0] /= (double)size;    
            y[i][1] /= (double)size;
        }
    }
};
};
struct FFTPlanRealFloat
{
    float * x;
    fftwf_complex * y;
    size_t size;
    fftwf_plan p;

    FFTPlanRealFloat(size_t n)     
    {
        x = fftwf_alloc_real(n);
        y = fftwf_alloc_complex(n);        
        size = n;
        pf = fftwf_plan_dft_r2c_1d(n, x, y, FFTW_ESTIMATE);
        pb = fftwf_plan_dft_c2r_1d(n, y, x, FFTW_ESTIMATE);
    }
    ~FFTPlanRealFloat()
    {
        if(x) fftwf_free(x);
        if(y) fftwf_free(y);
        if(pf) fftwf_destroy_plan(pf);
        if(pb) fftwf_destroy_plan(pb);
    }
    void setReal(const std::vector<float> & input)
    {
        memcpy(x,input.data(),size*sizeof(float));
    }
    void setComplex(const std::vector<std::complex<float>> & input)
    {
        for(size_t i = 0; i < size; i++) {
            y[i][0] = input[i].real();
            y[i][1] = input[i].imag();
        }
    }
    std::vector<float> getReal() {
        std::vector<float> r(size);
        memcpy(r.data(),x,size*sizeof(float));
        return r;
    }
    std::vector<std::complex<float>> getComplex() {
        std::vector<std::complex<float>> r(size);
        for(size_t i = 0; i < size; i++)        
            r[i] = std::complex<float>(y[i][0],y[i][1]);
        return r;
    }
    void normalize()
    {
        for(size_t i = 0; i < size; i++) {
            y[i][0] /= (float)size;    
            y[i][1] /= (float)size;
        }
    }
};


std::vector<std::complex<double>> 
fft(FFTPlanComplexDouble & plan, const std::vector<std::complex<double>> & v)
{
    plan.setInput(v);
    fftw_execute(plan.pf);
    plan.normalize();
    return plan.getOutput();
}
std::vector<std::complex<double>> 
ifft(FFTPlanComplexDouble & plan, const std::vector<std::complex<double>> & v)
{
    plan.setInput(v);
    fftw_execute(plan.pb);    
    return plan.getOutput();
}

std::vector<std::complex<float>> 
fft(FFTPlanComplexFloat & plan, const std::vector<std::complex<float>> & v)
{
    plan.setInput(v);
    fftwf_execute(plan.pf);
    plan.normalize();
    return plan.getOutput();
}
std::vector<std::complex<float>> 
ifft(FFTPlanComplexFloat & plan, const std::vector<std::complex<float>> & v)
{
    plan.setInput(v);
    fftwf_execute(plan.pb);    
    return plan.getOutput();
}
