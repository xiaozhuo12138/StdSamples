#include <vector>
#include <complex>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstddef>
#include "Core/core_sndfile.hpp"

//O(N^2)
void dftProcess(const float *x, float *y, size_t N)
{
    size_t m, n;
    #pragma omp parallel for simd
    for (m = 0; m < N; m++) {
        y[2 * m] = 0.0f;
        y[2 * m +1] = 0.0f;        
        for (n = 0; n < N; n++) {
            const float c = cosf(2 * M_PI * m * n / N);
            const float s = sinf(2 * M_PI * m * n / N);

            y[2 * m]      += x[2 * n]      * c + x[2 * n + 1] * s;
            y[2 * m + 1]  += x[2 * n + 1]  * c - x[2 * n] * s;
        }

    }  
}

// inverse dft
void idftProcess(const float *x, float *y, size_t N)
{
    size_t m, n;
    #pragma omp parallel for simd
    for (m = 0; m < N; m++) {
        y[2 * m] = 0.0f;
        y[2 * m +1] = 0.0f;        
        for (n = 0; n < N; n++) {
            const float c = cosf(2 * M_PI * m * n / N);
            const float s = sinf(2 * M_PI * m * n / N);

            y[2 * m]      += x[2 * n]      * c + x[2 * n + 1] * s;
            y[2 * m + 1]  += x[2 * n + 1]  * c - x[2 * n] * s;
        }

        y[2 * m] = y[2 * m] / N;
        y[2 * m +1] = y[2 * m + 1] / N;
    }  
}

int main()
{
    /*
    SndFileReaderFloat file("Data/AcGtr.wav");
    std::vector<float> in(file.size());
    std::cout << in.size() << std::endl;
    std::vector<std::complex<float>> x(1024),y(1024);
    file.read(in.size(),in.data());
    double max;
    for(size_t i = 0; i < file.size() / 1024; i++)
    {
        for(size_t j = 0; j < 1024; j++) {
            x[i].real(in[j+i*1024]);            
            x[i].imag(0);
        }
        dftProcess((float*)x.data(),(float*)y.data(),1024);
        max = abs(y[0]);
        for(size_t j = 0; j < 1024; j++) {
            if(abs(y[j]) > max) max = abs(y[j]);
        }
        std::cout << max << std::endl;
    }
    */
    std::vector<std::complex<float>> a(16),b(16),c(16);
    int i = 1;
    std::generate(a.begin(),a.end(),[&i]() { return std::complex<float>(i++,0); });
    dftProcess((float*)a.data(),(float*)b.data(),16);
    for(size_t i = 0; i < c.size(); i++) {
        b[i].real(b[i].real()/16.0);
        b[i].imag(b[i].imag()/16.0);
        std::cout << b[i] << ",";
    }
    std::cout << std::endl;
    idftProcess((float*)b.data(),(float*)c.data(),16);
    for(size_t i = 0; i < c.size(); i++) std::cout << c[i] << ",";
    std::cout << std::endl;
    
}