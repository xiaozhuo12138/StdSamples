#pragma once

#include <Eigen/Eigen>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <valarray>
#include <fftw3.h>


typedef float SampleType; 

//#include "TinyEigen.hpp"
/*
template<typename T>
using EigenArray   = Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using EigenArray2D = Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using EigenVector   = Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using EigenMatrix = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
*/
namespace AudioDSP::Samples
{   
    template<typename T>
    struct SampleVector : public Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>
    {
        using M = Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>;

        SampleVector() = default;
        SampleVector(size_t n) : Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>(n) {}
        SampleVector(const SampleVector<T> & v) { *this = v; }
        SampleVector(const std::vector<T> & d) {
            resize(d.size());
            memcpy(data(),d.data(),d.size()*sizeof(T));
        }

        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::setLinSpaced;

        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator [];
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator ();
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator =;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator +=;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator -=;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator *=;        
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator +;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator -;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator *;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator /;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator <<;
        //using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator <;
        //using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator <=;


        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::size;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::resize;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::fill;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::data;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::cols;
        
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::setZero;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::setOnes;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::setRandom;
        
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::sum;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::dot;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::cross;

        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::real;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::imag;
        
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::cwiseProduct;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::cwiseQuotient;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::cwiseSqrt;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::cwiseInverse;        
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::cwiseMin;        
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::cwiseMax; 
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::cwiseAbs;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::cwiseAbs2;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::unaryExpr;
        using Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>::array;
        


        

        // mean
        // min
        // max
        // stddev
        // rms
        // correlation
        // autocorrelation
        // count(n)
        // 

        void print() {
            std::cout << *this << std::endl;
        }
    };

    template<typename T>
    struct SampleMatrix : public Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>
    {
        SampleMatrix() = default;
        SampleMatrix(size_t i, size_t j) : Eigen::Matrix<T,1,Eigen::Dynamic,Eigen::RowMajor>(i,j) {}
        SampleMatrix(const SampleMatrix<T> & m) { *this = m; }
        SampleMatrix(const std::vector<T> & d,size_t i, size_t j) {
            resize(i,j);
            for(size_t x = 0; x < i; x++)
                for(size_t y = 0; y < j; y++)
                    (*this)(x,y) = d[x*j + y];
        }

        
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator ();
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator =;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator +=;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator -=;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator *=;        
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator +;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator -;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator *;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator /;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator <<;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator <;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator <=;


        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::size;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::resize;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::fill;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::data;        
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::setZero;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::setOnes;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::setRandom;            

        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::setIdentity;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::head;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::tail;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::segment;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::block;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::row;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::col;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::rows;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::cols;        
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::leftCols;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::middleCols;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::rightCols;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::topRows;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::middleRows;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::bottomRows;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::topLeftCorner;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::topRightCorner;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::bottomLeftCorner;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::bottomRightCorner;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::seq;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::seqN;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::A;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::v;        
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::adjoint;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::transpose;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::diagonal;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::eval;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::asDiagonal;        
        
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::replicate;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::reshaped;        
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::select;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::cwiseProduct;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::cwiseQuotient;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::cwiseSqrt;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::cwiseInverse;        
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::cwiseMin;        
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::cwiseMax; 
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::cwiseAbs;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::cwiseAbs2;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::unaryExpr;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::array;

        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::minCoeff;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::maxCoeff;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::sum;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::colwise;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::rowwise;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::trace;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::all;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::any;

        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::norm;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::squaredNorm;
        
        
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::real;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::imag;

        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::ldlt;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::llt;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::lu;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::qr;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::svd;
        using Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::eigenvalues;        
        
        void print() {
            std::cout << *this << std::endl;
        }
    };
    template<typename T>
    struct SampleArray : public Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>
    {        
        
        size_t channels;

        SampleArray() {
            channels = 1;
        }
        SampleArray(size_t size, size_t channels = 1) {
            resize(size * channels);
            this->channels = channels;
        }
        SampleArray(const std::vector<T> & s, size_t chans = 1) {
            resize(s.size());
            memcpy(data(),s.data(),s.size()*sizeof(T));
            channels = chans;
        }
        SampleArray(const T * ptr, size_t n, size_t chans = 1) {
            resize(n*chans);
            memcpy(data(),ptr,n*chans*sizeof(T));
            channels = chans;
        }
        SampleArray(const SampleArray<T> & s) {
            (*this) = s;
            channels = s.channels;
        }    
        SampleArray(const Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor> & v, size_t ch = 1) {
            (*this) = v;
            channels = ch;
        }
        SampleArray(size_t channels, const Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>& a) {
            (*this) = a;
            this->channels = channels;
        }

        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator [];
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator ();
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator =;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator +=;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator -=;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator *=;        
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator +;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator -;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator *;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator /;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator <<;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator <;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::operator <=;

        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::data;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::size;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::cols;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::resize;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::fill;        
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::setZero;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::setOnes;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::setRandom;        
        
        //using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::vector;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::inverse;        
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::pow;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::square;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::cube;        
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::sqrt;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::rsqrt;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::exp;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::log;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::log1p;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::log10;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::max;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::min;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::abs;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::abs2;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::sin;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::cos;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::tan;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::asin;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::acos;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::atan;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::sinh;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::cosh;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::tanh;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::asinh;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::acosh;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::atanh;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::ceil;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::floor;
        using Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::round;
        
        
        SampleArray<T> clamp(T min = (T)0.0, T max = (T)1.0) {
            SampleArray<T> r(*this,channels);
            r = SampleArray<T>(r.cwiseMin(max).cwiseMax(min));
            return r;
        }
        

        T&   get_stride(size_t ch, size_t pos) { return (*this)[pos*channels + ch]; }
        void set_stride(size_t ch, size_t pos, const T val) { (*this)[pos*channels + ch] = val; } 
        
        void swap_stereo_channels() {
            assert(channels == 2);
            for(size_t i = 0; i < size(); i+= channels) {
                T temp = (*this)[i];
                (*this)[i] = (*this)[i+1];
                (*this)[i+1] = temp;
            }
        }

        void set_data(size_t n, size_t channels, T * samples)
        {
            resize(n);
            this->channels = channels;
            memcpy(data(),samples,n*sizeof(T));
        }
        void copy_data(T * samples) {
            memcpy(samples,data(),size()*sizeof(T));
        }
        void set_channel(size_t ch, T * samples) {
            size_t x = 0;
            for(size_t i = ch; i < size(); i+=channels) (*this)[i] = samples[x++];
        }

        
        size_t num_channels() const { return channels; } 
        size_t samples_per_channel() const { return size() / num_channels(); }
                
        
        T& operator()(size_t i, size_t ch) {
            Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor> & r = *this;
            return r[i*channels + ch];
        }
        T operator()(size_t i, size_t ch) const {
            const Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor> & r = *this;
            return r[i*channels + ch];
        }
         
        SampleArray<T> get_channel(size_t channel) {            
            SampleArray<T> r(samples_per_channel());
            size_t x=0;
            for(size_t i = channel; i < size(); i+=channels)
                r[x++] = (*this)[i];
            return r;
        }
        void get_channel(size_t channel, T * r) {                        
            size_t x=0;
            for(size_t i = channel; i < size(); i+=channels)
                r[x++] = (*this)[i];            
        }
        SampleArray<T> get_channel(size_t channel) const {            
            SampleArray<T> r(samples_per_channel());
            size_t x=0;
            for(size_t i = channel; i < size(); i+=channels)
                r[x++] = (*this)[i];
            return r;
        }
        void set_channel(const SampleArray<T> & v, size_t ch) {            
            size_t x = 0;
            for(size_t i = ch; i < size(); i+=channels) (*this)[i] = v[x++];
        }
        void set_channel(const T* v, size_t ch) {            
            size_t x = 0;
            for(size_t i = ch; i < size(); i+=channels) (*this)[i] = v[x++];
        }
        void make_stereo(const T * in) {
            set_channel(in,0);
            set_channel(in,1);
        }
        void make_stereo(const T * left, const T * right) {
            set_channel(left,0);
            set_channel(right,1);
        }

        void set_channels(size_t c) {
            channels = c;
            resize(size()*c);
        }
        void resize(size_t n) {
            Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::resize(n);
        }
        void resize(size_t s, size_t c) {
            channels = c;
            Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>::resize(s * c);
        }

        SampleArray<T> get_channel_count(size_t ch, size_t pos, size_t n) {
            SampleArray<T> r(n);
            memcpy(r.data(), get_channel(ch).data()+pos, n*sizeof(T));
            return r;
        }

        bool operator == (const SampleArray<T> & s) {
            return s.channels == channels && size() == s.size();
        }

        void copy(SampleArray<T> & in, size_t block) {
            resize(in.size());
            memcpy(data(),in.data(),block*sizeof(T));
        }
        void copy(SampleArray<T> & in, size_t pos, size_t n) {
            resize((pos+n));
            memcpy(data(),in.data()+pos,n*sizeof(T));
        }
        void copy_from(const T* in, size_t block) {            
            memcpy(data(),in,block*sizeof(T));
        }
        void copy_to(T* in, size_t block) {            
            memcpy(in,data(),block*sizeof(T));
        }

        SampleArray<T> slice(size_t i, size_t n) {
            SampleArray<T> r(n);
            memcpy(r.data(), data()+i, n * sizeof(T));
            return r;
        }
        
        SampleArray<T> pan(SampleType pan) {
            SampleType pan_mapped = ((pan+1.0)/2.0) * (M_PI/2.0);
            SampleArray<T> r(size(),2);
            for(size_t i = 0; i < size(); i+=channels)
            {
                r[i] = (*this)[i] * std::sin(pan_mapped);
                if(num_channels() == 2)
                    r[i+1] = (*this)[i+1] * std::cos(pan_mapped);
                else
                    r[i+1] = (*this)[i] * std::cos(pan_mapped);
            }
            return r;        
        }
        SampleArray<T> stride_slice(const SampleArray<T> & in, size_t stride, size_t pos, size_t n)
        {
            SampleArray<T> r(n/stride);
            size_t x = pos;
            for(size_t i = 0; i < n; i++) {
                r[i] = (*this)[x];
                x += stride;
            } 
            return r;
        }
        
        void print() {
            std::cout << "Size=" << size() << std::endl;
            std::cout << "Channel=" << num_channels() << std::endl;
            std::cout << "Samples per channel=" << samples_per_channel() << std::endl;
            for(size_t i = 0; i < size(); i++)
                std:: cout << (*this)[i] << ",";
            std::cout << std::endl;
        }

    };

    template<typename T>
    struct SampleArray2D : public Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>
    {        
        SampleArray2D() = default;
        
        SampleArray2D(size_t chan, size_t samps) {
            resize(chan,samps);
        }
        SampleArray2D(const SampleArray<T> & v) {
            resize(v.num_channels(), v.size()/v.num_channels());
            for(size_t i = 0; i < v.num_channels(); i++)             
                this->row(i) = v.get_channel(i);
        }
        SampleArray2D(const Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & c) {
            (*this) = c;
        }
        SampleArray2D(const SampleArray2D<T> & m) {
            (*this) = m.rows();
        }

        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator [];
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator ();
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator =;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator +=;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator -=;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator *=;        
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator +;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator -;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator *;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator /;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator <<;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator <;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::operator <=;

        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::data;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::size;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::resize;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::fill;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::rows;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::cols;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::row;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::col;
        
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::setZero;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::setOnes;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::setRandom;        
        
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::matrix;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::inverse;        
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::pow;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::square;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::cube;        
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::sqrt;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::rsqrt;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::exp;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::log;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::log1p;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::log10;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::max;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::min;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::abs;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::abs2;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::sin;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::cos;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::tan;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::asin;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::acos;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::atan;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::sinh;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::cosh;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::tanh;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::asinh;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::acosh;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::atanh;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::ceil;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::floor;
        using Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::round;
        
        
        void deinterleave(const SampleArray<T> & s) {
            
            esize(s.num_channels(),s.samples_per_channel());            
            
            for(size_t i = 0; i < s.num_channels(); i++) {
                this->row(i) = s.get_channel(i);
            }
            
        }    

        SampleArray<T> interleave() { 
            SampleArray<T> r;
            r.resize(cols() * cols());
            for(size_t i = 0; i < rows(); i++)            
            {
                SampleArray2D<T> tmp(1,(*this)(i));            
                r.set_channel(tmp,i);
            }
            return r;
        }

        size_t num_channels() const { 
            return rows(); 
        }
        size_t samples_per_channel() const {
            return cols();
        }
        SampleArray<T> get_channel(size_t c) { 
            SampleArray<T> tmp(1,row(c));
            return tmp;
        }
        void set_channel(size_t c, const SampleArray<T> & s) 
        { 
            row(c) = s;
        }

        void print() {
            std::cout << "Size=" << size() << std::endl;
            std::cout << "Channel=" << num_channels() << std::endl;
            std::cout << "Samples per channel=" << samples_per_channel() << std::endl;
            for(size_t i = 0; i < size(); i++)
                std:: cout << (*this)[i] << ",";
            std::cout << std::endl;
        }
    };

    template<typename T>  SampleArray<T> abs( SampleArray<T> & m) { return m.abs(); }
    template<typename T>  SampleArray<T> abs2( SampleArray<T> & m) { return m.abs2(); }
    template<typename T>  SampleArray<T> inverse( SampleArray<T> & m) { return m.inverse(); }
    template<typename T>  SampleArray<T> exp( SampleArray<T> & m) { return m.exp(); }
    template<typename T>  SampleArray<T> log( SampleArray<T> & m) { return m.log(); }
    template<typename T>  SampleArray<T> log1p( SampleArray<T> & m) { return m.log1p(); }
    template<typename T>  SampleArray<T> log10( SampleArray<T> & m) { return m.log10(); }
    template<typename T>  SampleArray<T> pow( SampleArray<T> & m,  SampleArray<T> & p) { return m.pow(p); }
    template<typename T>  SampleArray<T> pow( SampleArray<T> & m,  T p) { return m.pow(p); }
    template<typename T>  SampleArray<T> sqrt( SampleArray<T> & m) { return m.sqrt(); }
    template<typename T>  SampleArray<T> rsqrt( SampleArray<T> & m) { return m.rsqrt(); }
    template<typename T>  SampleArray<T> square( SampleArray<T> & m) { return m.square(); }
    template<typename T>  SampleArray<T> sin( SampleArray<T> & m) { return m.sin(); }
    template<typename T>  SampleArray<T> cos( SampleArray<T> & m) { return m.cos(); }
    template<typename T>  SampleArray<T> tan( SampleArray<T> & m) { return m.tan(); }
    template<typename T>  SampleArray<T> asin( SampleArray<T> & m) { return m.asin(); }
    template<typename T>  SampleArray<T> acos( SampleArray<T> & m) { return m.acos(); }
    template<typename T>  SampleArray<T> atan( SampleArray<T> & m) { return m.atan(); }
    template<typename T>  SampleArray<T> sinh( SampleArray<T> & m) { return m.sinh(); }
    template<typename T>  SampleArray<T> cosh( SampleArray<T> & m) { return m.cosh(); }
    template<typename T>  SampleArray<T> tanh( SampleArray<T> & m) { return m.tanh(); }
    template<typename T>  SampleArray<T> ceil( SampleArray<T> & m) { return m.ceil(); }
    template<typename T>  SampleArray<T> floor( SampleArray<T> & m) { return m.floor(); }
    template<typename T>  SampleArray<T> round( SampleArray<T> & m) { return m.round(); }

    template<typename T>  SampleArray2D<T> abs( SampleArray2D<T> & m) { return m.abs(); } 
    template<typename T>  SampleArray2D<T> abs2( SampleArray2D<T> & m) { return m.abs2(); }
    template<typename T>  SampleArray2D<T> inverse( SampleArray2D<T> & m) { return m.inverse(); }
    template<typename T>  SampleArray2D<T> exp( SampleArray2D<T> & m) { return m.exp(); }
    template<typename T>  SampleArray2D<T> log( SampleArray2D<T> & m) { return m.log(); }
    template<typename T>  SampleArray2D<T> log1p( SampleArray2D<T> & m) { return m.log1p(); }
    template<typename T>  SampleArray2D<T> log10( SampleArray2D<T> & m) { return m.log10(); }
    template<typename T>  SampleArray2D<T> pow( SampleArray2D<T> & m,  T &p) { return m.pow(p); }
    template<typename T>  SampleArray2D<T> sqrt( SampleArray2D<T> & m) { return m.sqrt(); }
    template<typename T>  SampleArray2D<T> rsqrt( SampleArray2D<T> & m) { return m.rsqrt(); }
    template<typename T>  SampleArray2D<T> square( SampleArray2D<T> & m) { return m.square(); }
    template<typename T>  SampleArray2D<T> sin( SampleArray2D<T> & m) { return m.sin(); }
    template<typename T>  SampleArray2D<T> cos( SampleArray2D<T> & m) { return m.cos(); }
    template<typename T>  SampleArray2D<T> tan( SampleArray2D<T> & m) { return m.tan(); }
    template<typename T>  SampleArray2D<T> asin( SampleArray2D<T> & m) { return m.asin(); }
    template<typename T>  SampleArray2D<T> acos( SampleArray2D<T> & m) { return m.acos(); }
    template<typename T>  SampleArray2D<T> atan( SampleArray2D<T> & m) { return m.atan(); }
    template<typename T>  SampleArray2D<T> sinh( SampleArray2D<T> & m) { return m.sinh(); }
    template<typename T>  SampleArray2D<T> cosh( SampleArray2D<T> & m) { return m.cosh(); }
    template<typename T>  SampleArray2D<T> tanh( SampleArray2D<T> & m) { return m.tanh(); }
    template<typename T>  SampleArray2D<T> ceil( SampleArray2D<T> & m) { return m.ceil(); }
    template<typename T>  SampleArray2D<T> floor( SampleArray2D<T> & m) { return m.floor(); }
    template<typename T>  SampleArray2D<T> round( SampleArray2D<T> & m) { return m.round(); }
    
};