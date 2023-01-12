#pragma once

#include <iostream>
#include <ccomplex>
#include <complex>
#include <vector>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <random>
#include <functional>

#include "StdNoise.hpp"

#include "cppmkl/cppmkl_allocator.h"
#include "cppmkl/cppmkl_vml.h"
#include "cppmkl/cppmkl_cblas.h"
#include "cppmkl/matrix.h"

//#include "samples/Allocator.hpp"
//#include "Octopus.hpp"

namespace Casino::MKL
{
    
    template<typename T> using vector_base = std::vector<T, cppmkl::cppmkl_allocator<T> >;

    template<typename T>
    struct Vector : public vector_base<T>
    {                
        using vector_base<T>::size;
        using vector_base<T>::resize;
        using vector_base<T>::data;
        using vector_base<T>::push_back;
        using vector_base<T>::pop_back;
        using vector_base<T>::front;
        using vector_base<T>::back;
        using vector_base<T>::at;
        using vector_base<T>::operator [];
        using vector_base<T>::operator =;

        Vector() = default;
        Vector(size_t n) { assert(n > 0); resize(n); zero(); }
        Vector(const Vector<T> & v) { (*this) = v; }
        Vector(const std::vector<T> & v) {
            resize(v.size());
            memcpy(this->data(),v.data(),v.size()*sizeof(T));
        }
        /*
        Vector(const VectorXf& v) {
            resize(v.size());
            for(size_t i = 0; i < v.size(); i++) (*this)[i] = v[i];
        }
        Vector(const VectorXd& v) {
            resize(v.size());
            for(size_t i = 0; i < v.size(); i++) (*this)[i] = v[i];
        }
        */
        T min() { return *std::min_element(this->begin(),this->end()); }
        T max() { return *std::max_element(this->begin(),this->end()); }

        T& operator()(size_t i) { return (*this)[i]; }
        T operator()(size_t i) const { return (*this)[i]; }

        size_t min_index() { return std::distance(this->begin(), std::min_element(this->begin(),this->end())); }
        size_t max_index() { return std::distance(this->begin(), std::max_element(this->begin(),this->end())); }
                
        Vector<T>& operator +=  (const Vector<T> & v) { 
            cppmkl::vadd(*this,v,*this);
            return *this;
        }
        Vector<T>& operator -=  (const Vector<T> & v) { 
            cppmkl::vsub(*this,v,*this);
            return *this;
        }
        Vector<T>& operator *=  (const Vector<T> & v) { 
            cppmkl::vmul(*this,v,*this);
            return *this;
        }
        Vector<T>& operator /=  (const Vector<T> & v) { 
            cppmkl::vdiv(*this,v,*this);
            return *this;        
	    }
        Vector<T>& operator +=  (const T & x) { 
            Vector<T> v(size());
            v.fill(x);
            cppmkl::vadd(*this,v,*this);
            return *this;
        }
        Vector<T>& operator -=  (const T & x) { 
            Vector<T> v(size());
            v.fill(x);
            cppmkl::vsub(*this,v,*this);
            return *this;
        }
        Vector<T>& operator *=  (const T & x) { 
            Vector<T> v(size());
            v.fill(x);
            cppmkl::vmul(*this,v,*this);
            return *this;
        }
        Vector<T>& operator /=  (const T & x) { 
            Vector<T> v(size());
            v.fill(x);
            cppmkl::vdiv(*this,v,*this);
            return *this;        
	    }

        Vector<T> operator - () {
            Vector<T> r(*this);
            return (T)-1.0 * r;
        }

        Vector<T>& operator = (const T v)
        {
            fill(v);
            return *this;
        }
                
        void zero() {
            memset(data(),0x00,size()*sizeof(T));
        }
        void zeros() { zero(); }
        void fill(T x) {
            for(size_t i = 0; i < size(); i++) (*this)[i] = x;
        }
        void ones() {
            fill((T)1);
        }
        void random(T min = T(0), T max = T(1)) {
            Default noise;
            for(size_t i = 0; i < size(); i++) (*this)[i] = noise.random(min,max);
        }
        void randu(T min = T(0), T max = T(1)) { random(min,max); }
        void randn(T min = T(0), T max = T(1)) { random(min,max); }

        
        void clamp(T min = T(-1), T max = T(1)) {
            for(size_t i = 0; i < size(); i++)
            {
                if((*this)[i] < min) (*this)[i] = min;
                if((*this)[i] < max) (*this)[i] = max;
            }
        }

        void set_size(size_t n) { resize(n); }


        Vector<T> eval() {
            return Vector<T>(*this); 
        }
        Vector<T> slice(size_t start, size_t len) {
            Vector<T> x(len);
            memcpy(x.data(),data()+start,len*sizeof(T));
            return x;
        }

        void print() {
            std::cout << "Vector[" << size() << "]=";
            for(size_t i = 0; i < size()-1; i++) std::cout << (*this)[i] << ",";
            std::cout << (*this)[size()-1] << std::endl;
        }
        
        // eigen compatibility
        void setZero() { zero(); }
        void setOnes() { ones(); }
        void setRandom() { random(); }                



        Vector<T> cwiseMin(T min) {
            Vector<T> r(size());
            for(size_t i = 0; i < size(); i++)
                if((*this)[i] > min) r[i] = min;
                else r[i] = (*this)[i];
            return r;
        }
        Vector<T> cwiseMax(T max) {
            Vector<T> r(size());
            for(size_t i = 0; i < size(); i++)
                if((*this)[i] < max) r[i] = max;
                else r[i] = (*this)[i];
            return r;
        }
        Vector<T> cwiseProd(Vector<T> & x) {
            Vector<T> r(*this);
            r *= x;
            return r;
        }
        Vector<T> cwiseAdd(Vector<T> & x) {
            Vector<T> r(*this);
            r += x;
            return r;
        }
        Vector<T> cwiseSub(Vector<T> & x) {
            Vector<T> r(*this);
            r -= x;
            return r;
        }
        Vector<T> cwiseDiv(Vector<T> & x) {
            Vector<T> r(*this);
            r /= x;
            return r;
        }

        T sum() {        
            return cppmkl::cblas_asum(size(), data(),1);                
        }
        T prod() {
            T p = (T)1.0;
            for(size_t i = 0; i < size(); i++) p *= (*this)[i];
            return p;
        }
        T mean() {
            return sum()/(T)size();
        }
        T geometric_mean() {
            T r = prod();
            return std::pow(r,(T)1.0/(T)size());
        }
        T harmonic_mean() {
            T r = 1.0/sum();
            return size()*std::pow(r,(T)-1.0);
        }
        T stddev() {
            T u = sum();
            T r = 0;
            for(size_t i = 0; i < size(); i++)
                r += std::pow((*this)[i]-u,2.0);
            return r/(T)size();
        }
    };

    template<typename T>
    struct ComplexVector : public vector_base<std::complex<T>>
    {                
        using vecbase = vector_base<std::complex<T>>;
        

        using vecbase::size;
        using vecbase::resize;
        using vecbase::data;
        using vecbase::push_back;
        using vecbase::pop_back;
        using vecbase::front;
        using vecbase::back;
        using vecbase::at;
        using vecbase::operator [];
        using vecbase::operator =;

        ComplexVector() = default;
        ComplexVector(size_t i) : vecbase(i) {}
        ComplexVector(const vecbase& v) : vecbase(v) {}
        ComplexVector(const ComplexVector<T> & v) : vecbase(v) {}

        void fill(const std::complex<T>& c) {
            for(size_t i = 0; i < size(); i++) (*this)[i] = c;
        }

        ComplexVector<T>& operator +=  (const ComplexVector<T> & v) { 
            cppmkl::vadd(*this,v,*this);
            return *this;
        }
        ComplexVector<T>& operator -=  (const ComplexVector<T> & v) { 
            cppmkl::vsub(*this,v,*this);
            return *this;
        }
        ComplexVector<T>& operator *=  (const ComplexVector<T> & v) { 
            cppmkl::vmul(*this,v,*this);
            return *this;
        }
        ComplexVector<T>& operator /=  (const ComplexVector<T> & v) { 
            cppmkl::vdiv(*this,v,*this);
            return *this;        
	    }

        ComplexVector<T>& operator +=  (const std::complex<T> & x) { 
            ComplexVector<T> v(size());
            v.fill(x);
            cppmkl::vadd(*this,v,*this);
            return *this;
        }
        ComplexVector<T>& operator -=  (const std::complex<T> & x) { 
            ComplexVector<T> v(size());
            v.fill(x);
            cppmkl::vsub(*this,v,*this);
            return *this;
        }
        ComplexVector<T>& operator *=  (const std::complex<T> & x) { 
            ComplexVector<T> v(size());
            v.fill(x);
            cppmkl::vmul(*this,v,*this);
            return *this;
        }
        ComplexVector<T>& operator /=  (const std::complex<T> & x) { 
            ComplexVector<T> v(size());
            v.fill(x);
            cppmkl::vdiv(*this,v,*this);
            return *this;        
	    }

        ComplexVector<T> operator - () {
            ComplexVector<T> r(*this);
            return std::complex<T>(-1.0,0) * r;
        }
        ComplexVector<T>& operator = (const std::complex<T>& v)
        {
            fill(v);
            return *this;
        }

        Vector<T> real() {
            Vector<T> r(size());
            for(size_t i = 0; i < size(); i++) r[i] = (*this)[i].real();
            return r;
        }
        Vector<T> imag() {
            Vector<T> r(size());
            for(size_t i = 0; i < size(); i++) r[i] = (*this)[i].imag();
            return r;
        }
        void real(const Vector<T> & r) {
            for(size_t i = 0; i < size(); i++) (*this)[i].real(r[i]);
        }
        void imag(const Vector<T> & r) {
            for(size_t i = 0; i < size(); i++) (*this)[i].imag(r[i]);
        }

        Vector<T> abs() {
            Vector<T> r(size());
            for(size_t i = 0; i < size(); i++) r[i] = std::abs((*this)[i]);
            return r;
        }
        Vector<T> arg() {
            Vector<T> r(size());
            for(size_t i = 0; i < size(); i++) r[i] = std::abs((*this)[i]);
            return r;
        }
        ComplexVector<T> conj() {
            ComplexVector<T> r(size());
            for(size_t i = 0; i < size(); i++) r[i] = std::conj((*this)[i]);
            return r;
        }
        ComplexVector<T> proj() {
            ComplexVector<T> r(size());
            for(size_t i = 0; i < size(); i++) r[i] = std::proj((*this)[i]);
            return r;
        }
        Vector<T> norm() {
            Vector<T> r(size());
            for(size_t i = 0; i < size(); i++) r[i] = std::norm((*this)[i]);
            return r;
        }

        void print() {
            std::cout << "Vector[" << size() << "]=";
            for(size_t i = 0; i < size()-1; i++) std::cout << (*this)[i] << ",";
            std::cout << (*this)[size()-1] << std::endl;
        }        
    };
    
    template<typename T>
    std::ostream& operator << (std::ostream & o, const Vector<T> & v )
    {
        for(size_t i = 0; i < v.size(); i++)
        {               
            o << v[i] << ",";            
        }
        o << std::endl;
        return o;
    }
        
    template<typename T> struct Matrix;
    template<typename T>
    struct MatrixView 
    {
        Matrix<T> * matrix;
        size_t row;
        MatrixView(Matrix<T> * m, size_t r) {
            matrix = m;
            row = r;
        }

        T& operator[](size_t i);
        T  __getitem__(size_t i);
        void __setitem__(size_t i, T x);
    };
    
    template<typename T>
    struct Matrix  : public vector_base<T>
    {                
        using vector_base<T>::size;
        using vector_base<T>::resize;
        using vector_base<T>::data;
        using vector_base<T>::push_back;
        using vector_base<T>::pop_back;
        using vector_base<T>::front;
        using vector_base<T>::back;
        using vector_base<T>::at;
        using vector_base<T>::operator [];
        using vector_base<T>::operator =;

        size_t M;
        size_t N;

        Matrix() { M = N = 0; }
        Matrix(size_t m, size_t n) {
            resize(m,n);
            assert(M > 0);
            assert(N > 0);
        }
        Matrix(const Matrix<T> & m) {
            
            resize(m.M,m.N);
            memcpy(data(),m.data(),size()*sizeof(T));            
        }
        Matrix(T * ptr, size_t m, size_t n)
        {
            resize(m,n);            
            memcpy(data(),ptr,m*n*sizeof(T));
        }
        Matrix<T>& operator = (const T v) {
            fill(v);
            return *this;
        }

        Matrix<T>& operator = (const Matrix<T> & m) {            
            resize(m.M,m.N);
            memcpy(data(),m.data(),size()*sizeof(T));            
            return *this;
        }
        Matrix<T>& operator = (const Vector<T> & m) {            
            resize(1,m.size());
            memcpy(data(),m.data(),size()*sizeof(T));            
            return *this;
        }

        T& operator()(size_t i) { return (&this)[i]; }
        T  operator()(size_t i) const { return (&this)[i]; }

        size_t rows() const { return M; }
        size_t cols() const { return N; }
        
        
        MatrixView<T> __getitem__(size_t row) { return MatrixView<T>(this,row); }

        T get(size_t i,size_t j) { return (*this)(i,j); }
        void set(size_t i, size_t j, T v) { (*this)(i,j) = v; }

        Matrix<T> row(size_t m) { 
            Matrix<T> r(1,cols());
            for(size_t i = 0; i < cols(); i++) r(0,i) = (*this)(m,i);
            return r;
        }
        Matrix<T> col(size_t c) { 
            Matrix<T> r(cols(),1);
            for(size_t i = 0; i < rows(); i++) r(i,0) = (*this)(i,c);
            return r;
        }
        Vector<T> row_vector(size_t m) { 
            Vector<T> r(cols());
            for(size_t i = 0; i < cols(); i++) r[i] = (*this)(m,i);
            return r;
        }
        Vector<T> col_vector(size_t c) { 
            Vector<T> r(cols());
            for(size_t i = 0; i < rows(); i++) r[i] = (*this)(i,c);
            return r;
        }
        void row(size_t m, const Vector<T> & v)
        {            
            for(size_t i = 0; i < cols(); i++) (*this)(m,i) = v[i];
        }
        void col(size_t n, const Vector<T> & v)
        {
            for(size_t i = 0; i < rows(); i++) (*this)(i,n) = v[i];
        }
        
        void resize(size_t r, size_t c) {
            M = r;
            N = c;
            resize(r*c);            
        }

        T& operator()(size_t i, size_t j) {             
            return (*this)[i*N + j]; }

        T  operator()(size_t i, size_t j) const {             
            return (*this)[i*N + j]; }
        
        
        std::ostream& operator << (std::ostream & o )
        {
            for(size_t i = 0; i < rows(); i++)
            {
                for(size_t j = 0; j < cols(); j++)
                    std::cout << (*this)(i,j) << ",";
                std::cout << std::endl;
            }
            return o;
        }

        Matrix<T> operator - () {
            Matrix<T> r(*this);
            return T(-1.0)*r;
        }
        Matrix<T> addToEachRow(Vector<T> & v) {
            Matrix<T> r(*this);
            for(size_t i = 0; i < M; i++)
            {
                for(size_t j = 0; j < N; j++)
                {
                    r(i,j) += v[j];
                }
            }
            return r;
        }
        Matrix<T> addToEachRow(Matrix<T> & v) {
            Matrix<T> r(*this);
            for(size_t i = 0; i < M; i++)
            {
                for(size_t j = 0; j < N; j++)
                {
                    r(i,j) += v(0,j);
                }
            }
            return r;
        }
        Matrix<T> eval() {
            Matrix<T> r(*this);
            return r;
        }
        void printrowscols() const {
            std::cout << "rows=" << rows() << " cols=" << cols() << std::endl;
        }
        
        Matrix<T> matmul(const Matrix<T> & b)
        {            
            assert(N == b.M);
            Matrix<T> r(rows(),b.cols());
            r.zero();
                            
            int m = rows();
            int n = b.cols();
            int k = cols();       
            
            cppmkl::cblas_gemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        m,
                        n,
                        k,
                        T(1.0),
                        this->data(),
                        k,
                        b.data(),
                        n,
                        T(0.0),
                        r.data(),
                        n);
            return r;
        }
        
        Matrix<T>& operator += (const T b)
        {
            Matrix<T> x(rows(),cols());
            x.fill(b);
            cppmkl::vadd(*this,x,*this);
            return *this;
        }
        Matrix<T>& operator -= (const T b)
        {
            Matrix<T> x(rows(),cols());
            x.fill(b);
            cppmkl::vsub(*this,x,*this);
            return *this;
        }
        Matrix<T>& operator *= (const T b)
        {
            Matrix<T> x(rows(),cols());
            x.fill(b);
            cppmkl::vmul(*this,x,*this);
            return *this;
        }
        Matrix<T>& operator /= (const T b)
        {
            Matrix<T> x(rows(),cols());
            x.fill(b);
            cppmkl::vdiv(*this,x,*this);
            return *this;
        }
        
        Matrix<T>& operator += (const Matrix<T> b)
        {
            assert(rows() == b.rows() && cols() == b.cols());            
            cppmkl::vadd(*this,b,*this);
            return *this;
        }
        Matrix<T>& operator -= (const Matrix<T> b)
        {
            assert(rows() == b.rows() && cols() == b.cols());
            cppmkl::vsub(*this,b,*this);
            return *this;
        }
        Matrix<T>& operator *= (const Matrix<T> b)
        {
            assert(rows() == b.rows() && cols() == b.cols());
            cppmkl::vmul(*this,b,*this);
            return *this;
        }
        Matrix<T>& operator /= (const Matrix<T> b)
        {
            assert(rows() == b.rows() && cols() == b.cols());            
            cppmkl::vdiv(*this,b,*this);
            return *this;
        }
        
        Matrix<T> operator + (const T b)
        {
            Matrix<T> x(rows(),cols()),y(rows(),cols());
            x.fill(b);
            cppmkl::vadd(*this,x,y);
            return y;
        }
        Matrix<T> operator - (const T b)
        {
            Matrix<T> x(rows(),cols()),y(rows(),cols());
            x.fill(b);
            cppmkl::vsub(*this,x,y);
            return y;
        }
        Matrix<T> operator * (const T b)
        {
            Matrix<T> x(rows(),cols()),y(rows(),cols());
            x.fill(b);
            cppmkl::vmul(*this,x,y);
            return y;
        }
        Matrix<T> operator / (const T b)
        {
            Matrix<T> x(rows(),cols()),y(rows(),cols());
            x.fill(b);
            cppmkl::vdiv(*this,x,y);
            return y;
        }
        
        Matrix<T> operator + (const Matrix<T> b)
        {
            Matrix<T> r(rows(),cols());
            assert(rows() == b.rows() && cols() == b.cols());            
            cppmkl::vadd(*this,b,r);
            return r;
        }
        Matrix<T> operator - (const Matrix<T> b)
        {
            Matrix<T> r(rows(),cols());
            assert(rows() == b.rows() && cols() == b.cols());            
            cppmkl::vsub(*this,b,r);
            return r;
        }
        Matrix<T> operator * (const Matrix<T> b)
        {
            return matmul(b);
        }
        Matrix<T> operator / (const Matrix<T> b)
        {
            Matrix<T> r(rows(),cols());
            assert(rows() == b.rows() && cols() == b.cols());            
            cppmkl::vdiv(*this,b,r);
            return r;
        }

        Matrix<T> hadamard(const Matrix<T> & m) {
            Matrix<T> r(rows(),cols());
            assert(rows() == m.rows() && cols() == m.cols());
            cppmkl::vmul(*this,m,r);
            return r;
        }
        
        Matrix<T> transpose() {
            Matrix<T> r(cols(),rows());
            for(size_t i = 0; i < rows(); i++)
                for(size_t j = 0; j < cols(); j++)
                    r(j,i) =(*this)(i,j);
            return r;
        }
        Matrix<T> t() {
            return transpose();
        }
        void transposeInto(Matrix<T> &m) {
            m = transpose();
        }
        void zero() {
            memset(data(),0x00,size()*sizeof(T));
        }
        void zeros() { zero(); }

        void fill(T x) {
            for(size_t i = 0; i < size(); i++) (*this)[i] = x;
        }
        void ones() {
            fill((T)1);
        }
        void minus_ones() {
            fill((T)-1.0);
        }

        T& cell(size_t i, size_t j) { return (*this)(i,j); }
        void swap_cells(size_t i1, size_t j1, size_t i2, size_t j2)
        {
            T x = (*this)(i1,j1);
            (*this)(i1,j1) = (*this)(i2,j2);
            (*this)(i2,j2) = x;
        }
            
        
        void random(T min = T(0), T max = T(1)) {
            Default r;
            for(size_t i = 0; i < size(); i++) (*this)[i] = r.random(min,max);
        }
        void identity() {
            size_t x = 0;
            zero();
            for(size_t i = 0; i < rows(); i++)
            {
                (*this)[i*N + x++] = 1;
            }
        }

        
        void print() const {
            std::cout << "Matrix[" << M << "," << N << "]=";
            for(size_t i = 0; i < rows(); i++) 
            {
                for(size_t j = 0; j < cols(); j++) std::cout << (*this)(i,j) << ",";
                std::cout << std::endl;
            }
        }

        T sum() {        
            return cppmkl::cblas_asum(size(), data(),1);                
        }
        T prod() {
            T p = (T)1.0;
            for(size_t i = 0; i < size(); i++) p *= (*this)[i];
            return p;
        }
        T mean() {
            return sum()/(T)size();
        }
        T geometric_mean() {
            T r = prod();
            return std::pow(r,(T)1.0/(T)size());
        }
        T harmonic_mean() {
            T r = 1.0/sum();
            return size()*std::pow(r,(T)-1.0);
        }
        T stddev() {
            T u = sum();
            T r = 0;
            for(size_t i = 0; i < size(); i++)
                r += std::pow((*this)[i]-u,2.0);
            return r/(T)size();
        }
           
        // eigen compatibility
        void setZero() { zero(); }
        void setOnes() { ones(); }
        void setRandom() { random(); }
        Matrix<T> matrix() { return eval(); }        
        void setIdentity() { identity(); }

        Matrix<T> cwiseMin(T min) {
            Matrix<T> r(rows(),cols());
            for(size_t i = 0; i < rows(); i++)
            for(size_t j = 0; j < cols(); j++)
                if((*this)(i,j) > min) r(i,j) = min;
                else r(i,j) = (*this)(i,j);
            return r;
        }
        Matrix<T> cwiseMax(T max) {
            Matrix<T> r(rows(),cols());
            for(size_t i = 0; i < rows(); i++)
            for(size_t j = 0; j < cols(); j++)
                if((*this)(i,j) < max) r(i,j) = max;
                else r(i,j) = (*this)(i,j);
            return r;
        }
        Matrix<T> cwiseProd(Matrix<T> & x) {
            Matrix<T> r(*this);
            r *= x;
            return r;
        }
        Matrix<T> cwiseAdd(Matrix<T> & x) {
            Matrix<T> r(*this);
            r += x;
            return r;
        }
        Matrix<T> cwiseSub(Matrix<T> & x) {
            Matrix<T> r(*this);
            r -= x;
            return r;
        }
        Matrix<T> cwiseDiv(Matrix<T> & x) {
            Matrix<T> r(*this);
            r /= x;
            return r;
        }
    };
    
    template<typename T>
    T& MatrixView<T>::operator[](size_t i) {
        return (*matrix)[row*matrix->N + i];
    }

    template<typename T>    
    T  MatrixView<T>::__getitem__(size_t i) {
        return (*matrix)[row*matrix->N + i];
    }

    template<typename T>    
    void MatrixView<T>::__setitem__(size_t i, T v)
    {
        (*matrix)[row*matrix->N + i] = v;
    }

    template<typename T> struct ComplexMatrix;
    template<typename T>
    struct ComplexMatrixView 
    {
        ComplexMatrix<T> * matrix;
        size_t row;
        ComplexMatrixView(ComplexMatrix<T> * m, size_t r) {
            matrix = m;
            row = r;
        }

        T& operator[](size_t i);
        T  __getitem__(size_t i);
        void __setitem__(size_t i, T x);
    };

    template<typename T>
    struct ComplexMatrix : public vector_base<std::complex<T>>
    {                
        using vecbase = vector_base<std::complex<T>>;
        using vecbase::size;
        using vecbase::resize;
        using vecbase::data;
        using vecbase::push_back;
        using vecbase::pop_back;
        using vecbase::front;
        using vecbase::back;
        using vecbase::at;
        using vecbase::operator [];
        using vecbase::operator =;

        size_t M;
        size_t N;

        ComplexMatrix() = default;
        ComplexMatrix(size_t i,size_t j) : vecbase(i*j),M(i),N(j) {}        
        ComplexMatrix(const ComplexMatrix<T> & v) : vecbase(v),M(v.M),N(v.N) {}

        size_t rows() const { return M; }
        size_t cols() const { return N; }

        ComplexMatrixView<T> __getitem__(size_t row) { return ComplexMatrixView<T>(this,row); }

        void resize(size_t i, size_t j)
        {
            M = i;
            N = j;
            resize(M*N);
        }
        std::complex<T>& operator()(size_t i, size_t j) {
            return (*this)[i*N + j];
        }
        void fill(const std::complex<T>& c) {
            for(size_t i = 0; i < rows(); i++) 
            for(size_t j = 0; j < cols(); j++)             
                (*this)(i,j) = c;
        }
        void zero() { fill(0.0); }
        void ones() { fill(1.0); }
        
        ComplexMatrix<T> matmul(const ComplexMatrix<T> & b)
        {
            assert(N == b.M);
            ComplexMatrix<T> r(rows(),b.cols());
            r.zero();
                            
            int m = rows();
            int n = b.cols();
            int k = cols();       
            
            cppmkl::cblas_gemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        m,
                        n,
                        k,
                        std::complex<T>(1.0,0.0),
                        this->data(),
                        k,
                        b.data(),
                        n,
                        std::complex<T>(0.0,0.0),
                        r.data(),
                        n);
            return r;
        }


        ComplexMatrix<T>& operator +=  (const ComplexMatrix<T> & v) { 
            cppmkl::vadd(*this,v,*this);
            return *this;
        }
        ComplexMatrix<T>& operator -=  (const ComplexMatrix<T> & v) { 
            cppmkl::vsub(*this,v,*this);
            return *this;
        }
        ComplexMatrix<T>& operator *=  (const ComplexMatrix<T> & v) { 
            *this = matmul(v);
            return *this;
        }
        ComplexMatrix<T>& operator /=  (const ComplexMatrix<T> & v) { 
            cppmkl::vdiv(*this,v,*this);
            return *this;        
	    }

        ComplexMatrix<T>& operator +=  (const std::complex<T> & x) { 
            ComplexMatrix<T> v(rows(),cols());
            v.fill(x);
            cppmkl::vadd(*this,v,*this);
            return *this;
        }
        ComplexMatrix<T>& operator -=  (const std::complex<T> & x) { 
            ComplexMatrix<T> v(rows(),cols());;
            v.fill(x);
            cppmkl::vsub(*this,v,*this);
            return *this;
        }
        ComplexMatrix<T>& operator *=  (const std::complex<T> & x) { 
            ComplexMatrix<T> v(rows(),cols());;
            v.fill(x);
            cppmkl::vmul(*this,v,*this);
            return *this;
        }
        ComplexMatrix<T>& operator /=  (const std::complex<T> & x) { 
            ComplexMatrix<T> v(rows(),cols());;
            v.fill(x);
            cppmkl::vdiv(*this,v,*this);
            return *this;        
	    }

        ComplexMatrix<T> operator - () {
            ComplexMatrix<T> r(*this);
            r *= std::complex<T>(-1.0,0);
            return r;
        }
        ComplexMatrix<T>& operator = (const std::complex<T>& v)
        {
            fill(v);
            return *this;
        }

        Matrix<T> real() {
            Matrix<T> r(rows(),cols());
            for(size_t i = 0; i < size(); i++) r[i] = (*this)[i].real();
            return r;
        }
        Matrix<T> imag() {
            Matrix<T> r(rows(),cols());
            for(size_t i = 0; i < size(); i++) r[i] = (*this)[i].imag();
            return r;
        }
        void real(const Matrix<T> & r) {
            for(size_t i = 0; i < size(); i++) (*this)[i].real(r[i]);
        }
        void imag(const Matrix<T> & r) {
            for(size_t i = 0; i < size(); i++) (*this)[i].imag(r[i]);
        }

        Matrix<T> abs() {
            Matrix<T> r(rows(),cols());
            for(size_t i = 0; i < size(); i++) r[i] = std::abs((*this)[i]);
            return r;
        }
        Matrix<T> arg() {
            Matrix<T> r(rows(),cols());
            for(size_t i = 0; i < size(); i++) r[i] = std::abs((*this)[i]);
            return r;
        }
        ComplexMatrix<T> conj() {
            ComplexMatrix<T> r(rows(),cols());
            for(size_t i = 0; i < size(); i++) r[i] = std::conj((*this)[i]);
            return r;
        }
        ComplexMatrix<T> proj() {
            ComplexMatrix<T> r(rows(),cols());
            for(size_t i = 0; i < size(); i++) r[i] = std::proj((*this)[i]);
            return r;
        }
        Matrix<T> norm() {
            Matrix<T> r(rows(),cols());
            for(size_t i = 0; i < size(); i++) r[i] = std::norm((*this)[i]);
            return r;
        }

        void print() {
            std::cout << "Matrix[" << size() << "]=";
            for(size_t i = 0; i < size()-1; i++) std::cout << (*this)[i] << ",";
            std::cout << (*this)[size()-1] << std::endl;
        }
    };

    
    template<typename T>
    T& ComplexMatrixView<T>::operator[](size_t i) {
        return (*matrix)[row*matrix->N + i];
    }

    template<typename T>    
    T  ComplexMatrixView<T>::__getitem__(size_t i) {
        return (*matrix)[row*matrix->N + i];
    }

    template<typename T>    
    void ComplexMatrixView<T>::__setitem__(size_t i, T v)
    {
        (*matrix)[row*matrix->N + i] = v;
    }

/////////////////////////////////////
// Vector
/////////////////////////////////////
    template<typename T>
    std::ostream& operator << (std::ostream & o, const Matrix<T> & m )
    {
        for(size_t i = 0; i < m.rows(); i++)
        {
            for(size_t j = 0; j < m.cols(); j++)
                o << m(i,j) << ",";
            o << std::endl;
        }
        return o;
    }

    template<typename T>
    Matrix<T> deinterleave(size_t ch, size_t n, T * samples)
    {
        Matrix<T> r(ch,n);
        for(size_t i = 0; i < ch; i++)        
            for(size_t j = 0; j < n; j++)
                r(i,j) = samples[j*ch + i];
        return r;
    }
    template<typename T>
    Vector<T> interleave(const Matrix<T> & m)
    {
        Vector<T> r(m.rows()*m.cols());
        int ch = m.rows();
        for(size_t i = 0; i < m.rows(); i++)
        for(size_t j = 0; j < m.cols(); j++)
            r[j*ch+i] = m(i,j);
        return r;
    }
    template<typename T>
    Vector<T> channel(size_t c, const Matrix<T> & m) {
        Vector<T> r(m.cols());
        for(size_t i = 0; i < m.cols(); i++) r[i] = m(c,i);
        return r;
    }

    template<typename T>
    Vector<T> sqr(Vector<T> & a) {
        Vector<T> r(a.size());                
        cppmkl::vsqr(a,r);
        return r;
    }
    template<typename T>
    Vector<T> abs(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vabs(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> inv(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vinv(a,r);
        return r;            
    }    
    template<typename T>
    Vector<T> sqrt(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vsqrt(a,r);
        return r;
    }
    template<typename T>
    Vector<T> rsqrt(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vinvsqrt(a,r);
        return r;
    }
    template<typename T>
    Vector<T> cbrt(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vcbrt(a,r);
        return r;
    }
    template<typename T>
    Vector<T> rcbrt(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vinvcbrt(a,r);
        return r;
    }
    
    template<typename T>
    Vector<T> pow(const Vector<T> & a,const Vector<T> & b) {
        Vector<T> r(a.size());
        cppmkl::vpow(a,b,r);
        return r;
    }
    template<typename T>
    Vector<T> pow2o3(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vpow2o3(a,r);
        return r;
    }
    template<typename T>
    Vector<T> pow3o2(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vpow3o2(a,r);
        return r;
    }
    template<typename T>
    Vector<T> pow(const Vector<T> & a,const T b) {
        Vector<T> r(a.size());
        cppmkl::vpowx(a,b,r);
        return r;
    }
    template<typename T>
    Vector<T> hypot(const Vector<T> & a,const Vector<T> & b) {
        Vector<T> r(a.size());
        cppmkl::vhypot(a,b,r);
        return r;
    }    
    template<typename T>
    Vector<T> exp(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vexp(a,r);
        return r;
    }
    template<typename T>
    Vector<T> exp2(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vexp2(a,r);
        return r;
    }
    template<typename T>
    Vector<T> exp10(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vexp10(a,r);
        return r;
    }
    template<typename T>
    Vector<T> expm1(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vexpm1(a,r);
        return r;
    }
    template<typename T>
    Vector<T> log(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vln(a,r);
        return r;        
    }    
    template<typename T>
    Vector<T> log10(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vlog10(a,r);
        return r;
    }
    template<typename T>
    Vector<T> log2(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vlog2(a,r);
        return r;
    }
    template<typename T>
    Vector<T> logb(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vlogb(a,r);
        return r;
    }
    template<typename T>
    Vector<T> log1p(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vlog1p(a,r);
        return r;
    }
    
    template<typename T>
    Vector<T> cos(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vcos(a,r);
        return r;
    }
    template<typename T>
    Vector<T> sin(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vsin(a,r);
        return r;            
    }
    template<typename T>
    Vector<T> tan(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vtan(a,r);
        return r;
    }
    template<typename T>
    Vector<T> cosh(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vcosh(a,r);
        return r;
    }
    template<typename T>
    Vector<T> sinh(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vsinh(a,r);
        return r;            
    }
    template<typename T>
    Vector<T> tanh(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vtanh(a,r);
        return r;            
    }
    template<typename T>
    Vector<T> acos(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vacos(a,r);
        return r;
    }
    template<typename T>
    Vector<T> asin(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vasin(a,r);
        return r;            
    }
    template<typename T>
    Vector<T> atan(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vatan(a,r);
        return r;
    }
    template<typename T>
    Vector<T> atan2(const Vector<T> & a,const Vector<T> &n) {
        Vector<T> r(a.size());
        cppmkl::vatan2(a,n,r);
        return r;
    }
    template<typename T>
    Vector<T> acosh(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vacosh(a,r);
        return r;
    }
    template<typename T>
    Vector<T> asinh(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vasinh(a,r);
        return r;            
    }
    template<typename T>
    Vector<T> atanh(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vatanh(a,r);
        return r;
    }        
    template<typename T>
    void sincos(const Vector<T> & a, Vector<T> & b, Vector<T> & r) {        
        cppmkl::vsincos(a,b,r);        
    }
    template<typename T>
    Vector<T> erf(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::verf(a,r);
        return r;
    }
    template<typename T>
    Vector<T> erfinv(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::verfinv(a,r);
        return r;
    }
    template<typename T>
    Vector<T> erfc(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::verfc(a,r);
        return r;
    }
    template<typename T>
    Vector<T> cdfnorm(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vcdfnorm(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> cdfnorminv(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vcdfnorminv(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> floor(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vfloor(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> ceil(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vceil(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> trunc(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vtrunc(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> round(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vround(a,r);
        return r;        
    }    
    template<typename T>
    Vector<T> nearbyint(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vnearbyint(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> rint(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vrint(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> fmod(const Vector<T> & a, Vector<T> & b) {
        Vector<T> r(a.size());
        cppmkl::vmodf(a,b,r);
        return r;
    }    
    
    template<typename T>
    Vector<T> CIS(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vCIS(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> cospi(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vcospi(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> sinpi(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vsinpi(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> tanpi(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vtanpi(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> acospi(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vacospi(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> asinpi(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vasinpi(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> atanpi(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vatanpi(a,r);
        return r;
    }    
    template<typename T>
    Vector<T> atan2pi(const Vector<T> & a, Vector<T> & b) {
        Vector<T> r(a.size());
        cppmkl::vatan2pi(a,b,r);
        return r;
    }    
    template<typename T>
    Vector<T> cosd(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vcosd(a.size(),a.data(),r.data());
        return r;
    }    
    template<typename T>
    Vector<T> sind(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vsind(a.size(),a.data(),r.data());
        return r;
    }    
    template<typename T>
    Vector<T> tand(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vtand(a.size(),a.data(),r.data());
        return r;
    }       
    template<typename T>
    Vector<T> lgamma(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vlgamma(a,r);
        return r;
    }       
    template<typename T>
    Vector<T> tgamma(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vtgamma(a,r);
        return r;
    }       
    template<typename T>
    Vector<T> expint1(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vexpint1(a,r);
        return r;
    }       

/////////////////////////////////////
// ComplexVector
/////////////////////////////////////
    template<typename T>
    Vector<T> abs(const ComplexVector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::vabs(a,r);
        return r;
    }
    template<typename T>
    ComplexVector<T> sqrt(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vsqrt(a,r);
        return r;
    }
    template<typename T>
    ComplexVector<T> pow(const ComplexVector<T> & a,const ComplexVector<T> & b) {
        ComplexVector<T> r(a.size());
        cppmkl::vpow(a,b,r);
        return r;
    }
    template<typename T>
    ComplexVector<T> exp(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vexp(a,r);
        return r;
    }    
    template<typename T>
    ComplexVector<T> log(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vln(a,r);
        return r;        
    }
    template<typename T>
    ComplexVector<T> cos(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vcos(a,r);
        return r;
    }
    template<typename T>
    ComplexVector<T> sin(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vsin(a,r);
        return r;            
    }
    template<typename T>
    ComplexVector<T> tan(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vtan(a,r);
        return r;
    }
    template<typename T>
    ComplexVector<T> cosh(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vcosh(a,r);
        return r;
    }
    template<typename T>
    ComplexVector<T> sinh(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vsinh(a,r);
        return r;            
    }
    template<typename T>
    ComplexVector<T> tanh(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vtanh(a,r);
        return r;            
    }
    template<typename T>
    ComplexVector<T> acos(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vacos(a,r);
        return r;
    }
    template<typename T>
    ComplexVector<T> asin(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vasin(a,r);
        return r;            
    }
    template<typename T>
    ComplexVector<T> atan(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vatan(a,r);
        return r;
    }
    template<typename T>
    ComplexVector<T> acosh(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vacosh(a,r);
        return r;
    }
    template<typename T>
    ComplexVector<T> asinh(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vasinh(a,r);
        return r;            
    }
    template<typename T>
    ComplexVector<T> atanh(const ComplexVector<T> & a) {
        ComplexVector<T> r(a.size());
        cppmkl::vatanh(a,r);
        return r;
    }        
    

/////////////////////////////////////
// Matrix
/////////////////////////////////////    
    template<typename T>
    Matrix<T> sqr(Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vsqr(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> abs(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vabs(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> inv(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vinv(a,r);
        return r;            
    }
    template<typename T>
    Matrix<T> sqrt(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vsqrt(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> rsqrt(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vinvsqrt(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> cbrt(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vcbrt(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> rcbrt(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vinvcbrt(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> pow(const Matrix<T> & a,const Matrix<T> & b) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vpow(a,b,r);
        return r;
    }
    template<typename T>
    Matrix<T> pow2o3(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vpow2o3(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> pow3o2(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vpow3o2(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> pow(const Matrix<T> & a,const T b) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vpowx(a,b,r);
        return r;
    }
    template<typename T>
    Matrix<T> hypot(const Matrix<T> & a,const Matrix<T> & b) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vhypot(a,b,r);
        return r;
    }
    template<typename T>
    Matrix<T> exp(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vexp(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> exp2(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vexp2(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> exp10(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vexp10(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> expm1(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vexpm1(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> ln(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vln(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> log(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vln(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> log10(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vlog10(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> log2(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vlog2(a,r);        
        return r;
    }
    template<typename T>
    Matrix<T> logb(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vlogb(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> log1p(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vlog1p(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> cos(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vcos(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> sin(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vsin(a,r);
        return r;            
    }
    template<typename T>
    Matrix<T> tan(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vtan(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> cosh(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vcosh(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> sinh(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vsinh(a,r);
        return r;            
    }
    template<typename T>
    Matrix<T> tanh(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vtanh(a,r);
        return r;            
    }
    template<typename T>
    Matrix<T> acos(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vacos(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> asin(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vasin(a,r);
        return r;            
    }
    template<typename T>
    Matrix<T> atan(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vatan(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> atan2(const Matrix<T> & a,const Matrix<T> &n) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vatan2(a,n,r);
        return r;
    }
    template<typename T>
    Matrix<T> acosh(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vacosh(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> asinh(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vasinh(a,r);
        return r;            
    }
    template<typename T>
    Matrix<T> atanh(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vatanh(a,r);
        return r;
    }        
    template<typename T>
    void sincos(const Matrix<T> & a, Matrix<T> & b, Matrix<T> & r) {        
        cppmkl::vsincos(a,b,r);
    }
    template<typename T>
    Matrix<T> erf(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::verf(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> erfinv(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::verfinv(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> erfc(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::verfc(a,r);
        return r;
    }
    template<typename T>
    Matrix<T> cdfnorm(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vcdfnorm(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> cdfnorminv(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vcdfnorminv(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> floor(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vfloor(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> ceil(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vceil(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> trunc(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vtrunc(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> round(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vround(a,r);
        return r;        
    }    
    template<typename T>
    Matrix<T> nearbyint(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vnearbyint(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> rint(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vrint(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> fmod(const Matrix<T> & a, Matrix<T> & b) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vmodf(a,b,r);
        return r;
    }    
    template<typename T>
    Matrix<T> mulbyconj(const Matrix<T> & a, const Matrix<T> & b) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vmulbyconj(a,b,r);
        return r;
    }    
    template<typename T>
    Matrix<T> conj(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vconj(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> arg(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::varg(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> CIS(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vCIS(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> cospi(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vcospi(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> sinpi(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vsinpi(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> tanpi(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vtanpi(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> acospi(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vacospi(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> asinpi(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vasinpi(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> atanpi(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vatanpi(a,r);
        return r;
    }    
    template<typename T>
    Matrix<T> atan2pi(const Matrix<T> & a, Matrix<T> & b) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vatan2pi(a,b,r);
        return r;
    }    
    template<typename T>
    Matrix<T> cosd(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vcosd(a.size(),a.data(),r.data());
        return r;
    }    
    template<typename T>
    Matrix<T> sind(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vsind(a.size(),a.data(),r.data());
        return r;
    }    
    template<typename T>
    Matrix<T> tand(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vtand(a.size(),a.data(),r.data());
        return r;
    }       
    template<typename T>
    Matrix<T> lgamma(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vlgamma(a,r);
        return r;
    }       
    template<typename T>
    Matrix<T> tgamma(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vtgamma(a,r);
        return r;
    }       
    template<typename T>
    Matrix<T> expint1(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vexpint1(a,r);
        return r;
    }       

    template<typename T>
    Vector<T> copy(const Vector<T> & a) {
        Vector<T> r(a.size());
        cppmkl::cblas_copy(a.size(),a.data(),1,r.data(),1);
        return r;
    }       

    template<typename T> T sum(const Vector<T> & a) {        
        return cppmkl::cblas_asum(a.size(), a.data(),1);        
    }       

    template<typename T>
    Vector<T> add(const Vector<T> & a, const Vector<T> & b) {
        Vector<T> r(b);
        cppmkl::cblas_axpy(a.size(),1.0,a.data(),1,r.data(),1);
        return r;
    }       
    template<typename T>
    Vector<T> sub(const Vector<T> & a, const Vector<T> & b) {
        Vector<T> r(a);
        cppmkl::cblas_axpy(a.size(),-1.0,b.data(),1,r.data(),1);
        return r;
    }       
    template<typename T>
    T dot(const Vector<T> & a, Vector<T> & b) {
        return cppmkl::cblas_dot(a,b);
    }       
    template<typename T>
    T nrm2(const Vector<T> & a) {
        Vector<T> r(a);
        return cppmkl::cblas_nrm2(a);        
    }       
    
    template<typename T>
    void scale(Vector<T> & x, T alpha) {
        cppmkl::cblas_scal(x.size(),alpha,x.data(),1);
    }


    template<typename T>
    size_t min_index(const Matrix<T> & m) { return cppmkl::cblas_iamin(m.size(),m.data(),1); }
    template<typename T>
    size_t max_index(const Matrix<T> & m) { return cppmkl::cblas_iamax(m.size(),m.data(),1); }
    
    template<typename T>
    size_t min_index(const Vector<T> & v) { return cppmkl::cblas_iamin(v.size(),v.data(),1); }
    template<typename T>
    size_t max_index(const Vector<T> & v) { return cppmkl::cblas_iamax(v.size(),v.data(),1); }


/////////////////////////////////////
// Vector
/////////////////////////////////////
    template<typename T>
    Vector<T> linspace(size_t n, T start, T inc=1) {
        Vector<T> r(n);
        for(size_t i = 0; i < n; i++) {
            r[i] = start + i*inc;
        }
        return r;
    }
    template<typename T>
    Vector<T> linspace(T start, T end, T inc=1) {
        size_t n = (end - start)/inc;
        Vector<T> r(n);
        for(size_t i = 0; i < n; i++) {
            r[i] = start + i*inc;
        }
        return r;
    }

    template<typename T>
    Vector<T> operator * (T a, const Vector<T> & b) {
        Vector<T> x(b.size());
        x.fill(a);
        Vector<T> r(b.size());
        cppmkl::vmul(x,b,r);            
        return r;
    }
    template<typename T>
    Vector<T> operator / (T a, const Vector<T> & b) {
        Vector<T> x(b.size());
        x.fill(a);
        Vector<T> r(b.size());
        cppmkl::vdiv(x,b,r);            
        return r;
    }
    template<typename T>
    Vector<T> operator + (T a, const Vector<T> & b) {
        Vector<T> x(b.size());
        x.fill(a);
        Vector<T> r(b.size());
        cppmkl::vadd(x,b,r);            
        return r;
    }
    template<typename T>
    Vector<T> operator - (T a, const Vector<T> & b) {
        Vector<T> x(b.size());
        x.fill(a);
        Vector<T> r(b.size());
        cppmkl::vsub(x,b,r);            
        return r;
    }

    template<typename T>
    Vector<T> operator * (const Vector<T> & a, T b) {
        Vector<T> x(a.size());
        x.fill(b);
        Vector<T> r(a.size());
        cppmkl::vmul(a,x,r);            
        return r;
    }
    template<typename T>
    Vector<T> operator / (const Vector<T> & a , T b) {
        Vector<T> x(a.size());
        x.fill(b);
        Vector<T> r(a.size());
        cppmkl::vdiv(a,x,r);            
        return r;
    }
    template<typename T>
    Vector<T> operator + (const Vector<T> & a, T b) {
        Vector<T> x(a.size());
        x.fill(b);
        Vector<T> r(a.size());
        cppmkl::vadd(a,x,r);            
        return r;
    }
    template<typename T>
    Vector<T> operator - (const Vector<T> & a, T b) {
        Vector<T> x(a.size());
        x.fill(b);
        Vector<T> r(a.size());
        cppmkl::vsub(a,x,r);            
        return r;
    }
    
    template<typename T>
    Vector<T> operator - (const Vector<T> & a, const Vector<T> & b) {        
        Vector<T> r(a.size());
        cppmkl::vsub(a,b,r);            
        return r;
    }
    template<typename T>
    Vector<T> operator + (const Vector<T> & a, const Vector<T> & b) {        
        Vector<T> r(a.size());
        cppmkl::vadd(a,b,r);                    
        return r;
    }
    template<typename T>
    Vector<T> operator * (const Vector<T> & a, const Vector<T> & b) {        
        Vector<T> r(a.size());
        cppmkl::vmul(a,b,r);            
        return r;
    }
    template<typename T>
    Vector<T> operator / (const Vector<T> & a, const Vector<T> & b) {        
        Vector<T> r(a.size());
        cppmkl::vdiv(a,b,r);            
        return r;
    }

/////////////////////////////////////
// ComplexVector
/////////////////////////////////////
    template<typename T>
    ComplexVector<T> operator * (std::complex<T> a, const ComplexVector<T> & b) {
        ComplexVector<T> x(b.size());
        x.fill(a);
        ComplexVector<T> r(b.size());
        cppmkl::vmul(x,b,r);            
        return r;
    }
    template<typename T>
    ComplexVector<T> operator / (std::complex<T> a, const ComplexVector<T> & b) {
        ComplexVector<T> x(b.size());
        x.fill(a);
        ComplexVector<T> r(b.size());
        cppmkl::vdiv(x,b,r);            
        return r;
    }
    template<typename T>
    ComplexVector<T> operator + (std::complex<T> a, const ComplexVector<T> & b) {
        ComplexVector<T> x(b.size());
        x.fill(a);
        ComplexVector<T> r(b.size());
        cppmkl::vadd(x,b,r);            
        return r;
    }
    template<typename T>
    ComplexVector<T> operator - (std::complex<T> a, const ComplexVector<T> & b) {
        ComplexVector<T> x(b.size());
        x.fill(a);
        ComplexVector<T> r(b.size());
        cppmkl::vsub(x,b,r);            
        return r;
    }

    template<typename T>
    ComplexVector<T> operator * (const ComplexVector<T> & a, std::complex<T> b) {
        ComplexVector<T> x(a.size());
        x.fill(b);
        ComplexVector<T> r(a.size());
        cppmkl::vmul(a,x,r);            
        return r;
    }
    template<typename T>
    ComplexVector<T> operator / (const ComplexVector<T> & a , std::complex<T> b) {
        ComplexVector<T> x(a.size());
        x.fill(b);
        ComplexVector<T> r(a.size());
        cppmkl::vdiv(a,x,r);            
        return r;
    }
    template<typename T>
    ComplexVector<T> operator + (const ComplexVector<T> & a, std::complex<T> b) {
        ComplexVector<T> x(a.size());
        x.fill(b);
        ComplexVector<T> r(a.size());
        cppmkl::vadd(a,x,r);            
        return r;
    }
    template<typename T>
    ComplexVector<T> operator - (const ComplexVector<T> & a, std::complex<T> b) {
        ComplexVector<T> x(a.size());
        x.fill(b);
        ComplexVector<T> r(a.size());
        cppmkl::vsub(a,x,r);            
        return r;
    }
    
    template<typename T>
    ComplexVector<T> operator - (const ComplexVector<T> & a, const ComplexVector<T> & b) {        
        ComplexVector<T> r(a.size());
        cppmkl::vsub(a,b,r);            
        return r;
    }
    template<typename T>
    ComplexVector<T> operator + (const ComplexVector<T> & a, const ComplexVector<T> & b) {        
        ComplexVector<T> r(a.size());
        cppmkl::vadd(a,b,r);                    
        return r;
    }
    template<typename T>
    ComplexVector<T> operator * (const ComplexVector<T> & a, const ComplexVector<T> & b) {        
        ComplexVector<T> r(a.size());
        cppmkl::vmul(a,b,r);            
        return r;
    }
    template<typename T>
    ComplexVector<T> operator / (const ComplexVector<T> & a, const ComplexVector<T> & b) {        
        ComplexVector<T> r(a.size());
        cppmkl::vdiv(a,b,r);            
        return r;
    }

/////////////////////////////////////
// Matrix
/////////////////////////////////////
    template<typename T>
    Matrix<T> operator * (const Matrix<T> & a, const Matrix<T> & b) {
        assert(a.N == b.M);
        Matrix<T> r(a.rows(),b.cols());
        r.zero();
                        
        int m = a.rows();
        int n = b.cols();
        int k = a.cols();       
        
        cppmkl::cblas_gemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m,
                    n,
                    k,
                    T(1.0),
                    a.data(),
                    k,
                    b.data(),
                    n,
                    T(0.0),
                    r.data(),
                    n);       
        return r;
    } 
    
    template<typename T>
    Matrix<T> operator + (const Matrix<T> & a, const Matrix<T> & b) {
        assert(a.rows() == b.rows() && a.cols() == b.cols());
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vadd(a,b,r);            
        return r;
    }
    template<typename T>
    Matrix<T> operator / (const Matrix<T> & a, const Matrix<T> & b) {
        assert(a.rows() == b.rows() && a.cols() == b.cols());
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vdiv(a,b,r);            
        return r;
    }        
    template<typename T>
    Matrix<T> operator - (const Matrix<T> & a, const Matrix<T> & b) {
        assert(a.rows() == b.rows() && a.cols() == b.cols());
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::vsub(a,b,r);            
        return r;
    }        

    template<typename T>
    Matrix<T> operator * (T a, const Matrix<T> & b) {
        Matrix<T> x(b.M,b.N);
        x.fill(a);
        Matrix<T> r(b.M,b.N);
        cppmkl::vmul(x,b,r);
        return r;
    }
    
    template<typename T>
    Matrix<T> operator + (T a, const Matrix<T> & b) {
        Matrix<T> x(b.M,b.N);
        x.fill(a);
        Matrix<T> r(b.M,b.N);
        cppmkl::vadd(x,b,r);
        return r;
    }
    template<typename T>
    Matrix<T> operator - (T a, const Matrix<T> & b) {
        Matrix<T> x(b.M,b.N);
        x.fill(a);
        Matrix<T> r(b.M,b.N);
        cppmkl::vsub(x,b,r);
        return r;
    }
    template<typename T>
    Matrix<T> operator / (T a, const Matrix<T> & b) {
        Matrix<T> x(b.M,b.N);
        x.fill(a);
        Matrix<T> r(b.M,b.N);
        cppmkl::vdiv(x,b,r);
        return r;
    }
    template<typename T>
    Matrix<T> operator * (const Matrix<T> & a, T b) {
        Matrix<T> x(a.M,a.N);
        x.fill(b);
        Matrix<T> r(a.M,a.N);
        cppmkl::vmul(a,x,r);
        return r;
    }
    
    template<typename T>
    Matrix<T> operator + (const Matrix<T> & a, T b) {
        Matrix<T> x(a.M,a.N);
        x.fill(b);
        Matrix<T> r(a.M,a.N);
        cppmkl::vadd(a,x,r);
        return r;
    }
    template<typename T>
    Matrix<T> operator - (const Matrix<T> & a, T b) {
        Matrix<T> x(a.M,a.N);
        x.fill(b);
        Matrix<T> r(a.M,a.N);
        cppmkl::vsub(a,x,r);
        return r;
    }
    template<typename T>
    Matrix<T> operator / (const Matrix<T> & a, T b) {
        Matrix<T> x(a.M,a.N);
        x.fill(b);
        Matrix<T> r(a.M,a.N);
        cppmkl::vdiv(a,x,r);
        return r;
    }    
    template<typename T>
    Matrix<T> hadamard(const Matrix<T> & a, const Matrix<T> & b) {
        Matrix<T> r(a.rows(),a.cols());
        assert(a.rows() == b.rows() && a.cols() == b.cols());
        cppmkl::vmul(a,b,r);
        return r;
    }      
    template<typename T>
    Vector<T> operator *(const Vector<T> &a, const Matrix<T> &b) {
        Vector<T> r(a.size());        
        sgemv(CblasRowMajor,CblasNoTrans,b.rows(),b.cols(),1.0,b.data(),b.cols(),a.data(),1,1.0,r.data(),1);
        return r;
    }      
    template<typename T>
    Vector<T> operator *(const Matrix<T> &a, const Vector<T> &b) {
        Vector<T> r(b.size());        
        sgemv(CblasRowMajor,CblasNoTrans,a.rows(),a.cols(),1.0,a.data(),a.cols(),b.data(),1,1.0,r.data(),1);
        return r;
    }      

    template<typename T>
    Matrix<T> copy(const Matrix<T> & a) {
        Matrix<T> r(a.rows(),a.cols());
        cppmkl::cblas_copy(a.size(),a.data(),1,r.data(),1);
        return r;
    }       

    template<typename T> T sum(const Matrix<T> & a) {        
        return cppmkl::cblas_asum(a.size(), a.data(),1);                
    }       

    template<typename T>
    Matrix<T> add(const Matrix<T> & a, const Matrix<T> & b) {
        Matrix<T> r(b);
        cppmkl::cblas_axpy(a.size(),1.0,a.data(),1,r.data(),1);
        return r;
    }       
    template<typename T>
    Matrix<T> sub(const Matrix<T> & a, const Matrix<T> & b) {
        Matrix<T> r(a);
        cppmkl::cblas_axpy(a.size(),-1.0,b.data(),1,r.data(),1);
        return r;
    }       
    template<typename T>
    T dot(const Matrix<T> & a, Matrix<T> & b) {
        return cppmkl::cblas_dot(a,b);
    }       
    template<typename T>
    T nrm2(const Matrix<T> & a) {
        Matrix<T> r(a);
        return cppmkl::cblas_nrm2(a);        
    }       
    
    template<typename T>
    void scale(Matrix<T> & x, T alpha) {
        cppmkl::cblas_scal(x.size(),alpha,x.data(),1);
    }

    
/////////////////////////////////////
// FFT
/////////////////////////////////////
    template<typename T>
    struct RealFFT1D
    {
        DFTI_DESCRIPTOR_HANDLE handle1;        
        size_t size;
        
        RealFFT1D(size_t size) {
            DFTI_CONFIG_VALUE prec;
            if(typeid(T) == typeid(float)) prec = DFTI_SINGLE;
            else prec = DFTI_DOUBLE;
            DftiCreateDescriptor(&handle1, prec, DFTI_REAL,  1, size );
            DftiSetValue(handle1, DFTI_PLACEMENT, DFTI_NOT_INPLACE); //Out of place FFT
            DftiSetValue(handle1, DFTI_BACKWARD_SCALE, 1.0f / size);
            DftiCommitDescriptor(handle1);            
            this->size = size;
        }
        ~RealFFT1D() {
            DftiFreeDescriptor(&handle1);            
        }

        void Forward( Vector<T> & input, Vector<std::complex<T>> & output) {
            output.resize(size);
            Vector<float> x(size*2);            
            DftiComputeForward(handle1, input.data(),x.data());
            memcpy(output.data(),x.data(), x.size()*sizeof(float));            
        }
        void Backward( Vector<std::complex<T>> & input, Vector<T> & output) {
            output.resize(size);
            Vector<float> x(size*2);            
            memcpy(x.data(),input.data(),x.size()*sizeof(float));
            DftiComputeBackward(handle1, x.data(), output.data());
        }                
    };

    template<typename T = float>
    struct ComplexFFT1D
    {
        DFTI_DESCRIPTOR_HANDLE handle1;        
        size_t size;
        
        ComplexFFT1D(size_t size) {
            DFTI_CONFIG_VALUE prec;
            if(typeid(T) == typeid(float)) prec = DFTI_SINGLE;
            else prec = DFTI_DOUBLE;
            DftiCreateDescriptor(&handle1, prec, DFTI_COMPLEX, 1, size );
            DftiSetValue(handle1, DFTI_PLACEMENT, DFTI_NOT_INPLACE); //Out of place FFT
            DftiCommitDescriptor(handle1);            
            this->size = size;
        }
        ~ComplexFFT1D() {
            DftiFreeDescriptor(&handle1);            
        }

        void Forward( Vector<std::complex<T>> & input, Vector<std::complex<T>> & output) {
            output.resize(size);
            DftiComputeForward(handle1, input.data(),output.data());
        }
        void Backward( Vector<std::complex<T>> & input, Vector<std::complex<T>> & output) {
            output.resize(size);
            DftiComputeBackward(handle1, input.data(), output.data());
        }        
    };

    template<typename T>
    Matrix<T> matmul(const Matrix<T> & a, const Matrix<T> & b)
    {            
        assert(a.N == b.M);
        Matrix<T> r(a.rows(),b.cols());
        r.zero();
                        
        int m = a.rows();
        int n = b.cols();
        int k = a.cols();       
        
        cppmkl::cblas_gemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m,
                    n,
                    k,
                    T(1.0),
                    a.data(),
                    k,
                    b.data(),
                    n,
                    T(0.0),
                    r.data(),
                    n);
        return r;
    }
}