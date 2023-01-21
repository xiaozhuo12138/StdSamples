// viper_matx
#pragma once

#include "matx.h"
#include <cassert>
#include <cstdio>
#include <cuda/std/ccomplex>

template<typename T> using Complex = cuda::std::complex;

namespace Viper
{
    template<typename T>
    struct C2C
    {
        tensor_t tensor,output;
        size_t batches;
        size_t blocks;
        C2C(size_t batch,size_t block) : blocks(block),batches(batch)
        {
            tensor = matx::make_tensor<Complex<T>>(batches,blocks);
            output = matx::make_tensor<Complex<T>>(batches,blocks);
        }
        void setInput(size_t batch, Complex<T> * data)
        {
            for(size_t i = 0; i < blocks; i++)
            {
                tensor(batch,i) = data[i];
            }
        }
        void getOutput(size_t batch, Complex<T> * data)
        {
            for(size_t i = 0; i < blocks; i++)
            {
                data[i] = output(batch,i);
            }
        }
        void normalize() {
            for(size_t i = 0; i < batches; i++)
                for(size_t j = 0; j < blocks; j++)
                    tensor(i,j) /= (T)blocks;
        }
        void forward()
        {
            tensor.PrefetchData(0);
            matx::fft(tensor,output);
        }
        void inverse()
        {
            matx::ifft(tensor,output);
        }
    };
    template<typename T>
    struct R2C
    {
        tensor_t tensor,output;
        size_t batches;
        size_t blocks;
        R2C(size_t batch,size_t block) : blocks(block),batches(batch)
        {
            tensor = matx::make_tensor<T>(batches,blocks);
            output = matx::make_tensor<Complex<T>>(batches,blocks);
        }
        void setInput(size_t batch, T * data)
        {
            for(size_t i = 0; i < blocks; i++)
            {
                tensor(batch,i) = data[i];
            }
        }
        void getOutput(size_t batch, Complex<T> * data)
        {
            for(size_t i = 0; i < blocks; i++)
            {
                data[i] = output(batch,i);
            }
        }
        void normalize() {
            for(size_t i = 0; i < batches; i++)
                for(size_t j = 0; j < blocks; j++)
                    tensor(i,j) /= (T)blocks;
        }
        void forward()
        {
            tensor.PrefetchData(0);
            matx::fft(tensor,output);
        }        
    };
    template<typename T>
    struct C2R
    {
        tensor_t tensor,output;
        size_t batches;
        size_t blocks;
        C2R(size_t batch,size_t block) : blocks(block),batches(batch)
        {
            tensor = matx::make_tensor<Complex<T>>(batches,blocks);
            output = matx::make_tensor<T>(batches,blocks);
        }
        void setInput(size_t batch, Complex<T> * data)
        {
            for(size_t i = 0; i < blocks; i++)
            {
                tensor(batch,i) = data[i];
            }
        }
        void getOutput(size_t batch, T * data)
        {
            for(size_t i = 0; i < blocks; i++)
            {
                data[i] = output(batch,i);
            }
        }
        void normalize() {
            for(size_t i = 0; i < batches; i++)
                for(size_t j = 0; j < blocks; j++)
                    tensor(i,j) /= (T)blocks;
        }
        void forward()
        {
            tensor.PrefetchData(0);
            matx::fft(tensor,output);
        }
        void inverse()
        {
            matx::ifft(tensor,output);
        }
    };