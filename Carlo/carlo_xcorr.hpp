#pragma once

namespace Casino::IPP
{
    template<typename T>
    void CrossCorrNormGetBufferSize(int src1Len, int src2Len, int dstLen, int lowLag, IppEnum algType, int * pBufferSize)
    {
        IppDataType dataType = GetDataType<T>();        
        ippsCrossCorrNormGetBufferSize(src1Len,src2Len,dstLen,lowLag,dataType,algType,pBufferSize);    
    }

    template<typename T>
    void CrossCorrNorm(const T * pSrc1, int src1Len, const T * pSrc2, int src2Len, T * pDst, int dstLen, int lowLag, IppEnum algType, Ipp8u* pBuffer)
    {   
        throw std::runtime_error("Called abstract CrossCorrNorm");
    }
    
    template<>
    void CrossCorrNorm<Ipp32f>(const Ipp32f * pSrc1, int src1Len, const Ipp32f * pSrc2, int src2Len, Ipp32f * pDst, int dstLen, int lowLag, IppEnum algType, Ipp8u* pBuffer)
    {   
        IppStatus status;
        status = ippsCrossCorrNorm_32f(pSrc1,src1Len,pSrc2,src2Len,pDst,dstLen,lowLag,algType,pBuffer);
        checkStatus(status);
    }
    template<>
    void CrossCorrNorm<Ipp64f>(const Ipp64f * pSrc1, int src1Len, const Ipp64f * pSrc2, int src2Len, Ipp64f * pDst, int dstLen, int lowLag, IppEnum algType, Ipp8u* pBuffer)
    {   
        IppStatus status;
        status = ippsCrossCorrNorm_64f(pSrc1,src1Len,pSrc2,src2Len,pDst,dstLen,lowLag,algType,pBuffer);
        checkStatus(status);
    }
    template<>
    void CrossCorrNorm<Ipp32fc>(const Ipp32fc * pSrc1, int src1Len, const Ipp32fc * pSrc2, int src2Len, Ipp32fc * pDst, int dstLen, int lowLag, IppEnum algType, Ipp8u* pBuffer)
    {   
        IppStatus status;
        status = ippsCrossCorrNorm_32fc(pSrc1,src1Len,pSrc2,src2Len,pDst,dstLen,lowLag,algType,pBuffer);
        checkStatus(status);
    }
    template<>
    void CrossCorrNorm<Ipp64fc>(const Ipp64fc * pSrc1, int src1Len, const Ipp64fc * pSrc2, int src2Len, Ipp64fc * pDst, int dstLen, int lowLag, IppEnum algType, Ipp8u* pBuffer)
    {   
        IppStatus status;
        status = ippsCrossCorrNorm_64fc(pSrc1,src1Len,pSrc2,src2Len,pDst,dstLen,lowLag,algType,pBuffer);
        checkStatus(status);
    }
    
    template<typename T>
    struct CrossCorr
    {
        Ipp8u * buffer;
        int bufferLen;
        int      type;
        int src1Len,src2Len,dstLen,lowLag;

        CrossCorr(size_t src1Len, size_t src2Len, size_t dstLen, int lowLag, int algorithm = (int)Algorithm::ALG_FFT) {
            CrossCorrNormGetBufferSize<T>(src1Len,src2Len,dstLen,lowLag,algorithm,&bufferLen);        
            if(bufferLen > 0) buffer = Malloc<Ipp8u>(bufferLen);
            assert(buffer != NULL);
            type = algorithm;
            this->lowLag = lowLag;
            this->src1Len = src1Len;
            this->src2Len = src2Len;
            this->dstLen = dstLen;
        }
        ~CrossCorr() {
            if(buffer) Free(buffer);
        }
        void Process(T * src1, T * src2, T * dst) {
            CrossCorrNorm<T>(src1,src1Len,src2,src2Len,dst,dstLen,lowLag,type,buffer);            
        }
    };

    void xcorr(size_t srcLen, Ipp32f * src1, size_t srcLen2, Ipp32f* src2, size_t dstLen, Ipp32f * dst, int lowLag,int algorithm = (int)Algorithm::ALG_FFT)
    {
        CrossCorr<float> c(srcLen,srcLen2,dstLen,lowLag,algorithm);
        c.Process(src1,src2,dst);
    }        
    void xcorr(CrossCorr<float> &c, Ipp32f * src1, Ipp32f* src2, size_t dstLen, Ipp32f * dst)
    {        
        c.Process(src1,src2,dst);
    }        
    void xcorr(size_t srcLen, Ipp32fc * src1, size_t srcLen2, Ipp32fc* src2, size_t dstLen, Ipp32fc * dst, int lowLag,int algorithm = (int)Algorithm::ALG_FFT)
    {
        CrossCorr<Ipp32fc> c(srcLen,srcLen2,dstLen,lowLag,algorithm);
        c.Process(src1,src2,dst);
    }            
    void xcorr(CrossCorr<Ipp32fc> &c, Ipp32fc * src1, Ipp32fc* src2, Ipp32fc * dst)
    {    
        c.Process(src1,src2,dst);
    }            
    void xcorr(size_t srcLen, Ipp64f * src1, size_t srcLen2, Ipp64f* src2, size_t dstLen, Ipp64f * dst, int lowLag,int algorithm = (int)Algorithm::ALG_FFT)
    {
        CrossCorr<Ipp64f> c(srcLen,srcLen2,dstLen,lowLag,algorithm);
        c.Process(src1,src2,dst);
    }        
    void xcorr(CrossCorr<Ipp64f> &c, Ipp64f * src1, Ipp64f* src2, Ipp64f * dst)
    {        
        c.Process(src1,src2,dst);
    }        
    void xcorr(size_t srcLen, Ipp64fc * src1, size_t srcLen2, Ipp64fc* src2, size_t dstLen, Ipp64fc * dst, int lowLag,int algorithm = (int)Algorithm::ALG_FFT)
    {
        CrossCorr<Ipp64fc> c(srcLen,srcLen2,dstLen,lowLag,algorithm);
        c.Process(src1,src2,dst);
    }        
    void xcorr(CrossCorr<Ipp64fc> &c, Ipp64fc * src1, Ipp64fc* src2, Ipp64fc * dst)
    {    
        c.Process(src1,src2,dst);
    }        


}