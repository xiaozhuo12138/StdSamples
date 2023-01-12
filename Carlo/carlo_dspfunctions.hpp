#pragma once

#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <memory>
#include <cmath>
#include "ipp.h"

using namespace std;
struct xyz
{
    double x;
    double y;
    double z;
};

struct lla
{
    double lat;
    double lon;
    double alt;
};


/**
 *	Index sort
 *
 *	Returns index of data after data is sorted.
 *	Used in Phase Gradient Algorithm.
 *
 *  Inputs:
 *		v - vector of values to be sorted
 *
 *  Outputs:
 *      Returns vector of indices (std::size_t type) of original data elements, sorted.
 *
 */
template<typename T>
std::vector<std::size_t> sort_indexes(const std::vector<T> &v, bool ascending)
{

    // initialize original index locations
    std::vector<std::size_t> idx(v.size());
    for (std::size_t i = 0; i != idx.size(); ++i)
        idx[i] = i;

    if (ascending)
    {
        sort(idx.begin(), idx.end(), [&v](std::size_t i1, std::size_t i2)
        { return v[i1] < v[i2]; });
    }
    else
    {
        sort(idx.begin(), idx.end(), [&v](std::size_t i1, std::size_t i2)
        { return v[i1] > v[i2]; }); //-- descending order
    }

    return idx;
}

/**
 *	subVector
 *
 *	Returns subset of input vector given by firstIdx and lastIdx.
 *
 *  Inputs:
 *		vecIn - Input vector
 *		firstIdx - location of beginning of subset vector
 *		lastIdx - location of end of subset vector
 *
 *  Outputs:
 *      Returns a subset of the input vector.
 *
 */
template<typename T>
std::vector<T> subVector(const std::vector<T> &vecIn, int firstIdx, int lastIdx)
{
    typename std::vector<T>::const_iterator first = vecIn.begin() + firstIdx; // matlab: rgBin0(1:NrgBin1);
    typename std::vector<T>::const_iterator last = vecIn.begin() + lastIdx;
    typename std::vector<T> tempVec(first, last);
    return tempVec;
}

template<typename T>
void printVec(const std::vector<T> &vecIn)
{
    std::cout << "Vector size : " << vecIn.size() << std::endl;
    std::cout << "[";
    for (unsigned int i = 0; i < vecIn.size(); i++)
        std::cout << vecIn[i] << " ";
    std::cout << "]" << std::endl;
}

template<typename T>
void printArray(T *array, std::size_t size)
{
    std::cout << "Array size : " << size << std::endl;
    std::cout << "[";
    std::cout << std::scientific;
    for (unsigned int i = 0; i < size; i++)
        std::cout << i << ": " << array[i] << " ";
    std::cout << "]" << std::endl;
}

/**
 *	indexOfLargestElement
 *
 *	Returns the index of the largest element in the array.
 *
 *  Inputs:
 *		arr[] - input array
 *		size  - size of array
 *		largestVal - pointer to the address to store the largest value
 *
 *	Outputs:
 *		Returns largestIndexValue
 */
template<typename T>
std::size_t indexOfLargestElement(T arr[], std::size_t size, T *largestVal)
{
    std::size_t largestIndex = 0;
    for (std::size_t index = largestIndex; index < size; index++)
    {
        if (arr[largestIndex] < arr[index])
        {
            largestIndex = index;
        }
    }

    *largestVal = arr[largestIndex];
    return largestIndex;
}

/**
 *	indexOfLargestElement
 *
 *	Returns the index of the largest element in the array.
 *
 *  Inputs:
 *		arr[] - input array
 *		size  - size of array
 *		largestVal - pointer to the address to store the largest value
 *
 *	Outputs:
 *		Returns largestIndexValue
 *
 *	Note: Overloaded for cases where you don't need the actual value, just the index.
 */
template<typename T>
std::size_t indexOfLargestElement(T arr[], std::size_t size)
{
    std::size_t largestIndex = 0;
    for (std::size_t index = largestIndex; index < size; index++)
    {
        if (arr[largestIndex] < arr[index])
        {
            largestIndex = index;
        }
    }

    return largestIndex;
}

/**
 *	circshift
 *
 *	Does a 2D circular shift like the Matlab command.
 *
 *  Inputs:
 *		out* - pointer to a buffer for the result
 *		in*	 - pointer to the input data
 *		xdim - size of the x-dimension
 *		ydim - size of the y-dimension
 *		xshift - shifting to be done along x-dimension
 *		yshift - shifting to be done along y-dimension
 *
 *  Can be further optimized using std::rotate
 */
template<typename T>
inline void circshift(T *in, T *out, int xdim, int ydim, int xshift, int yshift)
{
    if (xshift == 0 && yshift == 0)
    {
        out = in; //-- no change
        return;
    }

    for (int i = 0; i < xdim; i++)
    {
        int ii = (i + xshift) % xdim;
        if (ii < 0)
            ii = xdim + ii;
        for (int j = 0; j < ydim; j++)
        {
            int jj = (j + yshift) % ydim;
            if (jj < 0)
                jj = ydim + jj;
            out[ii * ydim + jj] = in[i * ydim + j];
        }
    }
}

/**
 * Does 1D Circshift (in-place)
 *
 * @param in Input array of values, circshift done directly on this
 * @param ydim Length of the array
 * @param yshift Amount to be shifted (+ve is shift right, -ve is shift left)
 */
template<typename T>
inline void circshift1D_IP(T *in, int ydim, int yshift)
{
    if (yshift == 0)
        return;

    if (yshift > 0) // shift right
    {
        //std::rotate(&in[0], &in[ydim - yshift - 1], &in[ydim - 1]);
        std::rotate(in, in + (ydim - yshift), in + ydim);
    }
    else if (yshift < 0) // shift left
    {
        yshift = abs(yshift);
        //std::rotate(&in[0], &in[yshift], &in[ydim - 1]);
        std::rotate(in, in + yshift, in + ydim);
    }

    return;
}


/**
 * Does 1D Circshift (out-of-place)
 *
 * @param in Input array of values
 * @param out Circshifted array of values
 * @param ydim Length of the array
 * @param yshift Amount to be shifted (+ve is shift right, -ve is shift left)
 */
template<typename T>
inline void circshift1D_OP(T *in, T *out, int ydim, int yshift)
{
    if (yshift == 0)
    {
        out = in; //-- no change
        return;
    }

    memcpy(out, in, ydim * sizeof(T));

    if (yshift > 0) // shift right
    {
        //std::rotate(&out[0], &out[ydim - yshift], &out[ydim]); // TODO check indices may be ydim-yshift
        std::rotate(out, out + (ydim - yshift), out + ydim); // C++ idiom: out + ydim is not used, out + ydim -1 is referenced
    }
    else if (yshift < 0) // shift left
    {
        yshift = abs(yshift);
        //std::rotate(&out[0], &out[yshift], &out[ydim - 1]);
        std::rotate(out, out + yshift, out + ydim); // TODO check
    }

    return;

//    for (int j = 0; j < ydim; j++)
//    {
//        int jj = (j + yshift) % ydim;
//        if (jj < 0)
//            jj = ydim + jj;
//        out[jj] = in[j];
//    }
}


/**
 * Does 1D ifftshift
 * Note: T* out must already by memory allocated!!
 */
template<typename T>
inline void ifftshift1D(T *in, T *out, int ydim)
{
    //-- (ydim & 1)==0
    int pivot = (ydim % 2 == 0) ? (ydim / 2) : ((ydim + 1) / 2);
    //circshift1D(in, out, ydim, shiftBy);

    int rightHalf = ydim-pivot;
    int leftHalf = pivot;
    memcpy(out, in+(pivot), sizeof(T)*rightHalf);
    memcpy(out+rightHalf, in, sizeof(T)*leftHalf);
}

/**
 * Does 1D fftshift
 * Note: T* out must already by memory allocated!!
 */
template<typename T>
inline void fftshift1D(T *in, T *out, int ydim)
{
    int pivot = (ydim % 2 == 0) ? (ydim / 2) : ((ydim - 1) / 2);
    //circshift1D(in, out, ydim, shiftBy);
    int rightHalf = ydim-pivot;
    int leftHalf = pivot;
    memcpy(out, in+(pivot), sizeof(T)*rightHalf);
    memcpy(out+rightHalf, in, sizeof(T)*leftHalf);
}

/**
 * Slow due to the circshift, but works.
 */
template<typename T>
inline void ifftshift2D(T *in, T *out, int xdim, int ydim)
{
    int shiftYBy = (ydim % 2 == 0) ? (ydim / 2) : ((ydim + 1) / 2);
    int shiftXBy = (xdim % 2 == 0) ? (xdim / 2) : ((xdim + 1) / 2);
    circshift(in, out, xdim, ydim, shiftXBy, shiftYBy);
}

/**
 * Slow due to the circshift, but works
 */
template<typename T>
inline void fftshift2D(T *in, T *out, int xdim, int ydim)
{
    int shiftYBy = (ydim % 2 == 0) ? (ydim / 2) : ((ydim - 1) / 2);
    int shiftXBy = (xdim % 2 == 0) ? (xdim / 2) : ((xdim - 1) / 2);
    circshift(in, out, xdim, ydim, shiftXBy, shiftYBy);
}

/************************************************************************************
Function    : void detrend_IP(T *y, T *x, int m)
Description : Remove the linear trend of the input floating point data. Note that this
              will initialize a work buffer inside the function. So if you are calling
              this many, many times, create your work buffer in the calling scope and call
              detrend(T *y, T*x, int m) instead to avoid initializing memory over and over
              again.
Inputs      : y - Floating point input data
              m - Input data length
Outputs     : y - Data with linear trend removed
Copyright   : DSO National Laboratories
History     : 01/02/2008, TCK, Adapted from HYC code
              01/12/2008, TCK, Added in return value
              25/01/2016, Pier, Changed into template type, removed need for work buffer
*************************************************************************************/
template<typename T>
void detrend_IP(T *y, int m)
{
    T xmean, ymean;
    int i;
    T temp;
    T Sxy;
    T Sxx;

    T grad;
    T yint;

    std::unique_ptr<T[]> x(new T[m]);

    /********************************
    Set the X axis Liner Values
    *********************************/
    for (i = 0; i < m; i++)
        x[i] = i;

    /********************************
    Calculate the mean of x and y
    *********************************/
    xmean = 0;
    ymean = 0;
    for (i = 0; i < m; i++)
    {
        xmean += x[i];
        ymean += y[i];
    }
    xmean /= m;
    ymean /= m;

    /********************************
    Calculate Covariance
    *********************************/
    temp = 0;
    for (i = 0; i < m; i++)
        temp += x[i] * y[i];
    Sxy = temp / m - xmean * ymean;

    temp = 0;
    for (i = 0; i < m; i++)
        temp += x[i] * x[i];
    Sxx = temp / m - xmean * xmean;

    /********************************
    Calculate Gradient and Y intercept
    *********************************/
    grad = Sxy / Sxx;
    yint = -grad * xmean + ymean;

    /********************************
    Removing Linear Trend
    *********************************/
    for (i = 0; i < m; i++)
        y[i] = y[i] - (grad * i + yint);

}


/************************************************************************************
Function    : void detrend_OP(T *y, T *x, int m)
Description : Remove the linear trend of the input floating point data
Inputs      : y - Floating point input data
              x - Work buffer (must be initialized in calling scope!)
              m - Input data length
Outputs     : y - Data with linear trend removed
Copyright   : DSO National Laboratories
History     : 01/02/2008, TCK, Adapted from HYC code
              01/12/2008, TCK, Added in return value
              25/01/2016, Pier, Changed into template type
*************************************************************************************/
template<typename T>
void detrend_OP(T *y, T*x, int m)
{
    T xmean, ymean;
    int i;
    T temp;
    T Sxy;
    T Sxx;

    T grad;
    T yint;

    /********************************
    Set the X axis Liner Values
    *********************************/
    for (i = 0; i < m; i++)
        x[i] = i;

    /********************************
    Calculate the mean of x and y
    *********************************/
    xmean = 0;
    ymean = 0;
    for (i = 0; i < m; i++)
    {
        xmean += x[i];
        ymean += y[i];
    }
    xmean /= m;
    ymean /= m;

    /********************************
    Calculate Covariance
    *********************************/
    temp = 0;
    for (i = 0; i < m; i++)
        temp += x[i] * y[i];
    Sxy = temp / m - xmean * ymean;

    temp = 0;
    for (i = 0; i < m; i++)
        temp += x[i] * x[i];
    Sxx = temp / m - xmean * xmean;

    /********************************
    Calculate Gradient and Y intercept
    *********************************/
    grad = Sxy / Sxx;
    yint = -grad * xmean + ymean;

    /********************************
    Removing Linear Trend
    *********************************/
    for (i = 0; i < m; i++)
        y[i] = y[i] - (grad * i + yint);

}




/**
 * Works but stupid and slow. Take a look at simple_transpose_32fc
 */
template<typename T>
void matrixTranspose(T *in, int rows, int cols, T *out)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            *(out + (j * rows) + i) = *(in + (i * cols) + j); // a[i][j] == a + i * col + j
        }
    }
}

/*!
 * Same as matlab's diff in 1 dimension
 * f X is a vector, then diff(X) returns a vector, one element shorter than X, of differences between adjacent elements:
 *[X(2)-X(1) X(3)-X(2) ... X(n)-X(n-1)]
 *
 * @param in Input vector
 * @param out Output vector
 * @param noOfElements Self-explanatory
 */
template<typename T>
void diff(T *in, T *out, int noOfElements)
{
    for (int i = 0; i < noOfElements - 1; i++)
        out[i] = in[i + 1] - in[i];
}



void printArrayIpp32fc(Ipp32fc *array, std::size_t size)
{
    std::cout << "Array size : " << size << std::endl;
    std::cout << "[";
    std::cout << std::scientific;
    for (unsigned int i = 0; i < size; i++)
        std::cout << i << ": " << array[i].re << "," << array[i].im << " " << std::endl;
    std::cout << "]" << std::endl;
    std::cout << " " << std::endl;
}


/**
 *	subArray
 *
 *	Returns subset of input array given by start and end.
 *
 *  Inputs:
 *		arrayIn - Input array
 *		start - location of beginning of subset vector
 *		end - location of end of subset vector
 *		subArrayOut - output subset array
 *
 *  Notes: The output arary is inclusive of arrayIn[start] and arrayIn[end]
 */
void subArray(const Ipp32f *arrayIn, int start, int end, Ipp32f *subArrayOut)
{
    //Ipp32f mag_sum[end-start + 1];
    int count = 0;

    for (int i = start; i <= end; i++)
    {
        subArrayOut[count] = arrayIn[i];
        count++;
    }
}

/**
 *	mean
 *
 *	Returns mean of input array subset given by start and end.
 *
 *  Inputs:
 *		arrayIn - Input array
 *		start - location of beginning of array to calculate mean
 *		end - location of end of array for mean calculation
 *
 *	Outputs:
 *		mean
 *
 *  Notes:
 *      Total is inclusive of array[start] and array[end] when calculating mean
 */
Ipp32f mean(Ipp32f array[], int start, int end)
{

    double total = 0.0;
    for (int i = start; i <= end; i++)
    {
        total += array[i];
//		std::cout << std::scientific;
//		std::cout << i << " : " << array[i] << std::endl;
    }
//	std::cout << std::scientific;
//	std::cout << "total" << " : " << total << std::endl;
    return (Ipp32f) (total / (end - start + 1));
}

/**
 *	stddev
 *
 *	Returns standard deviation of input array subset given by start and end.
 *
 *  Inputs:
 *		arrayIn - Input array
 *		start - location of beginning of array to calculate standard deviation
 *		end - location of end of array for standard deviation calculation
 *
 *	Outputs:
 *		standard deviation
 *
 *  Notes:
 *      Total is inclusive of array[start] and array[end] when calculating standard deviation
 */
Ipp32f stddev(Ipp32f array[], int start, int end)
{

    long double sum = 0.0;
    Ipp32f meanVar = mean(array, start, end);

    for (int j = start; j <= end; j++)
    {
        //sum += pow((array[j]-meanVar), 2);
        //long double mulVal = (array[j]-meanVar) * (array[j]-meanVar);
        long double mulVal = pow((array[j] - meanVar), 2);
        sum += mulVal;
    }


    return (Ipp32f) sqrt((sum / (end - start))); //--  -1 from (end-start+1) (stddev formula has -1)
}

/**
 *	cumulativeSum
 *
 *	Returns the cumulative sum of the elements in the array
 *
 *  Inputs:
 *		arrayIn - Input array
 *		start - location of beginning of array to calculate standard deviation
 *		end - location of end of array for standard deviation calculation
 *
 *	Outputs:
 *		standard deviation
 *
 *  Notes:
 *      Total is inclusive of array[start] and array[end] when calculating standard deviation
 */
void cumulativeSum(Ipp32f *array, int size)
{

    if (size < 0) return;
    cumulativeSum(array, size - 1);
    array[size + 1] += array[size];
    //std::cout << "size[" << size+1 << "] += [ " << size << "]" << std::endl;
}


/**
 *	colonRangeVec
 *
 *	Returns a vector of range values according to the Matlab J:D:K syntax
 *	J:D:K  is the same as [J, J+D, ..., J+m*D] where m = fix((K-J)/D).
 *
 *  Inputs:
 *		startVal - Equivalent of J
 *		granularity - Equivalent of D
 *		endVal - Equivalent of J+m*D
 *
 *	Outputs:
 *		Returns vector of float values
 */
std::vector<double> colonRangeVec(double J, double D, double K)
{
    int m = static_cast<int>(((K - J) / D));
    std::vector<double> v;

    //-- Return empty vector if hit the cases below
    if (D == 0 || ((D > 0) && (J > K)) || ((D < 0) && (J < K)))
    {
        return v;
    }

    //-- Else create the vector
    for (int i = 0; i <= m; i++)
    {
        double newVal = J + i * D;
        v.insert(v.begin()+i, newVal);
    }

    return v;
}


/**
 *	round_to_digits
 *
 *	Rounds a double value to number of significant digits specified by input.
 *
 *  Inputs:
 *  	value - the input value to be rounded
 *  	digits - number of significant digits
 *
 *	Outputs:
 *		Returns double value rounded to the specified number of significant digits
 */
double round_to_digits(double value, int digits)
{
    if (value == 0.0)
        return 0.0;

    double factor = pow(10.0, digits - ceil(log10(fabs(value))));
    return round(value*factor) / factor;
}

//-- Equivalent to what IRL is using for Taylor Window (Ken Yew)
void Common_TaylorWin(float *wt, int len)
{
    /******************************
        Constant Parameters -
        Please ask Ken Yew if u want to change (quit liao)
     ******************************/
    int n = 5;
    double SLL = -35.0;
    const double pi = 4.0f * atan(1.0f);
    /******************************
        Parameters
     ******************************/
    double A, B;
    double sigma2;
    int m, i, k;
    double tmp[5][5];
    double Fm_num[5] = {0};
    double Fm_den[5] = {0};
    double Fm[5] = {0};

    /******************************
        Compute Constants
     ******************************/
    B = pow(10.0, -SLL / 20.0);
    A = log(B + sqrt(B * B - 1)) / pi;
    sigma2 = (double) n * (double) n / (A * A + ((double) n - 0.5) * ((double) n - 0.5));

    /******************************
        Generate Numerator
     ******************************/
    /*lint -e834 */
    for (m = 1; m < n; m++)
    {
        Fm_num[m] = 1.0;
        for (i = 1; i < n; i++)
        {
            tmp[m][i] = 1 - (double) m * (double) m / sigma2 * (1 / (A * A + ((double) i - 0.5) * ((double) i - 0.5)));
            Fm_num[m] = Fm_num[m] * tmp[m][i];
        }
        Fm_num[m] = pow(-1.0, (double) (m + 1)) * Fm_num[m];
    }

    /******************************
        Generate Denominator
     ******************************/
    for (m = 1; m < n; m++)
    {
        Fm_den[m] = 1.0;
        for (i = 1; i < n; i++)
        {
            tmp[m][i] = 1 - ((double) m * (double) m) / ((double) i * (double) i);
            if (i == m)
            {
                tmp[m][i] = 1.0;
            }
            Fm_den[m] = Fm_den[m] * tmp[m][i];
        }
        Fm_den[m] = 2 * Fm_den[m];
    }

    /******************************
        Generate Weights
     ******************************/
    for (m = 1; m < n; m++)
        Fm[m] = Fm_num[m] / Fm_den[m];

    for (k = 0; k < len; k++)
    {
        wt[k] = 0;
        for (m = 1; m < n; m++)
            wt[k] += (float) (Fm[m] *
                              cos((2 * pi * (double) m * ((double) k - (double) len / 2.0 + 0.5)) / (double) len));
        wt[k] = 1 + 2 * wt[k];
    }
}

//-- Transpose Ipp32fc matrix using ippi. (Even though we are using ipps mostly)
//-- Can do this because 16bits * 4 = 64bits == sizeof(Ipp32fc)
//-- Note: If Ipp32f, should use ippiTranspose_8u_C4R (32 bits) == sizeof(Ipp32f)
void simple_transpose_32fc(Ipp32fc *src, Ipp32fc *dst, int nrows, int ncols)
{
    int src_stride = ncols * sizeof(*src);
    int dst_stride = nrows * sizeof(*dst);
    // Note that IPPI uses [col, row] for Roi
    IppiSize srcRoi = {ncols, nrows};
    ippiTranspose_16u_C4R((Ipp16u *) src, src_stride, (Ipp16u *) dst, dst_stride, srcRoi);
}

/**
 * @brief Returns a vector containing the prime factors of n
 *
 * @param [in] The number to find the prime factors for
 * @return
 */
std::vector<int> primeFactors(int n)
{
    std::vector<int> vec;

    while (n % 2 == 0)
    {
        vec.push_back(2);
        n /= 2;
    }

    for (int i = 3; i <= sqrt(n); i += 2)
    {
        while (n % i == 0)
        {
            vec.push_back(i);
            n /= i;
        }
    }

    if (n > 2)
        vec.push_back(n);

//    std::cout << "Prime factors:" << std::endl;
//    for (int j=0; j < vec.size(); j++)
//    {
//        printf("%d ", vec[j]);
//    }
//    printf("\n");
    return vec;
}

/**
 * @brief Used to find the appropriate fft integer for the input n
 * This uses the "formula" (N + D - 1)/D * D
 * Criteria: Output nfft should be a factor of 2,3,5
 *
 * @param [in] Integer to find nfft for
 */
int findNFFT(int n)
{
    std::vector<int> ansPrimes;
    std::vector<int> firstPrimes;

    int d = 0;

    do
    {
        if (n > 2048) d = 512;
        else if (n > 1024) d = 256;
        else if (n > 128) d = 64;
        else if (n > 32) d = 32;
        else if (n > 8) d = 8;
        else d = 2;

        int fn = (n + d - 1) / d * d;
        firstPrimes = primeFactors(fn);

        for (int i = 0; i < firstPrimes.size(); i++)
        {
            if (firstPrimes[i] == 2 || firstPrimes[i] == 3 || firstPrimes[i] == 5)
            {
                ansPrimes.push_back(firstPrimes[i]);
                firstPrimes.erase(firstPrimes.begin() + i);
                i -= 1;
            }
        }

        int newN = 1;
        if (firstPrimes.size() > 0)
        {
            for (int i = 0; i < firstPrimes.size(); i++)
                newN *= firstPrimes[i];
        }

        n = newN;
        firstPrimes = {};

    } while (n != 1); // if n == 1 means that firstPrimes

    int ans = 1;
    for (int i = 0; i < ansPrimes.size(); i++)
        ans *= ansPrimes[i];

    return ans;
}



