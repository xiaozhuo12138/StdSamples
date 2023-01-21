#pragma once

namespace Octopus
{
    struct OctopusMatrixXf : public FloatMatrix
    {
        OctopusMatrixXf() = default;
        OctopusMatrixXf(const FloatMatrix &v) : FloatMatrix(v) {}
        OctopusMatrixXf(size_t i,size_t j) : FloatMatrix(i,j) {}

        using FloatMatrix::operator =;
        using FloatMatrix::operator ();
        using FloatMatrix::insert;
        using FloatMatrix::append;
        using FloatMatrix::fill;
        using FloatMatrix::extract;
        using FloatMatrix::extract_n;
        using FloatMatrix::transpose;
        using FloatMatrix::rows;
        using FloatMatrix::cols;
        using FloatMatrix::row;
        //using FloatMatrix::col;
        
        #ifdef SWIG
        %extend {
            void insert(const OctopusMatrixXf& v, size_t i) {
                $self->insert(v,i);
            }
            void append(const OctopusMatrixXf& v) {
                $self->append(v);
            }
            void fill(double value) {
                $self->fill(value);
            }
            void fill(double value, size_t c1, size_t c2) {
                $self->fill(value,c1,c2);
            }
            OctopusMatrixXf extract(size_t c1, size_t c2) { return $self->extract(c1,c2); }
            OctopusMatrixXf extract_n(size_t c1, size_t n) { return $self->extract(c1,n); }
            OctopusMatrixXf transpose() { return $self->transpose(); }

            double min() const { return $self->min(); }
            double max() const { return $self->max(); }

            void resize(size_t n, const double& rfv=0) { $self->resize(n,rfv); }
            void clear(size_t n) { $self->clear(n); }

        }
        #endif

        OctopusMatrixXf operator + (const OctopusMatrixXf & b) {
            OctopusMatrixXf r;
            ValueList l;
            l(0) = "add";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_matrix_value();
            return r;
        }
        OctopusMatrixXf operator - (const OctopusMatrixXf & b) {
            OctopusMatrixXf r;
            ValueList l;
            l(0) = "sub";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_matrix_value();
            return r;
        }
        OctopusMatrixXf operator * (const OctopusMatrixXf & b) {
            OctopusMatrixXf r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_matrix_value();
            return r;
        }
        OctopusMatrixXf operator / (const OctopusMatrixXf & b) {
            OctopusMatrixXf r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_matrix_value();
            return r;
        }
        OctopusMatrixXf operator + (const float b) {
            OctopusMatrixXf r;
            ValueList l;
            l(0) = "plus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_matrix_value();
            return r;
        }
        OctopusMatrixXf operator - (const float b) {
            OctopusMatrixXf r;
            ValueList l;
            l(0) = "minus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_matrix_value();
            return r;
        }
        OctopusMatrixXf operator * (const float b) {
            OctopusMatrixXf r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_matrix_value();
            return r;
        }
        OctopusMatrixXf operator / (const float b) {
            OctopusMatrixXf r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_matrix_value();
            return r;
        }   

        OctopusMatrixXf& operator += (const OctopusMatrixXf & b) {
            *this = *this + b;
            return *this;
        }
        OctopusMatrixXf& operator -= (const OctopusMatrixXf & b) {
            *this = *this - b;
            return *this;
        }
        OctopusMatrixXf& operator *= (const OctopusMatrixXf & b) {
            *this = *this * b;
            return *this;
        }
        OctopusMatrixXf& operator /= (const OctopusMatrixXf & b) {
            *this = *this / b;
            return *this;
        }
        OctopusMatrixXf& operator += (const float b) {
            *this = *this + b;
            return *this;
        }
        OctopusMatrixXf& operator -= (const float b) {
            *this = *this - b;
            return *this;
        }
        OctopusMatrixXf& operator *= (const float b) {
            *this = *this * b;
            return *this;
        }
        OctopusMatrixXf& operator /= (const float b) {
            *this = *this / b;
            return *this;
        }
        

        OctopusMatrixXf addToEachRow(OctopusRowVectorXf & v) {
            OctopusMatrixXf r(*this);
            
            for(size_t i = 0; i < rows(); i++)
            {
                for(size_t j = 0; j < cols(); j++)
                {
                    r(i,j) += v(j);
                }
            }
            return r;
        }
        OctopusMatrixXf addToEachRow(OctopusMatrixXf & v) {
            OctopusMatrixXf r(*this);
            for(size_t i = 0; i < rows(); i++)
            {
                for(size_t j = 0; j < cols(); j++)
                {
                    r(i,j) += v(0,j);
                }
            }
            return r;
        }
        OctopusMatrixXf eval() {
            OctopusMatrixXf r(*this);
            return r;
        }
        void printrowscols() const {
            std::cout << "rows=" << rows() << " cols=" << cols() << std::endl;
        }
        
        OctopusMatrixXf matmul(const OctopusMatrixXf & b)
        {            
            return *this * b;
        }
        OctopusMatrixXf hadamard(const OctopusMatrixXf & b)
        {            
            OctopusMatrixXf r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_matrix_value();
            return r;
        }            
    };   

    struct OctopusMatrixXd : public Matrix
    {
        OctopusMatrixXd() = default;
        OctopusMatrixXd(const Matrix &v) : Matrix(v) {}
        OctopusMatrixXd(size_t i,size_t j) : Matrix(i,j) {}

        using Matrix::operator =;
        using Matrix::operator ();
        using Matrix::insert;
        using Matrix::append;
        using Matrix::fill;
        using Matrix::extract;
        using Matrix::extract_n;
        using Matrix::transpose;
        using Matrix::size;
        using Matrix::min;
        using Matrix::max;
        using Matrix::resize;
        using Matrix::clear;
        using Matrix::rows;
        using Matrix::cols;
        using Matrix::row;

        #ifdef SWIG
        %extend {
            void insert(const OctopusMatrixXd& v, size_t i) {
                $self->insert(v,i);
            }
            void append(const OctopusMatrixXd& v) {
                $self->append(v);
            }
            void fill(double value) {
                $self->fill(value);
            }
            void fill(double value, size_t c1, size_t c2) {
                $self->fill(value,c1,c2);
            }
            //OctopusMatrixXd transpose();
            
            OctopusMatrixXd extract(size_t c1, size_t c2) { return $self->extract(c1,c2); }
            OctopusMatrixXd extract_n(size_t c1, size_t n) { return $self->extract(c1,n); }

            double min() const { return $self->min(); }
            double max() const { return $self->max(); }

            void resize(size_t n, const double& rfv=0) { $self->resize(n,rfv); }
            void clear(size_t n) { $self->clear(n); }
        }
        #endif
        OctopusMatrixXd operator + (const OctopusMatrixXd & b) {
            OctopusMatrixXd r;
            ValueList l;
            l(0) = "add";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).matrix_value();
            return r;
        }
        OctopusMatrixXd operator - (const OctopusMatrixXd & b) {
            OctopusMatrixXd r;
            ValueList l;
            l(0) = "sub";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).matrix_value();
            return r;
        }
        OctopusMatrixXd operator * (const OctopusMatrixXd & b) {
            OctopusMatrixXd r;
            ValueList l;
            l(0) = "mul";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).matrix_value();
            return r;
        }
        OctopusMatrixXd operator / (const OctopusMatrixXd & b) {
            OctopusMatrixXd r;
            ValueList l;
            l(0) = "div";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).matrix_value();
            return r;
        }
        OctopusMatrixXd operator + (const double b) {
            OctopusMatrixXd r;
            ValueList l;
            l(0) = "plus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).matrix_value();
            return r;
        }
        OctopusMatrixXd operator - (const double b) {
            OctopusMatrixXd r;
            ValueList l;
            l(0) = "minus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).matrix_value();
            return r;
        }
        OctopusMatrixXd operator * (const double b) {
            OctopusMatrixXd r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).matrix_value();
            return r;
        }
        OctopusMatrixXd operator / (const double b) {
            OctopusMatrixXd r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).matrix_value();
            return r;
        }   
    };    
    struct OctopusMatrixXcf : public FloatComplexMatrix
    {
        OctopusMatrixXcf() = default;
        OctopusMatrixXcf(const FloatComplexMatrix &v) : FloatComplexMatrix(v) {}
        OctopusMatrixXcf(size_t i,size_t j) : FloatComplexMatrix(i,j) {}

        using FloatComplexMatrix::operator =;
        using FloatComplexMatrix::operator ();
        using FloatComplexMatrix::insert;
        using FloatComplexMatrix::append;
        using FloatComplexMatrix::fill;
        using FloatComplexMatrix::extract;
        using FloatComplexMatrix::extract_n;
        using FloatComplexMatrix::transpose;
        using FloatComplexMatrix::size;
        using FloatComplexMatrix::min;
        using FloatComplexMatrix::max;
        using FloatComplexMatrix::resize;
        using FloatComplexMatrix::clear;
        using FloatComplexMatrix::rows;
        using FloatComplexMatrix::cols;
        using FloatComplexMatrix::row;

        #ifdef SWIG
        %extend {
            void insert(const OctopusMatrixXcf& v, size_t i) {
                $self->insert(v,i);
            }
            void append(const OctopusMatrixXcf& v) {
                $self->append(v);
            }
            void fill(double value) {
                $self->fill(value);
            }
            void fill(double value, size_t c1, size_t c2) {
                $self->fill(value,c1,c2);
            }
            //OctopusMatrixXcf transpose();
            
            OctopusMatrixXcf extract(size_t c1, size_t c2) { return $self->extract(c1,c2); }
            OctopusMatrixXcf extract_n(size_t c1, size_t n) { return $self->extract(c1,n); }

            double min() const { return $self->min(); }
            double max() const { return $self->max(); }

            void resize(size_t n, const double& rfv=0) { $self->resize(n,rfv); }
            void clear(size_t n) { $self->clear(n); }
        }
        #endif
        
        OctopusMatrixXcf operator + (const OctopusMatrixXcf & b) {
            OctopusMatrixXcf r;
            ValueList l;
            l(0) = "add";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_matrix_value();
            return r;
        }
        OctopusMatrixXcf operator - (const OctopusMatrixXcf & b) {
            OctopusMatrixXcf r;
            ValueList l;
            l(0) = "sub";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_matrix_value();
            return r;
        }
        OctopusMatrixXcf operator * (const OctopusMatrixXcf & b) {
            OctopusMatrixXcf r;
            ValueList l;
            l(0) = "mul";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_matrix_value();
            return r;
        }
        OctopusMatrixXcf operator / (const OctopusMatrixXcf & b) {
            OctopusMatrixXcf r;
            ValueList l;
            l(0) = "div";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_matrix_value();
            return r;
        }
        OctopusMatrixXcf operator + (const std::complex<float> b) {
            OctopusMatrixXcf r;
            ValueList l;
            l(0) = "plus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_matrix_value();
            return r;
        }
        OctopusMatrixXcf operator - (const std::complex<float> b) {
            OctopusMatrixXcf r;
            ValueList l;
            l(0) = "minus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_matrix_value();
            return r;
        }
        OctopusMatrixXcf operator * (const std::complex<float> b) {
            OctopusMatrixXcf r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_matrix_value();
            return r;
        }
        OctopusMatrixXcf operator / (const std::complex<float> b) {
            OctopusMatrixXcf r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_matrix_value();
            return r;
        }   
    };    
    struct OctopusMatrixXcd : public ComplexMatrix
    {
        OctopusMatrixXcd() = default;
        OctopusMatrixXcd(const ComplexMatrix &v) : ComplexMatrix(v) {}
        OctopusMatrixXcd(size_t i,size_t j) : ComplexMatrix(i,j) {}

        using ComplexMatrix::operator =;
        using ComplexMatrix::operator ();
        using ComplexMatrix::insert;
        using ComplexMatrix::append;
        using ComplexMatrix::fill;
        using ComplexMatrix::extract;
        using ComplexMatrix::extract_n;
        using ComplexMatrix::transpose;
        using ComplexMatrix::size;
        using ComplexMatrix::min;
        using ComplexMatrix::max;
        using ComplexMatrix::resize;        
        using ComplexMatrix::clear;
        using ComplexMatrix::rows;
        using ComplexMatrix::cols;
        using ComplexMatrix::row;

        
        OctopusMatrixXcd operator + (const OctopusMatrixXcd & b) {
            OctopusMatrixXcd r;
            ValueList l;
            l(0) = "add";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_matrix_value();
            return r;
        }
        OctopusMatrixXcd operator - (const OctopusMatrixXcd & b) {
            OctopusMatrixXcd r;
            ValueList l;
            l(0) = "sub";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_matrix_value();
            return r;
        }
        OctopusMatrixXcd operator * (const OctopusMatrixXcd & b) {
            OctopusMatrixXcd r;
            ValueList l;
            l(0) = "mul";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_matrix_value();
            return r;
        }
        OctopusMatrixXcd operator / (const OctopusMatrixXcd & b) {
            OctopusMatrixXcd r;
            ValueList l;
            l(0) = "div";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_matrix_value();
            return r;
        }
        OctopusMatrixXcd operator + (const std::complex<double> b) {
            OctopusMatrixXcd r;
            ValueList l;
            l(0) = "plus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_matrix_value();
            return r;
        }
        OctopusMatrixXcd operator - (const std::complex<double> b) {
            OctopusMatrixXcd r;
            ValueList l;
            l(0) = "minus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_matrix_value();
            return r;
        }
        OctopusMatrixXcd operator * (const std::complex<double> b) {
            OctopusMatrixXcd r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_matrix_value();
            return r;
        }
        OctopusMatrixXcd operator / (const std::complex<double> b) {
            OctopusMatrixXcd r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_matrix_value();
            return r;
        }   
    };        
}