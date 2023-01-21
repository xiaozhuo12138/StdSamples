#pragma once

namespace Octopus
{
    struct OctopusColVectorXf : public FloatColumnVector
    {
        OctopusColVectorXf() = default;
        OctopusColVectorXf(const FloatColumnVector &v) : FloatColumnVector(v) {}
        OctopusColVectorXf(size_t i) : FloatColumnVector(i) {}

        using FloatColumnVector::operator =;
        using FloatColumnVector::operator ();
        using FloatColumnVector::insert;
        //using FloatColumnVector::append;
        using FloatColumnVector::fill;
        using FloatColumnVector::extract;
        using FloatColumnVector::extract_n;
        using FloatColumnVector::transpose;
        using FloatColumnVector::size;
        using FloatColumnVector::min;
        using FloatColumnVector::max;
        using FloatColumnVector::resize;
        using FloatColumnVector::clear;


        #ifdef SWIG
        %extend {

            void insert(const OctopusColVectorXf& v, size_t i) {
                $self->insert(v,i);
            }
            void append(const OctopusColVectorXf& v) {
                $self->append(v);
            }
            void fill(double value) {
                $self->fill(value);
            }
            void fill(double value, size_t c1, size_t c2) {
                $self->fill(value,c1,c2);
            }
            //OctopusColVectorXf transpose();
            
            OctopusColVectorXf extract(size_t c1, size_t c2) { return $self->extract(c1,c2); }
            OctopusColVectorXf extract_n(size_t c1, size_t n) { return $self->extract(c1,n); }

            double min() const { return $self->min(); }
            double max() const { return $self->max(); }

            void resize(size_t n, const double& rfv=0) { $self->resize(n,rfv); }
            void clear(size_t n) { $self->clear(n); }
        }
        #endif

        OctopusColVectorXf operator + (const OctopusColVectorXf & b) {
            OctopusColVectorXf r;
            ValueList l;
            l(0) = "add";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_column_vector_value();
            return r;
        }
        OctopusColVectorXf operator - (const OctopusColVectorXf & b) {
            OctopusColVectorXf r;
            ValueList l;
            l(0) = "sub";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_column_vector_value();
            return r;
        }
        OctopusColVectorXf operator * (const OctopusColVectorXf & b) {
            OctopusColVectorXf r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_column_vector_value();
            return r;
        }
        OctopusColVectorXf operator / (const OctopusColVectorXf & b) {
            OctopusColVectorXf r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_column_vector_value();
            return r;
        }
        OctopusColVectorXf operator + (const float b) {
            OctopusColVectorXf r;
            ValueList l;
            l(0) = "plus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_column_vector_value();
            return r;
        }
        OctopusColVectorXf operator - (const float b) {
            OctopusColVectorXf r;
            ValueList l;
            l(0) = "minus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_column_vector_value();
            return r;
        }
        OctopusColVectorXf operator * (const float b) {
            OctopusColVectorXf r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_column_vector_value();
            return r;
        }
        OctopusColVectorXf operator / (const float b) {
            OctopusColVectorXf r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_column_vector_value();
            return r;
        }   
    };
    struct OctopusColVectorXd: public ColumnVector
    {
        OctopusColVectorXd() = default;
        OctopusColVectorXd(const ColumnVector &v) : ColumnVector(v) {}
        OctopusColVectorXd(size_t i) : ColumnVector(i) {}

        using ColumnVector::operator =;
        using ColumnVector::operator ();
        using ColumnVector::insert;
        //using ColumnVector::append;
        using ColumnVector::fill;
        using ColumnVector::extract;
        using ColumnVector::extract_n;
        using ColumnVector::transpose;
        using ColumnVector::size;
        using ColumnVector::min;
        using ColumnVector::max;
        using ColumnVector::resize;
        using ColumnVector::clear;

        #ifdef SWIG
        %extend {
            void insert(const OctopusColVectorXd& v, size_t i) {
                $self->insert(v,i);
            }
            void append(const OctopusColVectorXd& v) {
                $self->append(v);
            }
            void fill(double value) {
                $self->fill(value);
            }
            void fill(double value, size_t c1, size_t c2) {
                $self->fill(value,c1,c2);
            }

            //OctopusColVectorXd transpose();
            
            OctopusColVectorXd extract(size_t c1, size_t c2) { return $self->extract(c1,c2); }
            OctopusColVectorXd extract_n(size_t c1, size_t n) { return $self->extract(c1,n); }

            double min() const { return $self->min(); }
            double max() const { return $self->max(); }

            void resize(size_t n, const double& rfv=0) { $self->resize(n,rfv); }
            void clear(size_t n) { $self->clear(n); }
        }
        #endif


        OctopusColVectorXd operator + (const OctopusColVectorXd & b) {
            OctopusColVectorXd r;
            ValueList l;
            l(0) = "plus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).column_vector_value();
            return r;
        }
        OctopusColVectorXd operator - (const OctopusColVectorXd & b) {
            OctopusColVectorXd r;
            ValueList l;
            l(0) = "minus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).column_vector_value();
            return r;
        }
        OctopusColVectorXd operator * (const OctopusColVectorXd & b) {
            OctopusColVectorXd r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).column_vector_value();
            return r;
        }
        OctopusColVectorXd operator / (const OctopusColVectorXd & b) {
            OctopusColVectorXd r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).column_vector_value();
            return r;
        }
        OctopusColVectorXd operator + (const double b) {
            OctopusColVectorXd r;
            ValueList l;
            l(0) = "plus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).column_vector_value();
            return r;
        }
        OctopusColVectorXd operator - (const double b) {
            OctopusColVectorXd r;
            ValueList l;
            l(0) = "minus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).column_vector_value();
            return r;
        }
        OctopusColVectorXd operator * (const double b) {
            OctopusColVectorXd r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).column_vector_value();
            return r;
        }
        OctopusColVectorXd operator / (const double b) {
            OctopusColVectorXd r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).column_vector_value();
            return r;
        }   
    };
    struct OctopusColVectorXcf : public FloatComplexColumnVector
    {
        OctopusColVectorXcf() = default;
        OctopusColVectorXcf(const FloatComplexColumnVector &v) : FloatComplexColumnVector(v) {}
        OctopusColVectorXcf(size_t i) : FloatComplexColumnVector(i) {}

        using FloatComplexColumnVector::operator =;
        using FloatComplexColumnVector::operator ();
        using FloatComplexColumnVector::insert;
        //using FloatComplexColumnVector::append;
        using FloatComplexColumnVector::fill;
        using FloatComplexColumnVector::extract;
        using FloatComplexColumnVector::extract_n;
        using FloatComplexColumnVector::transpose;
        using FloatComplexColumnVector::size;
        using FloatComplexColumnVector::min;
        using FloatComplexColumnVector::max;
        using FloatComplexColumnVector::resize;
        using FloatComplexColumnVector::clear;

        #ifdef SWIG
        %extend {
            void insert(const OctopusColVectorXd& v, size_t i) {
                $self->insert(v,i);
            }
            void append(const OctopusColVectorXd& v) {
                $self->append(v);
            }
            void fill(double value) {
                $self->fill(value);
            }
            void fill(double value, size_t c1, size_t c2) {
                $self->fill(value,c1,c2);
            }

            //OctopusColVectorXd transpose();
            
            OctopusColVectorXd extract(size_t c1, size_t c2) { return $self->extract(c1,c2); }
            OctopusColVectorXd extract_n(size_t c1, size_t n) { return $self->extract(c1,n); }

            double min() const { return $self->min(); }
            double max() const { return $self->max(); }

            void resize(size_t n, const double& rfv=0) { $self->resize(n,rfv); }
            void clear(size_t n) { $self->clear(n); }
        }
        #endif
        OctopusColVectorXcf operator + (const OctopusColVectorXcf & b) {
            OctopusColVectorXcf r;
            ValueList l;
            l(0) = "add";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcf operator - (const OctopusColVectorXcf & b) {
            OctopusColVectorXcf r;
            ValueList l;
            l(0) = "sub";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcf operator * (const OctopusColVectorXcf & b) {
            OctopusColVectorXcf r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcf operator / (const OctopusColVectorXcf & b) {
            OctopusColVectorXcf r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcf operator + (const std::complex<float> b) {
            OctopusColVectorXcf r;
            ValueList l;
            l(0) = "plus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcf operator - (const std::complex<float> b) {
            OctopusColVectorXcf r;
            ValueList l;
            l(0) = "minus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcf operator * (const std::complex<float> b) {
            OctopusColVectorXcf r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcf operator / (const std::complex<float> b) {
            OctopusColVectorXcf r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_column_vector_value();
            return r;
        }   
    };
    struct OctopusColVectorXcd: public ComplexColumnVector
    {
        OctopusColVectorXcd() = default;
        OctopusColVectorXcd(const ComplexColumnVector &v) : ComplexColumnVector(v) {}
        OctopusColVectorXcd(size_t i) : ComplexColumnVector(i) {}

        using ComplexColumnVector::operator =;
        using ComplexColumnVector::operator ();
        using ComplexColumnVector::insert;
        //using ComplexColumnVector::append;
        using ComplexColumnVector::fill;
        using ComplexColumnVector::extract;
        using ComplexColumnVector::extract_n;
        using ComplexColumnVector::transpose;
        using ComplexColumnVector::size;
        using ComplexColumnVector::min;
        using ComplexColumnVector::max;
        using ComplexColumnVector::resize;
        using ComplexColumnVector::clear;

        #ifdef SWIG
        %extend {

            void insert(const OctopusColVectorXcd& v, size_t i) {
                $self->insert(v,i);
            }
            void append(const OctopusColVectorXcd& v) {
                $self->append(v);
            }
            void fill(double value) {
                $self->fill(value);
            }
            void fill(double value, size_t c1, size_t c2) {
                $self->fill(value,c1,c2);
            }
            //OctopusColVectorXcd transpose();
            
            OctopusColVectorXcd extract(size_t c1, size_t c2) { return $self->extract(c1,c2); }
            OctopusColVectorXcd extract_n(size_t c1, size_t n) { return $self->extract(c1,n); }

            double min() const { return $self->min(); }
            double max() const { return $self->max(); }

            void resize(size_t n, const double& rfv=0) { $self->resize(n,rfv); }
            void clear(size_t n) { $self->clear(n); }
        }
        #endif

        OctopusColVectorXcd operator + (const OctopusColVectorXcd & b) {
            OctopusColVectorXcd r;
            ValueList l;
            l(0) = "add";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcd operator - (const OctopusColVectorXcd & b) {
            OctopusColVectorXcd r;
            ValueList l;
            l(0) = "sub";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcd operator * (const OctopusColVectorXcd & b) {
            OctopusColVectorXcd r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcd operator / (const OctopusColVectorXcd & b) {
            OctopusColVectorXcd r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcd operator + (const std::complex<double> b) {
            OctopusColVectorXcd r;
            ValueList l;
            l(0) = "plus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcd operator - (const std::complex<double> b) {
            OctopusColVectorXcd r;
            ValueList l;
            l(0) = "minus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcd operator * (const std::complex<double> b) {
            OctopusColVectorXcd r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_column_vector_value();
            return r;
        }
        OctopusColVectorXcd operator / (const std::complex<double> b) {
            OctopusColVectorXcd r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_column_vector_value();
            return r;
        }   
    };
}