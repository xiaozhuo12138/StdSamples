#pragma once

namespace Octopus
{
    struct OctopusRowVectorXf : public FloatRowVector
    {
        OctopusRowVectorXf() : FloatRowVector() {}
        OctopusRowVectorXf(const FloatRowVector & v) : FloatRowVector(v) {}
        OctopusRowVectorXf(size_t i) : FloatRowVector(i) {}

        using FloatRowVector::operator =;
        using FloatRowVector::operator ();
        using FloatRowVector::insert;
        using FloatRowVector::append;
        using FloatRowVector::fill;
        using FloatRowVector::extract;
        using FloatRowVector::extract_n;
        using FloatRowVector::transpose;
        using FloatRowVector::size;
        using FloatRowVector::min;
        using FloatRowVector::max;
        using FloatRowVector::resize;
        using FloatRowVector::clear;

        #ifdef SWIG
        %extend
        {
            void insert(const OctopusRowVectorXf& v, size_t i) {
                $self->insert(v,i);
            }
            void append(const OctopusRowVectorXf& v) {
                $self->append(v);
            }
            void fill(float value) {
                $self->fill(value);
            }
            void fill(float value, size_t c1, size_t c2) {
                $self->fill(value,c1,c2);
            }
            /*
            OctopusColVectorXf transpose() {
                return OctopusColVectorXf($self->transpose(); 
            }
            */
            OctopusRowVectorXf extract(size_t c1, size_t c2) { return $self->extract(c1,c2); }
            OctopusRowVectorXf extract_n(size_t c1, size_t n) { return $self->extract(c1,n); }

            float min() const { return $self->min(); }
            float max() const { return $self->max(); }

            void resize(size_t n, const float& rfv=0) { $self->resize(n,rfv); }
            void clear(size_t n) { $self->clear(n); }        

            void size() const { return $self->size(); }
        }
        #endif

        OctopusRowVectorXf operator + (const OctopusRowVectorXf & b) {
            OctopusRowVectorXf r;
            ValueList l;
            l(0) = "add";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_row_vector_value();
            return r;
        }
        OctopusRowVectorXf operator - (const OctopusRowVectorXf & b) {
            OctopusRowVectorXf r;
            ValueList l;
            l(0) = "sub";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_row_vector_value();
            return r;
        }
        OctopusRowVectorXf operator * (const OctopusRowVectorXf & b) {
            OctopusRowVectorXf r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_row_vector_value();
            return r;
        }
        OctopusRowVectorXf operator / (const OctopusRowVectorXf & b) {
            OctopusRowVectorXf r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_row_vector_value();
            return r;
        }
        OctopusRowVectorXf operator + (const float b) {
            OctopusRowVectorXf r;
            ValueList l;
            l(0) = "plus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_row_vector_value();
            return r;
        }
        OctopusRowVectorXf operator - (const float b) {
            OctopusRowVectorXf r;
            ValueList l;
            l(0) = "minus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_row_vector_value();
            return r;
        }
        OctopusRowVectorXf operator * (const float b) {
            OctopusRowVectorXf r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_row_vector_value();
            return r;
        }
        OctopusRowVectorXf operator / (const float b) {
            OctopusRowVectorXf r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_row_vector_value();
            return r;
        }        
    };

    struct OctopusRowVectorXd: public RowVector
    {
        OctopusRowVectorXd() : RowVector() {}
        OctopusRowVectorXd(const RowVector & v) : RowVector(v) {} 
        OctopusRowVectorXd(size_t i) : RowVector(i) {}

        using RowVector::operator =;
        using RowVector::operator ();
        using RowVector::insert;
        using RowVector::append;
        using RowVector::fill;
        using RowVector::extract;
        using RowVector::extract_n;
        using RowVector::transpose;
        using RowVector::size;
        using RowVector::min;
        using RowVector::max;
        using RowVector::resize;
        using RowVector::clear;

        #ifdef SWIG
        %extend {
            void insert(const OctopusRowVectorXd& v, size_t i) {
                $self->insert(v,i);
            }
            void append(const OctopusRowVectorXd& v) {
                $self->append(v);
            }
            void fill(double value) {
                $self->fill(value);
            }
            void fill(double value, size_t c1, size_t c2) {
                $self->fill(value,c1,c2);
            }
            /*
            OctopusColVectorXd transpose() {
                return OctopusColVectorXd(this->transpose());
            }
            */
            
            OctopusRowVectorXd extract(size_t c1, size_t c2) { return $self->extract(c1,c2); }
            OctopusRowVectorXd extract_n(size_t c1, size_t n) { return $self->extract(c1,n); }

            double min() const { return $self->min(); }
            double max() const { return $self->max(); }

            void resize(size_t n, const double& rfv=0) { $self->resize(n,rfv); }
            void clear(size_t n) { $self->clear(n); }
        }
        #endif

        OctopusRowVectorXd operator + (const OctopusRowVectorXd & b) {
            OctopusRowVectorXd r;
            ValueList l;
            l(0) = "add";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).row_vector_value();
            return r;
        }
        OctopusRowVectorXd operator - (const OctopusRowVectorXd & b) {
            OctopusRowVectorXd r;
            ValueList l;
            l(0) = "sub";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).row_vector_value();
            return r;
        }
        OctopusRowVectorXd operator * (const OctopusRowVectorXd & b) {
            OctopusRowVectorXd r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).row_vector_value();
            return r;
        }
        OctopusRowVectorXd operator / (const OctopusRowVectorXd & b) {
            OctopusRowVectorXd r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).row_vector_value();
            return r;
        }
        OctopusRowVectorXd operator + (const double b) {
            OctopusRowVectorXd r;
            ValueList l;
            l(0) = "plus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).row_vector_value();
            return r;
        }
        OctopusRowVectorXd operator - (const double b) {
            OctopusRowVectorXd r;
            ValueList l;
            l(0) = "minus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).row_vector_value();
            return r;
        }
        OctopusRowVectorXd operator * (const double b) {
            OctopusRowVectorXd r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).row_vector_value();
            return r;
        }
        OctopusRowVectorXd operator / (const double b) {
            OctopusRowVectorXd r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).row_vector_value();
            return r;
        }        
    };
    struct OctopusRowVectorXcf : public FloatComplexRowVector
    {
        OctopusRowVectorXcf() = default;
        OctopusRowVectorXcf(const FloatComplexRowVector & v) : FloatComplexRowVector(v) {}
        OctopusRowVectorXcf(size_t i) : FloatComplexRowVector(i) {}

        using FloatComplexRowVector::operator =;
        using FloatComplexRowVector::operator ();
        using FloatComplexRowVector::insert;
        using FloatComplexRowVector::append;
        using FloatComplexRowVector::fill;
        using FloatComplexRowVector::extract;
        using FloatComplexRowVector::extract_n;
        using FloatComplexRowVector::transpose;
        using FloatComplexRowVector::size;
        using FloatComplexRowVector::min;
        using FloatComplexRowVector::max;
        using FloatComplexRowVector::resize;
        using FloatComplexRowVector::clear;

        #ifdef SWIG
        %extend {
            void insert(const OctopusRowVectorXcf& v, size_t i) {
                $self->insert(v,i);
            }
            void append(const OctopusRowVectorXcf& v) {
                $self->append(v);
            }
            void fill(double value) {
                $self->fill(value);
            }
            void fill(double value, size_t c1, size_t c2) {
                $self->fill(value,c1,c2);
            }
            //OctopusColVectorXcf transpose();
            
            OctopusRowVectorXcf extract(size_t c1, size_t c2) { return $self->extract(c1,c2); }
            OctopusRowVectorXcf extract_n(size_t c1, size_t n) { return $self->extract(c1,n); }

            double min() const { return $self->min(); }
            double max() const { return $self->max(); }

            void resize(size_t n, const double& rfv=0) { $self->resize(n,rfv); }
            void clear(size_t n) { $self->clear(n); }
        }
        #endif

        OctopusRowVectorXcf operator + (const OctopusRowVectorXcf & b) {
            OctopusRowVectorXcf r;
            ValueList l;
            l(0) = "add";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcf operator - (const OctopusRowVectorXcf & b) {
            OctopusRowVectorXcf r;
            ValueList l;
            l(0) = "sub";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcf operator * (const OctopusRowVectorXcf & b) {
            OctopusRowVectorXcf r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcf operator / (const OctopusRowVectorXcf & b) {
            OctopusRowVectorXcf r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcf operator + (const std::complex<float> b) {
            OctopusRowVectorXcf r;
            ValueList l;
            l(0) = "plus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcf operator - (const std::complex<float> b) {
            OctopusRowVectorXcf r;
            ValueList l;
            l(0) = "minus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcf operator * (const std::complex<float> b) {
            OctopusRowVectorXcf r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcf operator / (const std::complex<float> b) {
            OctopusRowVectorXcf r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).float_complex_row_vector_value();
            return r;
        }        
    };

    struct OctopusRowVectorXcd: public ComplexRowVector
    {
        OctopusRowVectorXcd() =default;
        OctopusRowVectorXcd(const ComplexRowVector & v) : ComplexRowVector(v) {}   
        OctopusRowVectorXcd(size_t i) : ComplexRowVector(i) {}   

        using ComplexRowVector::operator =;
        using ComplexRowVector::operator ();
        using ComplexRowVector::insert;
        using ComplexRowVector::append;
        using ComplexRowVector::fill;
        using ComplexRowVector::extract;
        using ComplexRowVector::extract_n;
        using ComplexRowVector::transpose;
        using ComplexRowVector::size;
        using ComplexRowVector::min;
        using ComplexRowVector::max;
        using ComplexRowVector::resize;
        using ComplexRowVector::clear;

        #ifdef SWIG
        %extend {
            void insert(const OctopusRowVectorXcd& v, size_t i) {
                $self->insert(v,i);
            }
            void append(const OctopusRowVectorXcd& v) {
                $self->append(v);
            }
            void fill(double value) {
                $self->fill(value);
            }
            void fill(double value, size_t c1, size_t c2) {
                $self->fill(value,c1,c2);
            }
            //OctopusColVectorXcd transpose();
            
            OctopusRowVectorXcd extract(size_t c1, size_t c2) { return $self->extract(c1,c2); }
            OctopusRowVectorXcd extract_n(size_t c1, size_t n) { return $self->extract(c1,n); }

            double min() const { return $self->min(); }
            double max() const { return $self->max(); }

            void resize(size_t n, const double& rfv=0) { $self->resize(n,rfv); }
            void clear(size_t n) { $self->clear(n); }
        }
        #endif

        OctopusRowVectorXcd operator + (const OctopusRowVectorXcd & b) {
            OctopusRowVectorXcd r;
            ValueList l;
            l(0) = "add";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcd operator - (const OctopusRowVectorXcd & b) {
            OctopusRowVectorXcd r;
            ValueList l;
            l(0) = "sub";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcd operator * (const OctopusRowVectorXcd & b) {
            OctopusRowVectorXcd r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcd operator / (const OctopusRowVectorXcd & b) {
            OctopusRowVectorXcd r;
            ValueList l;
            l(0) = "div";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcd operator + (const std::complex<float> b) {
            OctopusRowVectorXcd r;
            ValueList l;
            l(0) = "plus";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcd operator - (const std::complex<float> b) {
            OctopusRowVectorXcd r;
            ValueList l;
            l(0) = "divide";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcd operator * (const std::complex<float> b) {
            OctopusRowVectorXcd r;
            ValueList l;
            l(0) = "times";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_row_vector_value();
            return r;
        }
        OctopusRowVectorXcd operator / (const std::complex<float> b) {
            OctopusRowVectorXcd r;
            ValueList l;
            l(0) = "div";
            l(1) = *this;
            l(2) = b;
            l = octave::feval("cwiseops",l,1);
            r = l(0).complex_row_vector_value();
            return r;
        }   
    };
}