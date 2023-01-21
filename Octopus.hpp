
#include <iostream>
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/interpreter.h>
#include <Eigen/Core>

#include "carlo_mkl.hpp"
#include "carlo_samples.hpp"
//#include "carlo_sampledsp.hpp"
//#include "carlo_samplevector.hpp"

using ArrayXf = Array<float>;
using ArrayXd = Array<double>;
using ArrayXcf = Array<std::complex<float>>;
using ArrayXcd = Array<std::complex<double>>;
using VectorXf = FloatRowVector;
using VectorXd = RowVector;
using VectorXcf= FloatComplexRowVector;
using VectorXcd= ComplexRowVector;
using ColVectorXf = FloatColumnVector;
using ColVectorXd = ColumnVector;
using ColVectorXcf= FloatComplexColumnVector;
using ColVectorXcd= ComplexColumnVector;
using MatrixXf = FloatMatrix;
using MatrixXd = Matrix;
using MatrixXcf= FloatComplexMatrix;
using MatrixXcd= ComplexMatrix;
using Value=octave_value;
using ValueList=octave_value_list;


namespace Octave
{    
    struct Application : public octave::application
    {
        Application() {
        forced_interactive(true);
        }
        int execute() {
        return 0;
        }
    };

    struct Octopus
    {   
        octave::interpreter *interpreter;
        Application pita;    

        Octopus() {                  
            interpreter = new octave::interpreter();
            interpreter->interactive(false);
            interpreter->initialize_history(false);       
            interpreter->initialize();            
            interpreter->execute();
            std::string path = ".";
            octave_value_list p;
            p(0) = path;
            octave_value_list o1 = interpreter->feval("addpath", p, 1);            
            run_script("startup.m");
        }
        ~Octopus()
        {
            if(interpreter) delete interpreter;
        }
        
        void run_script(const std::string& s) {
            octave::source_file(s);
        }
        
        ValueList eval_string(std::string func, bool silent=false, int noutputs=1)
        {          
            octave_value_list out =interpreter->eval_string(func.c_str(), silent, noutputs);
            return out;
        }
        ValueList eval(std::string func, ValueList inputs, int noutputs=1)
        {          
            octave_value_list out =interpreter->feval(func.c_str(), inputs, noutputs);
            return out;
        }

        ValueList operator()(std::string func, ValueList inputs, int noutputs=1)
        {
            return eval(func,inputs,noutputs);
        }
        
            void createVar(const std::string& name, const Value& v, bool global=true)
            {
                interpreter->install_variable(name,v,global);
            }        
            Value getGlobalVar(const std::string& name)
            {    
                return interpreter->global_varval(name);
            }
            void setGlobalVar(const std::string& name, const Value& v)
            {
                interpreter->set_global_value(name,v);
            }
            Value getVarVal(const std::string& name) {
                return interpreter->varval(name);
            }
            void assign(const std::string& name, const Value& v) {
                interpreter->assign(name,v);
            }

    };

    struct OctopusVar
    {
        Octopus& interp;
        std::string var;
        bool _global;
        OctopusVar(Octopus& i, const std::string& name, const Value & v, bool global=true)
        : interp(i),var(name),_global(global)
        {
            interp.createVar(var,v,global);
        }

        Value getValue() { 
            if(_global) return interp.getGlobalVar(var); 
            return interp.getVarVal(var);
        }
        void setValue(const Value& v) { 
            if(_global) interp.setGlobalVar(var,v); 
            else interp.assign(var,v);    
        }
    };

    /*
    struct OctopusArrayXf : public Array<float>
    {
        OctopusArrayXf(const Array<float> & a) : Array<float>(a) {}
        OctopusArrayXf(size_t i) : Array<float>(i) {}
    };
    struct OctopusArrayXd : public Array<double>
    {
        OctopusArrayXd(const Array<double> & a) : Array<double>(a) {}
        OctopusArrayXd(size_t i) : Array<double>(i) {}
    };
    struct OctopusArrayXcf : public Array<std::complex<float>>
    {
        OctopusArrayXcf(const Array<std::complex<float>> & a) : Array<std::complex<float>>(a) {}
        OctopusArrayXcf(size_t i) : : Array<std::complex<float>>(i) {}
    };
    struct OctopusArrayXcd : public Array<std::complex<double>>
    {
        OctopusArrayXcd(const Array<std::complex<double>> & a) : Array<std::complex<double>>(a) {}
        OctopusArrayXcd(size_t i) : : Array<std::complex<double>>(i) {}
    };
    */
    struct OctopusColVectorXf;
    struct OctopusMatrixXf;
    struct OctopusColVectorXd;
    struct OctopusMatrixXd;
    struct OctopusColVectorXcf;
    struct OctopusMatrixXcf;
    struct OctopusColVectorXcd;
    struct OctopusMatrixXcd;

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

    OctopusRowVectorXf Octavate(const Casino::sample_vector<float> & m)
    {
        OctopusRowVectorXf r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusRowVectorXd Octavate(const Casino::sample_vector<double> & m)
    {
        OctopusRowVectorXd r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusRowVectorXcf Octavate(const Casino::complex_vector<float> & m)
    {
        OctopusRowVectorXcf r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusRowVectorXcd Octavate(const Casino::complex_vector<double> & m)
    {
        OctopusRowVectorXcd r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }  

    OctopusColVectorXf Octavate(const Casino::sample_vector<float> & m,bool x)
    {
        OctopusColVectorXf r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusColVectorXd Octavate(const Casino::sample_vector<double> & m,bool x)
    {
        OctopusColVectorXd r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusColVectorXcf Octavate(const Casino::complex_vector<float> & m,bool x)
    {
        OctopusColVectorXcf r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }
    OctopusColVectorXcd Octavate(const Casino::complex_vector<double> & m,bool x)
    {
        OctopusColVectorXcd r(m.size());
        for(size_t i = 0; i < m.size(); i++)        
                r(i) = m(i);
        return r;
    }  

    OctopusMatrixXf Octavate(const Casino::sample_matrix<float> & m)
    {
        OctopusMatrixXf r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }
    OctopusMatrixXd Octavate(const Casino::sample_matrix<double> & m)
    {
        OctopusMatrixXd r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }
    OctopusMatrixXcf Octavate(const Casino::complex_matrix<float> & m)
    {
        OctopusMatrixXcf r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }
    OctopusMatrixXcd Octavate(const Casino::complex_matrix<double> & m)
    {
        OctopusMatrixXcd r(m.rows(),m.cols());
        for(size_t i = 0; i < m.rows(); i++)
            for(size_t j = 0; j < m.cols(); j++)
                r(i,j) = m(i,j);
        return r;
    }   

    struct OctopusValue : public octave_value
    {
        OctopusValue() = default;
        OctopusValue(const octave_value & v) : octave_value(v) {}
        
        OctopusValue(double v) : octave_value(v) {}

        OctopusValue(const ArrayXf& v) : octave_value(v) {}
        OctopusValue(const ArrayXd& v) : octave_value(v) {}
        OctopusValue(const ArrayXcf& v) : octave_value(v) {}
        OctopusValue(const ArrayXcd& v) : octave_value(v) {}

        OctopusValue(const VectorXf& v) : octave_value(v) {}
        OctopusValue(const VectorXd& v) : octave_value(v) {}
        OctopusValue(const VectorXcf& v) : octave_value(v) {}
        OctopusValue(const VectorXcd& v) : octave_value(v) {}

        OctopusValue(const ColVectorXf& v) : octave_value(v) {}
        OctopusValue(const ColVectorXd& v) : octave_value(v) {}
        OctopusValue(const ColVectorXcf& v) : octave_value(v) {}
        OctopusValue(const ColVectorXcd& v) : octave_value(v) {}

        OctopusValue(const MatrixXf& v) : octave_value(v) {}
        OctopusValue(const MatrixXd& v) : octave_value(v) {}
        OctopusValue(const MatrixXcf& v) : octave_value(v) {}
        OctopusValue(const MatrixXcd& v) : octave_value(v) {}
        
        
        OctopusValue(const Casino::sample_vector<float>& v) : octave_value(Octavate(v)) {}
        OctopusValue(const Casino::sample_vector<double>& v) : octave_value(Octavate(v)) {}
        OctopusValue(const Casino::complex_vector<float>& v) : octave_value(Octavate(v)) {}
        OctopusValue(const Casino::complex_vector<double>& v) : octave_value(Octavate(v)) {}

        OctopusValue(const Casino::sample_matrix<float>& v) : octave_value(Octavate(v)) {}
        OctopusValue(const Casino::sample_matrix<double>& v) : octave_value(Octavate(v)) {}
        OctopusValue(const Casino::complex_matrix<float>& v) : octave_value(Octavate(v)) {}
        OctopusValue(const Casino::complex_matrix<double>& v) : octave_value(Octavate(v)) {}
        

        double getScalarValue() {
            return this->scalar_value();
        }
        
        OctopusRowVectorXf getFloatRowVector() {
            return OctopusRowVectorXf(this->float_row_vector_value());
        }
        OctopusColVectorXf getFloatColVector() {
            return OctopusColVectorXf(this->float_column_vector_value());
        }
        OctopusRowVectorXcf getFloatComplexRowVector() {
            return OctopusRowVectorXcf(this->float_complex_row_vector_value());
        }
        OctopusColVectorXcf getFloatComplexColVector() {
            return OctopusColVectorXcf(this->float_complex_column_vector_value());
        }
        OctopusRowVectorXd getRowVector() {
            return OctopusRowVectorXd(this->row_vector_value());
        }
        OctopusColVectorXd getColVector() {
            return OctopusColVectorXd(this->column_vector_value());
        }
        OctopusRowVectorXcd getComplexRowVector() {
            return OctopusRowVectorXcd(this->complex_row_vector_value());
        }
        OctopusColVectorXcd getComplexColVector() {
            return OctopusColVectorXcd(this->complex_column_vector_value());
        }

        OctopusMatrixXf getFloatMatrix() {
            return OctopusMatrixXf(this->float_matrix_value());
        }
        OctopusMatrixXd getMatrix() {
            return OctopusMatrixXd(this->matrix_value());
        }
        OctopusMatrixXcf getFloatComplexMatrix() {
            return OctopusMatrixXcf(this->float_complex_matrix_value());
        }
        OctopusMatrixXcd getComplexMatrix() {
            return OctopusMatrixXcd(this->complex_matrix_value());
        }

        Casino::sample_matrix<float> getFloatSamplesMatrix() {
            Casino::sample_matrix<float> dst;
            MatrixXf src = this->float_matrix_value();
            dst.resize((size_t)src.rows(),(size_t)src.cols());
            for(size_t i = 0; i < src.rows(); i++)
                for(size_t j = 0; j < src.cols(); j++)
                    dst(i,j) = src(i,j);
            return dst;
        }
        Casino::sample_matrix<double> getDoubleSamplesMatrix() {
            Casino::sample_matrix<double> dst;
            MatrixXd src = this->matrix_value();
            dst.resize((size_t)src.rows(),(size_t)src.cols());
            for(size_t i = 0; i < src.rows(); i++)
                for(size_t j = 0; j < src.cols(); j++)
                    dst(i,j) = src(i,j);
            return dst;
        }
        Casino::complex_matrix<float> getComplexFloatSamplesMatrix() {
            Casino::complex_matrix<float> dst;
            MatrixXcf src = this->float_complex_matrix_value();
            for(size_t i = 0; i < src.rows(); i++)
                for(size_t j = 0; j < src.cols(); j++)
                    dst(i,j) = src(i,j);
            return dst;
        }
        Casino::complex_matrix<double> getComplexDoubleSamplesMatrix() {
            Casino::complex_matrix<double> dst;
            MatrixXcd src = this->matrix_value();
            for(size_t i = 0; i < src.rows(); i++)
                for(size_t j = 0; j < src.cols(); j++)
                    dst(i,j) = src(i,j);
            return dst;
        }

        Casino::sample_vector<float> getFloatSamplesRowVector() {
            Casino::sample_vector<float> dst;
            VectorXf src = this->float_row_vector_value();
            dst.resize(src.size(1));
            for(size_t j = 0; j < src.cols(); j++) dst[j] = src(j);
            return dst;
        }
        Casino::sample_vector<double> getDoubleSamplesRowVector() {
            Casino::sample_vector<double> dst;
            VectorXd src = this->row_vector_value();
            dst.resize(src.size(1));
            for(size_t j = 0; j < src.cols(); j++) dst[j] = src(j);
            return dst;
        }
        Casino::complex_vector<float> getComplexFloatSamplesRowVector() {
            Casino::complex_vector<float> dst;
            VectorXcf src = this->float_complex_row_vector_value();
            dst.resize(src.size(1));
            for(size_t j = 0; j < src.cols(); j++) dst[j] = src(j);
            return dst;
        }
        Casino::complex_vector<double> getComplexDoubleSamplesRowVector() {
            Casino::complex_vector<double> dst;
            VectorXcd src = this->complex_row_vector_value();
            dst.resize(src.size(1));
            for(size_t j = 0; j < src.cols(); j++) dst[j] = src(j);
            return dst;
        }

        Casino::sample_vector<float> getFloatSamplesColVector() {
            Casino::sample_vector<float> dst;
            ColVectorXf src = this->float_column_vector_value();
            dst.resize(src.size(1));
            for(size_t j = 0; j < src.rows(); j++) dst[j] = src(j);
            return dst;
        }
        Casino::sample_vector<double> getDoubleSamplesColVector() {
            Casino::sample_vector<double> dst;
            ColVectorXd src = this->column_vector_value();
            dst.resize(src.size(1));
            for(size_t j = 0; j < src.cols(); j++) dst[j] = src(j);
            return dst;
        }
        Casino::complex_vector<float> getComplexFloatSamplesColVector() {
            Casino::complex_vector<float> dst;
            ColVectorXcf src = this->float_complex_column_vector_value();
            dst.resize(src.size(1));
            for(size_t j = 0; j < src.cols(); j++) dst[j] = src(j);
            return dst;
        }
        Casino::complex_vector<double> getComplexDoubleSamplesColVector() {
            Casino::complex_vector<double> dst;
            ColVectorXcd src = this->complex_column_vector_value();
            dst.resize(src.size(1));
            for(size_t j = 0; j < src.cols(); j++) dst[j] = src(j);
            return dst;
        }
    };

    
    
    struct OctopusValueList : public octave_value_list
    {
        OctopusValueList() = default;
        OctopusValueList(const octave_value_list& v) : octave_value_list(v) {}

        OctopusValue get(size_t i) {
            return OctopusValue((*this)(i));
        }
        void set(size_t i, const OctopusValue & v) {
            (*this)(i) = v;
        }
        void set(size_t i, const Value & v) {
            (*this)(i) = OctopusValue(v);
        }
        
        using octave_value_list::operator ();
        
        OctopusValue __getitem__(size_t i) { return get(i); }

        void __setitem__(size_t i, const OctopusValue& v) { set(i,v); }

        void __setitem__(size_t i, const double& v) { set(i,Value(v)); }

        void __setitem__(size_t i, const ArrayXf& v) { set(i,Value(v)); }
        void __setitem__(size_t i, const ArrayXd& v) { set(i,Value(v)); }
        void __setitem__(size_t i, const ArrayXcf& v) { set(i,Value(v)); }
        void __setitem__(size_t i, const ArrayXcd& v) { set(i,Value(v)); }
        
        void __setitem__(size_t i, const VectorXf& v) { set(i,Value(v)); }
        void __setitem__(size_t i, const VectorXd& v) { set(i,Value(v)); }
        void __setitem__(size_t i, const VectorXcf& v) { set(i,Value(v)); }
        void __setitem__(size_t i, const VectorXcd& v) { set(i,Value(v)); }
        
        void __setitem__(size_t i, const ColVectorXf& v) { set(i,Value(v)); }
        void __setitem__(size_t i, const ColVectorXd& v) { set(i,Value(v)); }
        void __setitem__(size_t i, const ColVectorXcf& v) { set(i,Value(v)); }
        void __setitem__(size_t i, const ColVectorXcd& v) { set(i,Value(v)); }

        void __setitem__(size_t i, const MatrixXf& v) { set(i,Value(v)); }
        void __setitem__(size_t i, const MatrixXd& v) { set(i,Value(v)); }
        void __setitem__(size_t i, const MatrixXcf& v) { set(i,Value(v)); }
        void __setitem__(size_t i, const MatrixXcd& v) { set(i,Value(v)); }
        
        
        void __setitem__(size_t i, const Casino::sample_vector<float>& v) { set(i,Value(Octavate(v))); }
        void __setitem__(size_t i, const Casino::sample_vector<double>& v) { set(i,Value(Octavate(v))); }
        void __setitem__(size_t i, const Casino::complex_vector<float>& v) { set(i,Value(Octavate(v))); }
        void __setitem__(size_t i, const Casino::complex_vector<double>& v) { set(i,Value(Octavate(v))); }

        void __setitem__(size_t i, const Casino::sample_matrix<float>& v) { set(i,Value(Octavate(v))); }
        void __setitem__(size_t i, const Casino::sample_matrix<double>& v) { set(i,Value(Octavate(v))); }
        void __setitem__(size_t i, const Casino::complex_matrix<float> & v) { set(i,Value(Octavate(v))); }
        void __setitem__(size_t i, const Casino::complex_matrix<double>& v) { set(i,Value(Octavate(v))); }        
    };

    struct OctopusFunction
    {
        std::string code;
        std::string name;
        Octopus& interp;
        
        OctopusFunction(Octopus& i, const std::string& n, const std::string& c)
        : interp(i),code(c),name(n) 
        {                        
            interp.interpreter->eval(code,0);
        }
        OctopusValueList operator()(OctopusValueList & inputs, int numOut)
        {
            return interp.eval(name.c_str(),inputs,numOut);
        }
    };


    struct Function
    {
        std::string name;
        

        Function(const std::string& f) : name(f) {}

        ValueList operator()()
        {
            ValueList input;
            int num_outputs=0;
            return octave::feval(name.c_str(),input,num_outputs);
        }

        ValueList operator()(ValueList & input, int num_outputs=1)
        {
            return octave::feval(name.c_str(),input,num_outputs);
        }
        
        OctopusValueList eval(const OctopusValueList &input, int numOutputs=1)
        {
            return OctopusValueList(octave::feval(name.c_str(),input,numOutputs));
        }
        
    };

    #define def(x) Function x(#x)  

    def(fft);
    def(ifft);
    def(fft2);
    def(ifft2);
    def(fftconv);
    def(fftfilt);
    def(fftn);
    def(fftshift);
    def(fftw);
    def(ifftn);
    def(ifftshift);
    def(ifht);
    def(ifourier);
    def(ifwht);
    def(ifwt);
    def(ifwt2);
    def(buffer);
    def(chirp);
    def(cmorwavf);  
    def(gauspuls);
    def(gmonopuls);
    def(mexihat);
    def(meyeraux);  
    def(morlet);
    def(pulstran);
    def(rectpuls);
    def(sawtooth);
    def(shanwavf);
    def(shiftdata);
    def(sigmoid_train);
    def(specgram);
    def(square);
    def(tripuls);
    def(udecode);
    def(uencoder);
    def(unshiftdata);
    def(findpeaks);
    def(peak2peak);
    def(peak2rms);
    def(rms);
    def(rssq);
    def(cconv);
    def(convmtx);  
    def(wconv);
    def(xcorr);
    def(xcorr2);
    def(xcov);
    def(filtfilt);
    def(fltic);
    def(medfilt1);
    def(movingrms);
    def(sgolayfilt);
    def(sosfilt);
    def(freqs);
    def(freqs_plot);
    def(freqz);
    def(freqz_plot);
    def(impz);
    def(zplane);
    def(filter);
    def(filter2);  
    def(fir1);
    def(fir2);
    def(firls);
    def(sinc);
    def(unwrap);

    def(bartlett);
    def(blackman);  
    def(blackmanharris);
    def(blackmannuttal);
    def(dftmtx);
    def(hamming);
    def(hann);
    def(hanning);
    def(pchip);
    def(periodogram);
    def(sinetone);
    def(sinewave);
    def(spectral_adf);
    def(spectral_xdf);
    def(spencer);
    def(stft);
    def(synthesis);
    def(yulewalker);
    def(polystab);
    def(residued);
    def(residuez);
    def(sos2ss);
    def(sos2tf);
    def(sos2zp);
    def(ss2tf);
    def(ss2zp);
    def(tf2sos);
    def(tf2ss);
    def(tf2zp);
    def(zp2sos);
    def(zp2ss);
    def(zp2tf);
    def(besselap);
    def(besself);
    def(bilinear);
    def(buttap);
    def(butter);
    def(buttord);
    def(cheb);
    def(cheb1ap);
    def(cheb1ord);
    def(cheb2ap);
    def(cheb2ord);
    def(chebywin);
    def(cheby1);
    def(cheby2);
    def(ellip);
    def(ellipap);  
    def(ellipord);
    def(impinvar);
    def(ncauer);
    def(pei_tseng_notch);
    def(sftrans);
    def(cl2bp);
    def(kaiserord);
    def(qp_kaiser);
    def(remez);
    def(sgplay);
    def(bitrevorder);
    def(cceps);
    def(cplxreal);
    def(czt);
    def(dct);
    def(dct2);  
    def(dctmtx);
    def(digitrevorder);
    def(dst);
    def(dwt);
    def(rceps);
    def(ar_psd);
    def(cohere);
    def(cpsd);
    def(csd);
    def(db2pow);
    def(mscohere);
    def(pburg);
    def(pow2db);
    def(pwelch);
    def(pyulear);
    def(tfe);
    def(tfestimate);
    def(__power);
    def(barthannwin);
    def(bohmanwin);
    def(boxcar);
    def(flattopwin);
    def(chebwin);
    def(gaussian);
    def(gausswin);
    def(kaiser);  
    def(nuttalwin);
    def(parzenwin);
    def(rectwin);
    def(tukeywin);
    def(ultrwin);
    def(welchwin);
    def(window);
    def(arburg);
    def(aryule);
    def(invfreq);
    def(invfreqz);
    def(invfreqs);
    def(levinson);
    def(data2fun);
    def(decimate);
    //def(interp);
    def(resample);
    def(upfirdn);
    def(upsample);
    def(clustersegment);
    def(fracshift);
    def(marcumq);
    def(primitive);
    def(sampled2continuous);
    def(schtrig);
    def(upsamplefill);
    def(wkeep);
    def(wrev);
    def(zerocrossing);


    def(fht);
    def(fwht);  
    def(hilbert);
    def(idct);
    def(idct2);

    def(max);
    def(mean);
    def(meansq);
    def(median);
    def(min);

    def(plot);
    def(pause);

    def(abs);
    def(accumarray);
    def(accumdim);
    def(acos);
    def(acosd);
    def(acosh);
    def(acot);
    def(acotd);
    def(acoth);
    def(acsc);
    def(acsch);
    def(acscd);
    def(airy);
    def(adjoint);
    def(all);
    def(allow_non_integer_range_as_index);
    def(amd);
    def(ancestor);
    //def(and);
    def(angle);
    def(annotation);
    def(anova);
    def(ans);
    def(any);    
    def(arch_fit);
    def(arch_rnd);
    def(arch_test);
    def(area);
    def(arg);
    def(arrayfun);  
    def(asec);
    def(asecd);
    def(asech);
    def(asin);
    def(asind);
    def(asinh);
    def(assume);
    def(assumptions);
    def(atan);
    def(atand);
    def(atanh);
    def(atan2);
    def(audiodevinfo);
    def(audioformats);
    def(audioinfo);
    def(audioread);
    def(audiowrite);
    def(autoreg_matrix);
    def(autumn);
    def(axes);
    def(axis);
    def(balance);
    def(bandwidth);


    def(bar);
    def(barh);
    def(bathannwin);
    def(bartlett_test);
    def(base2dec);
    def(base64_decode);
    def(base64_encode);
    def(beep);
    def(beep_on_error);
    def(bernoulli);  
    def(besseli);
    def(besseljn);
    def(besselk);
    def(bessely);
    def(beta);
    def(betacdf);
    def(betainc);
    def(betaincinv);
    def(betainv);
    def(betain);
    def(betapdf);
    def(betarnd);
    def(bicg);
    def(bicgstab);  
    def(bin2dec);
    def(bincoeff);
    def(binocdf);
    def(binoinv);
    def(binopdf);
    def(binornd);
    //def(bitand);
    def(bitcmp);
    def(bitget);
    //def(bitor);
    def(bitpack);  
    def(bitset);
    def(bitshift);
    def(bitunpack);
    def(bitxor);
    def(blanks);
    def(blkdiag);
    def(blkmm);
    def(bone);
    def(box);  
    def(brighten);
    def(bsxfun);
    def(builtin);
    def(bzip2);

    def(calendar);
    def(camlight);
    def(cart2pol);
    def(cart2sph);
    def(cast);
    def(cat);
    def(catalan);
    def(cauchy);
    def(cauchy_cdf);
    def(cauchy_inv);
    def(cauchy_pdf);
    def(cauchy_rnd);
    def(caxis);
    def(cbrt);  
    def(ccode);
    def(ccolamd);  
    def(ceil);
    def(center);
    def(centroid);
    def(cgs);  
    def(chi2cdf);
    def(chi2inv);
    def(chi2pdf);
    def(chi2rnd);
    def(children);  
    def(chisquare_test_homogeneity);  
    def(chebyshevpoly);
    def(chebyshevT);
    def(chebyshevU);
    def(chol);
    def(chol2inv);
    def(choldelete);
    def(cholinsert);
    def(colinv);
    def(cholshift);
    def(cholupdate);
    def(chop);
    def(circshift);  
    def(cla);
    def(clabel);
    def(clc);
    def(clf);
    def(clock);
    def(cloglog);  
    def(cmpermute);
    def(cmunique);
    def(coeffs);  
    def(colamd);
    def(colloc);
    def(colon);
    def(colorbar);
    def(colorcube);
    def(colormap);
    def(colperm);
    def(columns);
    def(comet);
    def(compan);
    def(compass);
    def(complex);
    def(computer);
    def(cond);
    def(condeig);
    def(condest);
    def(conj);
    def(contour);
    def(contour3);
    def(contourc);
    def(contourf);
    def(contrast);
    def(conv);
    def(conv2);
    def(convhull);
    def(convhulln);  
    def(cool);
    def(copper);
    def(copyfile);
    def(copyobj);
    def(cor_test);
    def(cos);
    def(cosd);
    def(cosh);
    def(coshint);
    def(cosint);
    def(cot);
    def(cotd);
    def(coth);
    def(cov);
    def(cplxpair);    
    def(cputime);
    def(cross);
    def(csc);
    def(cscd);
    def(csch);  
    def(cstrcat);
    def(cstrcmp);
    def(csvread);
    def(csvwrite);
    def(csymamd);
    def(ctime);
    def(ctranspose);
    def(cubehelix);
    def(cummax);
    def(cummin);
    def(cumprod);
    def(cumsum);
    def(cumtrapz);
    def(cylinder);

    def(daspect);
    def(daspk);
    def(dasrt_options);
    def(dassl);
    def(dassl_options);  
    def(date);
    def(datenum);
    def(datestr);
    def(datetick);
    def(dawson);  
    def(dbclear);
    def(dbcont);
    def(dbdown);
    def(dblist);
    def(dblquad);
    def(dbquit);
    def(dbstack);
    def(dbstatus);
    def(dbstep);
    def(dbstop);
    def(dbtype);
    def(dbup);
    def(dbwhere);  
    def(deal);
    def(deblank);
    def(dec2base);
    def(dec2hex);  
    def(deconv);
    def(deg2rad);
    def(del2);
    def(delaunay);
    def(delaunayn);  
    def(det);
    def(detrend);  
    def(diag);
    def(diff);
    def(diffpara);
    def(diffuse);  
    def(digits);
    def(dilog);
    def(dir);
    def(dirac);  
    def(discrete_cdf);
    def(discrete_inv);
    def(discrete_pdf);
    def(discrete_rnd);
    def(disp);
    def(display);
    def(divergence);
    def(dimread);
    def(dimwrite);
    def(dmperm);
    def(do_string_escapes);
    def(doc);
    def(dot);
    //def(double);
    def(downsample);
    def(dsearch);
    def(dsearchn);
    def(dsolve);  
    def(dup2);
    def(duplication_matrix);
    def(durblevinson);


    def(e);
    def(ei);
    def(eig);
    def(ellipke);  
    def(ellipsoid);
    def(ellipticCE);
    def(ellipticCK);
    def(ellipticCPi);
    def(ellipticE);
    def(ellipticF);
    def(ellipticK);
    def(ellipticPi);
    def(empirical_cdf);
    def(empirical_inv);
    def(empirical_pdf);
    def(empirical_rnd);
    def(end);
    def(endgrent);
    def(endpwent);
    def(eomday);
    def(eps);
    def(eq);
    def(equationsToMatrix);
    def(erf);
    def(erfc);
    def(erfinv);
    def(erfi);
    //def(errno);
    def(error);
    def(error_ids);
    def(errorbar);
    def(etime);
    def(etree);
    def(etreeplot);
    def(eulier);
    def(eulergamma);
    def(evalin);
    def(exp);
    def(expand);
    def(expcdf);
    def(expint);
    def(expinv);
    def(expm);
    def(expm1);
    def(exppdf);
    def(exprnd);
    def(eye);
    def(ezcontour);
    def(ezcontourf);
    def(ezmesh);
    def(explot);
    def(ezplot3);
    def(ezsurf);
    def(ezpolar);
    def(ezsurfc);

    def(f_test_regression);
    def(factor);
    def(factorial);
    //def(false);
    def(fcdf);
    def(fclear);
    def(fcntl);
    def(fdisp);
    def(feather);
    def(ff2n);  
    def(fibonacci);  
    def(find);  
    def(findsym);
    def(finiteset);
    def(finv);
    def(fix);  
    def(flintmax);
    def(flip);
    def(flipir);
    def(flipud);
    def(floor);
    def(fminbnd);
    def(fminunc);
    def(formula);
    def(fortran);  
    def(fourier);
    def(fpdf);
    def(fplot);
    def(frac);
    def(fractdiff);
    def(frame2im);
    def(freport);  
    def(fresneic);
    def(frnd);
    def(fskipl);
    def(fsolve);
    def(full);
    def(fwhm);  
    def(fzero);

    def(gallery);
    def(gamcdf);
    def(gaminv);
    def(gamma);
    def(gammainc);
    def(gammaln);    
    def(gca);
    def(gcbf);
    def(gcbo);
    def(gcd);
    def(ge);
    def(geocdf);
    def(geoinv);
    def(geopdf);
    def(geornd);
    def(givens);
    def(glpk);  
    def(gmres);
    def(gmtime);
    def(gnplot_binary);
    def(gplot);
    def(gradient);
    def(gray);
    def(gray2ind);
    def(gt);
    def(gunzip);
    def(gzip);

    def(hadamard);  
    def(hankel);  
    def(harmonic);
    def(has);
    def(hash);
    def(heaviside);
    def(help);
    def(hess);
    def(hex2dec);
    def(hex2num);
    def(hilb);  
    def(hilbert_curve);
    def(hist);
    def(horner);
    def(horzcat);
    def(hot);
    def(housh);
    def(hsv2rgb);
    def(hurst);
    def(hygecdf);
    def(hygeinv);
    def(hygepdf);
    def(hygernd);
    def(hypergeom);
    def(hypot);

    def(I);
    def(ichol);  
    def(idist);
    def(idivide);  
    def(igamma);  
    def(ilaplace);
    def(ilu);
    def(im2double);
    def(im2frame);
    def(im2int16);
    def(im2single);
    def(im2uint16);
    def(im2uint8);
    def(imag);
    def(image);
    def(imagesc);
    def(imfinfo);
    def(imformats);
    def(importdata);  
    def(imread);
    def(imshow);
    def(imwrite);
    def(ind2gray);
    def(ind2rgb);
    def(int2sub);
    def(index);
    def(Inf);
    def(inpolygon);
    def(input);  
    def(interp1);
    def(interp2);
    def(interp3);
    def(intersect);
    def(intmin);
    def(inv);
    def(invhilb);
    def(inimpinvar);
    def(ipermute);
    def(iqr);
    def(isa);
    def(isequal);
    def(ishermitian);
    def(isprime);

    def(jit_enable);

    def(kbhit);
    def(kendall);
    def(kron);
    def(kurtosis);

    def(laplace);
    def(laplace_cdf);
    def(laplace_inv);
    def(laplace_pdf);
    def(laplace_rnd);
    def(laplacian);
    def(lcm);
    def(ldivide);
    def(le);
    def(legendre);
    def(length);
    def(lgamma);
    def(limit);
    def(line);
    def(linprog);
    def(linsolve);
    def(linspace);
    def(load);
    def(log);
    def(log10);
    def(log1p);
    def(log2);
    def(logical);
    def(logistic_cdf);
    def(logistic_inv);
    def(logistic_pdf);
    def(logistic_regression);
    def(logit);
    def(loglog);
    def(loglogerr);
    def(logm);
    def(logncdf);
    def(logninv);
    def(lognpdf);
    def(lognrnd);
    def(lognspace);
    def(lookup);
    def(lscov);
    def(lsode);
    def(lsqnonneg);
    def(lt);

    def(magic);
    def(manova);
    def(minus);
    def(mkpp);
    def(mldivide);
    def(mod);
    def(moment);    
    def(mpoles);
    def(mpower);
    def(mrdivide);
    def(mu2lin);

    def(NA);
    def(NaN);
    def(nextpow2);
    def(nnz);
    def(nonzeros);
    def(norm);
    def(normcdf);
    def(normest);
    def(normest1);
    def(norminv);
    def(normpdf);
    def(normrnd);
    def(nth_element);
    def(nth_root);
    def(null);
    def(numel);

    def(ode23);
    def(ode45);
    def(ols);
    def(ones);

    def(prod);
    def(power);

    def(sin);
    def(sqrt);
    def(sum);
    def(sumsq);

    def(tan);
    def(tanh);
    def(sinh);

    // image
    // fuzzy-logic-toolkit

    /*  
    // splines  
    namespace splines
    {  
    def(bin_values);
    def(catmullrom);  
    def(csape);
    def(csapi);
    def(csaps);
    def(csaps_sel);
    def(dedup);
    def(fnder);
    def(fnplt);
    def(fnval);
    def(regularization);
    def(regularization2D);
    def(tpaps);
    def(tps_val);
    def(tps_val_der);
    }
    */
    /* ltfat = not installed
    namspace ltfat {  
    def(rms);
    def(normalize);
    def(gaindb);
    def(crestfactor);
    def(uquant);
    def(firwin);
    def(firkaiser);
    def(fir2long);
    def(long2fir);
    def(freqwin);
    def(firfilter);
    def(blfilter);
    def(warpedblfilter);
    def(freqfilter);
    def(pfilt);
    def(magresp);
    def(transferfunction);
    def(pgrdelay);
    def(rampup);
    def(rampdown);
    def(thresh);
    def(largestr);
    def(largestn);
    def(dynlimit);
    def(groupthresh);
    def(rgb2jpeg);
    def(jpeg2rgb);
    def(qam4);
    def(iqam4);
    def(semiaudplot);
    def(audtofreq);
    def(freqtoaud);
    def(audspace);
    def(audspacebw);
    def(erbtofreq);
    def(freqtoerb);
    def(erbspace);
    */
    
};

Octave::OctopusMatrixXf cos(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::cos.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf sin(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::sin.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf tan(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::tan.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf acos(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::acos.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf asin(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::asin.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf atan(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::atan.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf cosh(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::cosh.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf sinh(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::sinh.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf tanh(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::tanh.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf acosh(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::acosh.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf asinh(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::asinh.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf atanh(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::atanh.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf exp(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::exp.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf log(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::log.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf log10(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::log10.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}
Octave::OctopusMatrixXf pow(const Octave::OctopusMatrixXf & a, double b)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l(1) = b;
    l = Octave::power.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}    
Octave::OctopusMatrixXf sqrt(const Octave::OctopusMatrixXf & a)
{
    Octave::OctopusValueList l;
    l(0) = a;
    l = Octave::sqrt.eval(l,1);  
    return Octave::OctopusMatrixXf(l(0).float_matrix_value());
}    