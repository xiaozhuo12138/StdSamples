
#include <iostream>
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/interpreter.h>
#include <Eigen/Core>

//#include "carlo_mkl.hpp"
//#include "carlo_samples.hpp"

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

#include "octopus_rowvector.hpp"
#include "octopus_colvector.hpp"
#include "octopus_matrix.hpp"
#include "octopus_octavate.hpp"


namespace Octopus
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

    struct OctaveInterpreter
    {   
        octave::interpreter *interpreter;
        Application pita;    

        OctaveInterpreter() {                  
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
        ~OctaveInterpreter()
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
        OctaveInterpreter& interp;
        std::string var;
        bool _global;
        OctopusVar(OctaveInterpreter& i, const std::string& name, const Value & v, bool global=true)
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
        
        /*        
        OctopusValue(const Casino::sample_vector<float>& v) : octave_value(Octavate(v)) {}
        OctopusValue(const Casino::sample_vector<double>& v) : octave_value(Octavate(v)) {}
        OctopusValue(const Casino::complex_vector<float>& v) : octave_value(Octavate(v)) {}
        OctopusValue(const Casino::complex_vector<double>& v) : octave_value(Octavate(v)) {}

        OctopusValue(const Casino::sample_matrix<float>& v) : octave_value(Octavate(v)) {}
        OctopusValue(const Casino::sample_matrix<double>& v) : octave_value(Octavate(v)) {}
        OctopusValue(const Casino::complex_matrix<float>& v) : octave_value(Octavate(v)) {}
        OctopusValue(const Casino::complex_matrix<double>& v) : octave_value(Octavate(v)) {}
        */

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
        
        /*
        void __setitem__(size_t i, const Casino::sample_vector<float>& v) { set(i,Value(Octavate(v))); }
        void __setitem__(size_t i, const Casino::sample_vector<double>& v) { set(i,Value(Octavate(v))); }
        void __setitem__(size_t i, const Casino::complex_vector<float>& v) { set(i,Value(Octavate(v))); }
        void __setitem__(size_t i, const Casino::complex_vector<double>& v) { set(i,Value(Octavate(v))); }

        void __setitem__(size_t i, const Casino::sample_matrix<float>& v) { set(i,Value(Octavate(v))); }
        void __setitem__(size_t i, const Casino::sample_matrix<double>& v) { set(i,Value(Octavate(v))); }
        void __setitem__(size_t i, const Casino::complex_matrix<float> & v) { set(i,Value(Octavate(v))); }
        void __setitem__(size_t i, const Casino::complex_matrix<double>& v) { set(i,Value(Octavate(v))); }        
        */
    };

    struct OctopusFunction
    {
        std::string code;
        std::string name;
        OctaveInterpreter& interp;
        
        OctopusFunction(OctaveInterpreter& i, const std::string& n, const std::string& c)
        : interp(i),code(c),name(n) 
        {                        
            interp.interpreter->eval(code,0);
        }
        OctopusValueList operator()(OctopusValueList & inputs, int numOut)
        {
            return interp.eval(name.c_str(),inputs,numOut);
        }
    };  


}

// this is just awkward for now

#include "octopus_functions.hpp"

namespace Octopus
{   
    OctopusMatrixXf cos(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::cos.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf sin(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::sin.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf tan(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::tan.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf acos(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::acos.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf asin(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::asin.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf atan(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::atan.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf cosh(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::cosh.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf sinh(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::sinh.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf tanh(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::tanh.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf acosh(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::acosh.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf asinh(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::asinh.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf atanh(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::atanh.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf exp(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::exp.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf log(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::log.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf log10(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::log10.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }
    OctopusMatrixXf pow(const OctopusMatrixXf & a, double b)
    {
        OctopusValueList l;
        l(0) = a;
        l(1) = b;
        l = Functions::power.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }    
    OctopusMatrixXf sqrt(const OctopusMatrixXf & a)
    {
        OctopusValueList l;
        l(0) = a;
        l = Functions::sqrt.eval(l,1);  
        return OctopusMatrixXf(l(0).float_matrix_value());
    }    
}
