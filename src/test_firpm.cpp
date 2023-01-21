#include "DSP/dsp_firpm.hpp"

using pm::firpm;
using pm::firpmRS;
using pm::firpmAFP;
using pm::filter_t;


template<typename T>
void printInfo(pm::pmoutput_t<T>& output, double eps)
{
    if(output.q < eps)
    {
        std::cout << "Final delta     = " << output.delta << std::endl;
        std::cout << "Iteration count = " << output.iter << std::endl;
    }
    else
    {
        std::cout << "Iteration count = NC\n";
    }
}

template<typename T>
void compareInfoRS(pm::pmoutput_t<T>& output1, pm::pmoutput_t<T>& output2, double eps)
{
    if(output1.q < eps)
    {
        std::cout << "Iteration reduction for final filter  RS: " << 1.0 - (double)output2.iter / output1.iter << std::endl;
    }
}

template<typename T>
void compareInfoAFP(pm::pmoutput_t<T>& output1, pm::pmoutput_t<T>& output2, double eps)
{
    if(output1.q < eps)
    {
        std::cout << "Iteration reduction for final filter AFP: " << 1.0 - (double)output2.iter / output1.iter << std::endl;
    }
}

void firpm()
{
    #ifdef HAVE_MPFR
    mpfr::mpreal::set_default_prec(165ul);
    #endif

    std::cout << "START Parks-McClellan with reference scaling\n";
    auto output2 = firpmRS<double>(400, {0.0, 0.38, 0.45, 1.0}, {1.0, 1.0, 0.0, 0.0}, {1.0, 1.0});
    printInfo(output2, 1e-2);
    std::cout << "FINISH Parks-McClellan with reference scaling\n";
    assert(output2.q < 1e-2);

    std::cout << "START Parks-McClellan with AFP\n";
    auto output3 = firpmAFP<double>(400, {0.0, 0.38, 0.45, 1.0}, {1.0, 1.0, 0.0, 0.0}, {1.0, 1.0});
    printInfo(output3, 1e-2);
    std::cout << "FINISH Parks-McClellan with AFP\n";
    assert(output3.q < 1e-2);
    assert(pm::pmmath::fabs((output2.delta-output3.delta)/output2.delta) <= 2e-2);

    compareInfoAFP(output2, output3, 1e-2);
}
