swig -lua -c++ fir_filter.i
gcc -O2 -fPIC -march=native -mavx2 -shared -o fir_filter.so fir_filter_wrap.cxx fir_filter.cpp -lstdc++ -lm -lluajit
