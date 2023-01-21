swig -lua -c++ -Iinclude iir1.i
gcc -Iinclude -O2 -march=native -mavx2 -fPIC -shared -o iir1.so iir1_wrap.cxx -lstdc++ -lm -lluajit -Lbin -lfir
