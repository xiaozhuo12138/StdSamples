swig -lua -c++ -Iinclude Functions.i
gcc -Wfatal-errors -fmax-errors=1 -std=c++17 -I. -Iinclude \
-O2 -fPIC -mavx2 -mfma -march=native -shared \
-o Functions.so Functions_wrap.cxx  \
-lstdc++ -lm -lluajit
