swig -Iinclude/Csv -lua -c++ core_csv.i
gcc -Iinclude/Csv -O2 -fPIC -shared -o csv.so core_csv_wrap.cxx -lstdc++ -lm -lluajit
