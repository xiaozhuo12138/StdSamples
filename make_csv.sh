swig -Iinclude/Csv -lua -c++ Core/core_csv.i
gcc -Iinclude/Csv -O2 -fPIC -shared -o csv.so Core/core_csv_wrap.cxx -lstdc++ -lm -lluajit
