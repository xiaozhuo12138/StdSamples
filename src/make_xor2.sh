clear
echo 'nvcc'
nvcc -use_fast_math -arch=sm_61 -gencode=arch=compute_61,code=sm_61 --compiler-options '-fopenmp -pthread -O3 -march=native -mavx2 -mfma -fPIC -fmax-errors=1' -o xor2 xor2.cpp viper_vector.o viper_matrix.o viper_complex_vector.o viper_complex_matrix.o -lgomp -lstdc++ -lcublas -lcudart -lcurand

