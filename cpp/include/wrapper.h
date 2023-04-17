#ifndef WRAPPER_H
#define WRAPPER_H

#include <armadillo>
#include <string>
#include <assert.h>
#include "bmruntime_cpp.h"
#include "bmcv_api.h"

using namespace bmruntime;

arma::fmat as_strided(const arma::fmat& X, int n_rows, int n_cols, int row_stride, int col_stride);

// dim = 0, expand the number of rows; dim = 1, expand the number of columns
arma::fmat pad(const arma::fmat& X, int num, int dim);

// Generate a matrix of shape [size, 1] with elements increasing linearly from 0 to size-1 
arma::fmat arange(int size);

// Convert fmat to the contents pointed to by void*, noting that mat is stored by column
template<typename T>
void* mat_to_sys_mem(const arma::Mat<T>& X) {
    arma::Mat<T> trans_X = arma::trans(X);
    int n_rows = trans_X.n_rows;  // get the number of rows
    int n_cols = trans_X.n_cols;  // get the number of columns
    
    // allocate memory for the void pointer
    void *ptr = std::malloc(n_rows * n_cols * sizeof(T));
    if (ptr == nullptr) {
        std::cerr << "Failed to request memory space" << std::endl;
        exit(1);
    }
    
    // copy the Armadillo matrix to the void pointer
    std::memcpy(ptr, trans_X.memptr(), n_rows * n_cols * sizeof(T));

    return ptr;
}

// Convert the contents pointed to by void* to fmat, noting that mat is stored by column
arma::fmat sys_mem_to_fmat(void* ptr, int n_rows, int n_cols);

// Convert frowvec to the contents pointed to by void*
void* frowvec_to_sys_mem(const arma::frowvec& X);

// Matrix multiplication, no bugs :)
arma::fmat matmul(const arma::fmat& A, const arma::fmat& B);

// real fft based on bmcv
arma::fmat bm_fft(const arma::fmat& A);

#endif // WRAPPER_H
