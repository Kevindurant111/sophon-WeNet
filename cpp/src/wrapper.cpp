#include "wrapper.h"

arma::fmat as_strided(const arma::fmat& X, int n_rows, int n_cols, int row_stride, int col_stride) {
    arma::fmat result(n_rows, n_cols);
    arma::fmat X0 = resize(X, 1, X.n_rows * X.n_cols);
    int start = 0;
    int cur = start;
    for(int i = 0; i < n_rows; i++) {
        cur = start;
        for(int j = 0; j < n_cols; j++) {
            result(i, j) = X(0, cur);
            cur += col_stride;
            cur %= X0.n_cols;
        }
        start += row_stride;
        start %= X0.n_cols;
    } 
    return result;
}

arma::fmat pad(const arma::fmat& X, int num, int dim) {
    arma::fmat result(X);
    for(int i = 0; i < num; i++) {
        if(dim == 0) {
            result = arma::join_cols(result, X);
        }
        else {
            result = arma::join_rows(result, X);
        }
    }
    return result;
}

arma::fmat arange(int size) {
    arma::fvec lin_vec = arma::linspace<arma::fvec>(0, size - 1, size);
    arma::fmat lin_mat(size, 1);
    lin_mat.col(0) = lin_vec;
    return lin_mat;
}

void* fmat_to_sys_mem(const arma::fmat& X) {
    arma::fmat trans_X = arma::trans(X);
    int n_rows = trans_X.n_rows;  // get the number of rows
    int n_cols = trans_X.n_cols;  // get the number of columns
    
    // allocate memory for the void pointer
    void *ptr = std::malloc(n_rows * n_cols * sizeof(float));
    if (ptr == nullptr) {
        std::cerr << "Failed to request memory space" << std::endl;
        exit(1);
    }
    
    // copy the Armadillo matrix to the void pointer
    std::memcpy(ptr, trans_X.memptr(), n_rows * n_cols * sizeof(float));
    
    // print the values in the void pointer
    // float *data = static_cast<float*>(ptr);
    // for (int i = 0; i < n_rows * n_cols; ++i) {
    //     std::cout << data[i] << ' ';
    // }
    // std::cout << std::endl;
    return ptr;
}

arma::fmat matmul(const arma::fmat& A, const arma::fmat& B) {
    assert(A.n_cols == B.n_rows && "Matrix multiplication: dimensional mismatch!");
    arma::fmat result(A.n_rows, B.n_cols, arma::fill::zeros);
    for(arma::uword i = 0; i < result.n_rows; i++) {
        for(arma::uword j = 0; j < result.n_cols; j++) {
            for(arma::uword k = 0; k < A.n_cols; k++) {
                result(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return result;
}