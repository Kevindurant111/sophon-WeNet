#include <armadillo>
#include <string>
#include <assert.h>

arma::fmat as_strided(const arma::fmat& X,
    int n_rows, 
    int n_cols, 
    int row_stride, 
    int col_stride
);

// dim = 0, expand the number of rows; dim = 1, expand the number of columns
arma::fmat pad(const arma::fmat& X,
    int num,
    int dim = 0
);

// Generate a matrix of shape [size, 1] with elements increasing linearly from 0 to size-1 
arma::fmat arange(int size);

// Convert fmat to the contents pointed to by void*, noting that mat is stored by column
void* fmat_to_sys_mem(const arma::fmat& X);

// Matrix multiplication, no bugs :)
arma::fmat matmul(const arma::fmat& A, const arma::fmat& B);