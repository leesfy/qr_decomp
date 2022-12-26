#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cblas.h>

using namespace std;

inline void houshh_alt(const double* x, int n, double* h) {
    double alpha = cblas_dnrm2(n, x, 1);
    
    memset(h, 0, n*sizeof(x[0]));
    h[0] = alpha;

    for (int i = 0; i < n; ++i)
        h[i] = x[i] - h[i];

    double u_n = cblas_dnrm2(n, h, 1);

    cblas_dscal(n, 1/u_n, h, 1);
}

inline void hhMulRight(const double* u, int k, int n, double* Q, double* tmp) {
    memset(tmp, 0, n*sizeof(tmp[0]));

    cblas_dgemv(CblasColMajor, CblasNoTrans, n, n - k, 1.0d, &Q[k*n], n, u, 1, 0.0d, tmp, 1);

    for(int i = 0; i < n; ++i) {
        for(int j = k; j < n; ++j) {
            Q[i*n + j] -= 2*tmp[i] * u[j-k];
        }
    }
}
 
inline void hhMulLeft(const double* u, int k, int n, double* Q, double* tmp) {
    memset(tmp, 0, n*sizeof(tmp[0]));

    cblas_dgemv(CblasRowMajor, CblasTrans, n - k, n, 1.0d, &Q[k*n], n, u, 1, 0.0d, tmp, 1);

    cblas_dger(CblasRowMajor, n - k, n, -2.0d, u, 1, tmp, 1, &Q[k*n], n);
}

pair<double* , double* > qr_reflect(const double* A, const int n) {
    double* R = new double[n*n];
    memcpy(R, A, n*n*sizeof(A[0]));

    //make eye
    double* Q = new double[n*n];
    memset(Q, 0, n*n*sizeof(A[0]));
    for(int i = 0; i < n; ++i){
        Q[i*n + i] = 1;
    }

    //qr
    double vec_i[n] = {0};   
    double h[n] = {0};
    double tmp[n] = {0};

    for(int i = 0; i < n - 1; ++i){
        cblas_dcopy(n - i, &R[i*n + i], n, vec_i, 1);

        houshh_alt(vec_i, n - i, h);
        hhMulLeft(h, i, n, R, tmp);
        hhMulRight(h, i, n, Q, tmp);
    }

    return pair<double*, double* >(R, Q);
}


int main(int argc, char ** argv){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int n = 1000;
    if (argc > 1) {
        n = stoi(argv[1]);
    }

    double* A = new double[n*n];

    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            A[i*n + j] = i + j;
        }
    }

    auto start = chrono::steady_clock::now();

    pair<double* , double* > t = qr_reflect(A, n);
    double* R = t.first;
    double* Q = t.second;

    auto end = chrono::steady_clock::now();

    cout << "Time in milliseconds:" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << "\n";

    cout << "Time in seconds:" << chrono::duration_cast<chrono::seconds>(end - start).count() << " sec" << "\n";

    delete[] A;
    delete[] R;
    delete[] Q;

    return 0;
}
