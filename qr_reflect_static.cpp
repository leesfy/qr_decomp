#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>

using namespace std;


inline double norm(const double* v, int n) {
    double res = 0;
    for (int i = 0; i < n; ++i) {
        res += v[i]*v[i];
    }
    return sqrt(res);
}


inline void houshh_alt(const double* x, int n, double* h) {
    double alpha = norm(x, n);
    
    memset(h, 0, n*sizeof(x[0]));
    h[0] = alpha;

    for (int i = 0; i < n; ++i)
        h[i] = x[i] - h[i];

    double u_n = norm(h, n);
    for (int i = 0; i < n; ++i)
        h[i] /= u_n;

}


inline void hhMulRight(const double* u, int k, int n, double* Q, double* tmp) {
    for(int i = 0; i < n; ++i) {
        tmp[i] = 0;
        for(int j = k; j < n; ++j) {
            tmp[i] += Q[i*n + j] * u[j-k];
        }
    }
    for(int i = 0; i < n; ++i) {
        for(int j = k; j < n; ++j) {
            Q[i*n + j] -= 2*tmp[i]*u[j-k];
        }
    }
}

inline void hhMulLeft(const double* u, int k, int n, double* Q, double* tmp) {    // !!!!!!
    for(int i = 0; i < n; ++i) {
        tmp[i] = 0;
        for(int j = k; j < n; ++j) {
            tmp[i] += Q[j*n + i] * u[j-k];
        }
    }
    for(int i = 0; i < n; ++i) {
        for(int j = k; j < n; ++j) {
            Q[j*n + i] -= 2* tmp[i]*u[j-k];
        }
    }
}

pair<double* , double* > qr_reflect(const double* A, const int n) {
    double* A_tmp = new double[n*n];
    memcpy(A_tmp, A, n*n*sizeof(A[0]));

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
        for(int j = i; j < n; ++j){
            vec_i[j - i] = A_tmp[j*n + i];
        }
        houshh_alt(vec_i, n - i, h);
        hhMulLeft(h, i, n, A_tmp, tmp);
        hhMulRight(h, i, n, Q, tmp);
    }

    return pair<double*, double* >(A_tmp, Q);
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

    return 0;
}
