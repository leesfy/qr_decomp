#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>

using namespace std;

extern "C" void dgeqrf_(int* m, int* n, double* a, int* lda, double* tau, double* work, int* lwork, int* info);

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

    int lwork = -1, info = 0, lda = n;
    double* tau = new double[n];
    double* exmp_work = new double[n];


    dgeqrf_(&n, &n, A, &lda, tau, exmp_work, &lwork, &info);
    lwork = exmp_work[0];

    double* work = new double[lwork];


    auto start = chrono::steady_clock::now();  

    dgeqrf_(&n, &n, A, &lda, tau, work, &lwork, &info);

    auto end = chrono::steady_clock::now();
    cout << "Time in milliseconds:" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << "\n";
    cout << "Time in seconds:" << chrono::duration_cast<chrono::seconds>(end - start).count() << " sec" << "\n";

    delete[] A;
    delete[] tau;
    delete[] work;
    delete[] exmp_work;

    return 0;
}