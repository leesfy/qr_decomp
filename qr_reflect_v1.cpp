#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>
#include <cassert>
#include <chrono>

using namespace std;


inline double norm(const vector<double> &v, int &n) {
    double res = 0;
    for (int i = 0; i < n; ++i) {
        res += v[i]*v[i];
    }
    return sqrt(res);
}


inline void houshh_alt(const vector<double> &x, int n, vector<double> &h) {
    double alpha = norm(x, n);

    h[0] = alpha;
    fill(h.begin() + 1, h.end(), 0);

    transform(x.begin(), x.end(), h.begin(), h.begin(), minus<double>());

    double u_n = norm(h, n);
    for_each(h.begin(), h.end(), [u_n](double &c){ c /= u_n; });
}


inline void hhMulRight(const vector<double> &u, int k, int n, vector<double> &Q, vector<double> &tmp) {
    for(int i = 0; i < n; i++) {
        tmp[i] = 0;
        for(int j = k; j < n; j++) {
            tmp[i] += Q[i*n + j] * u[j-k];
        }
    }
    for(int i = 0; i < n; ++i) {
        for(int j = k; j < n; ++j) {
            Q[i*n + j] -= 2*tmp[i]*u[j-k];
        }
    }
}

inline void hhMulLeft(const vector<double> &u, int k, int n, vector<double> &Q, vector<double> &tmp) {
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

pair<vector<double>, vector<double> > qr_reflect(const vector<double> &A, const int n) {
    vector<double> A_tmp = A;

    //make eye
    vector<double> Q(n*n, 0);
    for(int i = 0; i < n; ++i){
        Q[i*n + i] = 1;
    }

    //qr
    vector<double> vec_i(n);   
    vector<double> h(n);
    vector<double> tmp(n);

    for(int i = 0; i < n - 1; ++i){
        for(int j = i; j < n; ++j){
            vec_i[j - i] = A_tmp[j*n + i];
        }
        houshh_alt(vec_i, n - i, h);
        hhMulLeft(h, i, n, A_tmp, tmp);
        hhMulRight(h, i, n, Q, tmp);
    }

    return pair<vector<double>, vector<double> >(A_tmp, Q);
}
	

int main(int argc, char ** argv){
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);

    int n = 1000;
    if (argc > 1) {
        n = stoi(argv[1]);
    }

    vector<double> A(n*n, 0);

    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            A[i*n + j] = i + j;
        }
    }

    auto start = chrono::steady_clock::now();

    pair<vector<double>, vector<double> > t = qr_reflect(A, n);
    vector<double> R = t.first;
    vector<double> Q = t.second;

    auto end = chrono::steady_clock::now();

    cout << "Time in milliseconds:" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << "\n";

    cout << "Time in seconds:" << chrono::duration_cast<chrono::seconds>(end - start).count() << " sec" << "\n";

    delete[] A, R, Q;

    return 0;
}
