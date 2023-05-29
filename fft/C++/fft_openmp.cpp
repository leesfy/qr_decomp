#include <iostream>
#include <cmath>
#include <complex>
#include <chrono>
#include <omp.h>

using namespace std;

#define _USE_MATH_DEFINES ;
typedef complex<long double> cd;

inline int rev (int num, int lg_n) {
    int res = 0;
    for (int i = 0; i < lg_n; ++i)
        if (num & (1<<i))
            res |= 1<<(lg_n - 1 - i);
    return res;
}


inline void fft(cd *arr, const bool inv, const int n, const int num_thr) {
    if (n == 1) {
        return;
    }

    int lg_n = ceil(log2(n));
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)     // bit-reversal permutation convenient for omp
        if (i < rev(i,lg_n))
            swap (arr[i], arr[rev(i,lg_n)]);

    cd* wlen_pw = new cd[n];

    if (n >= 4) {
        cd conj (0, (inv ? -1 : 1));
        #pragma omp parallel for num_threads(num_thr)
        for (int i = 0; i < n; i += 4) {                    // manually made fft for len=4
            cd t1 = arr[i], t2 = arr[i+1], t3 = arr[i+2];
            arr[i] = t1 + t2 + t3 + arr[i + 3];
            arr[i + 1] = t1 - t2 - conj*(t3 - arr[i + 3]);
            arr[i + 2] = t1 + t2 - t3 - arr[i + 3];
            arr[i + 3] = t1 - t2 + conj*(t3 - arr[i + 3]);
        }
    } else if (n >= 2) {
        #pragma omp parallel for
        for (int i = 0; i < n; i += 2) {        // manually made fft for len=2
            cd t = arr[i];
            arr[i] = arr[i] + arr[i + 1];
            arr[i + 1] = t - arr[i + 1];
        }
    }

    for (int len = 8; len <= n; len <<= 1) {         // fft for every degree of 2, starting from 8
        int max_deg = len >> 1;
        long double ang = 2 * M_PI / len * (inv ? 1 : -1);

        cd wlen (cos(ang), sin(ang));
        wlen_pw[0] = cd (1, 0);
		
        for (int i = 1; i < max_deg; ++i) {         // pre-calculate all degrees of wlen
            wlen_pw[i] = wlen_pw[i - 1] * wlen;
        }

        #pragma omp parallel for num_threads(num_thr)
        for (int i = 0; i < n; i += len) {                  // butterfly operation in every block for all blocks
            cd t, *pu = arr + i, *pv = arr + i + max_deg;
            cd *pu_end = arr + i + max_deg,	*pw = wlen_pw;
            for (; pu != pu_end; ++pu, ++pv, ++pw) {
                t = *pv * *pw;
                *pv = *pu - t;
                *pu += t;
            }
        }

    }

    if (inv)
        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
            arr[i] /= n;

    delete[] wlen_pw;
}


int main(int argc, char ** argv) {
    int n_degr, i, num_thread;
    bool inv = true;               // FFT if false, and IFFT if true

    int n = 1000;
    if (argc > 1) {
        n = stoi(argv[1]);        // number of elements of input array
    }

    if (argc > 2) {
    	num_thread = stoi(argv[2]);   // number of threads for omp parallel
    } else {
    	num_thread = 3;
    }

    int lg_n = ceil(log2(n));
    n_degr = pow(2, lg_n);
    cout << "Extension to: " << n_degr << "\n";     // nearest degree of 2 to be extended for

    cd* arr = new cd[n_degr];
    #pragma omp parallel for num_threads(num_thread)
    for (i = 0; i < n; ++i) {
        arr[i] = i;                             // example array
    }

    auto start = chrono::steady_clock::now();
 
    fft(arr, inv, n_degr, num_thread);

    auto end = chrono::steady_clock::now();

    cout << "Time in milliseconds:" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << "\n";
    cout << "Time in seconds:" << chrono::duration_cast<chrono::seconds>(end - start).count() << " sec" << "\n";

    delete[] arr;

    return 0;
}
