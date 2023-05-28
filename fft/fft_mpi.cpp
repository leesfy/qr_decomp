#include <iostream>
#include <cmath>
#include <complex>
#include <chrono>
#include <mpi.h>

using namespace std;

#define _USE_MATH_DEFINES ;
typedef complex<long double> cd;


inline void fft(cd *arr, const bool inv, const int n) {
    if (n == 1) {
        return;
    }

    int pr_id, num_pr;
    MPI_Comm_size(MPI_COMM_WORLD, &num_pr);
    MPI_Comm_rank(MPI_COMM_WORLD, &pr_id);

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            swap(arr[i], arr[j]);
    }

    cd* wlen_pw = new cd[n];

    if (n >= 4) {
        cd conj (0, (inv ? -1 : 1));
        for (int i = 0; i < n; i += 4) {
            cd t1 = arr[i], t2 = arr[i+1], t3 = arr[i+2];
            arr[i] = t1 + t2 + t3 + arr[i + 3];
            arr[i + 1] = t1 - t2 - conj*(t3 - arr[i + 3]);
            arr[i + 2] = t1 + t2 - t3 - arr[i + 3];
            arr[i + 3] = t1 - t2 + conj*(t3 - arr[i + 3]);
        }
    } else if (n >= 2) {
        for (int i = 0; i < n; i += 2) {
            cd t = arr[i];
            arr[i] = arr[i] + arr[i + 1];
            arr[i + 1] = t - arr[i + 1];
        }
    }


    for (int len = 8; len <= n; len <<= 1) {
        int max_deg = len >> 1;
        long double ang = 2 * M_PI / len * (inv ? 1 : -1);

        cd wlen (cos(ang), sin(ang));
        wlen_pw[0] = cd (1, 0);
		
        for (int i = 1; i < max_deg; ++i) {
            wlen_pw[i] = wlen_pw[i - 1] * wlen;
        }

        int n_blocks = n / len;

        if (n_blocks < num_pr) {
            num_pr = n_blocks;
        }

        int source = (pr_id + num_pr - 1) % num_pr;
        int dest = (pr_id + 1) % num_pr;

        int block_per_process = n_blocks / num_pr;
        int rem_blocks = n_blocks % num_pr;
        int extra = ((pr_id < rem_blocks) ? 1 : 0); 
        int offset = pr_id * block_per_process + min(pr_id, rem_blocks);

        if (pr_id < num_pr) {
            for (int k = 0; k < block_per_process + extra; ++k) {
                cd t, *pu = arr + offset*len + k*len, *pv = arr + offset*len + k*len + max_deg;
                cd *pu_end = arr + offset*len + k*len + max_deg,	*pw = wlen_pw;
                for (; pu != pu_end; ++pu, ++pv, ++pw) {
                    t = *pv * *pw;
                    *pv = *pu - t;
                    *pu += t;
                }
            }

            for (int q = 0; q < num_pr; ++q) {
                MPI_Request reqs[2];
                MPI_Status stats[2];

                int block_send_id = (pr_id - q + num_pr) % num_pr;
                int offset_send = block_send_id * block_per_process + min(block_send_id, rem_blocks);
                int extra_send = ((block_send_id < rem_blocks) ? 1 : 0);

                int block_recv_id = (block_send_id - 1 + num_pr) % num_pr; 
                int offset_recv = block_recv_id * block_per_process + min(block_recv_id, rem_blocks);
                int extra_recv = ((block_recv_id < rem_blocks) ? 1 : 0);

                MPI_Irecv(arr + offset_recv*len, (block_per_process + extra_recv)*len, MPI_C_LONG_DOUBLE_COMPLEX, source, 42, MPI_COMM_WORLD, &reqs[0]);
                MPI_Isend(arr + offset_send*len, (block_per_process + extra_send)*len, MPI_C_LONG_DOUBLE_COMPLEX, dest, 42, MPI_COMM_WORLD, &reqs[1]); 
                
                MPI_Waitall(2, reqs, stats);
            }
        }
    }

    if (inv)
        for (int i = 0; i < n; ++i)
            arr[i] /= n;

    delete[] wlen_pw;
}


int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);

    int pr_id, num_pr;
    MPI_Comm_size(MPI_COMM_WORLD, &num_pr);
    MPI_Comm_rank(MPI_COMM_WORLD, &pr_id);

    int n_degr, i;
    bool inv = true;

    int n = 1000;
    if (argc > 1) {
        n = stoi(argv[1]);
    }

    int lg_n = ceil(log2(n));
    n_degr = pow(2, lg_n);
    if (pr_id == 0) {
        cout << "Extension to: " << n_degr << "\n";
    }

    cd* arr = new cd[n_degr];
    for (i = 0; i < n; ++i) {
        arr[i] = i;
    }

    auto start = chrono::steady_clock::now();

    fft(arr, inv, n_degr);

    auto end = chrono::steady_clock::now();


    if (pr_id == 0) {
        cout << "Time in milliseconds:" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << "\n";
        cout << "Time in seconds:" << chrono::duration_cast<chrono::seconds>(end - start).count() << " sec" << "\n";
    }

    delete[] arr;

    MPI_Finalize();

    return 0;
}
