#!/usr/bin/env python
# coding: utf-8

import numpy as np
from mpi4py import MPI
import sys 
import time


def fft(arr, inv, n, comm):
    if (n == 1):
        return
    
    pr_id = comm.Get_rank()
    num_pr = comm.Get_size()
    
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while (j & bit):
            j ^= bit 
            bit >>= 1
            
        j ^= bit
        if (i < j):
            arr[i], arr[j] = arr[j], arr[i]

    wlen_pw = np.zeros(n, dtype=np.complex128)

    if n >= 4:
        conj = 1j* (-1 if inv else 1)
        for i in range(0, n, 4):
            t1, t2 = arr[i], arr[i+1]
            t3 = arr[i+2]
            arr[i] = t1 + t2 + t3 + arr[i + 3]
            arr[i + 1] = t1 - t2 - conj*(t3 - arr[i + 3])
            arr[i + 2] = t1 + t2 - t3 - arr[i + 3]
            arr[i + 3] = t1 - t2 + conj*(t3 - arr[i + 3])
    elif (n >= 2):
        for i in range(0, n, 2):
            arr[i], arr[i + 1] = arr[i] + arr[i + 1], arr[i] - arr[i + 1]
        
    ln = 8
    while (ln <= n):
        max_deg = ln >> 1
        ang = 2 * np.pi / ln * (1 if inv else -1)
        wlen = np.cos(ang) + 1j*np.sin(ang)
        wlen_pw = np.power(wlen, np.arange(0, max_deg), dtype=np.complex128)
        
        n_blocks = n // ln
        if n_blocks < num_pr:
            num_pr = n_blocks

        dest = (pr_id + num_pr - 1) % num_pr
        source = (pr_id + 1) % num_pr
        
        block_per_process = n_blocks // num_pr
        rem_blocks = n_blocks % num_pr
        extra = 1 if (pr_id < rem_blocks) else 0
        offset = pr_id*block_per_process + min(pr_id, rem_blocks)

        if (pr_id < num_pr):            
            for k in range(block_per_process + extra):
                for j in range(max_deg):
                    u, v = arr[offset*ln + k*ln + j], arr[offset*ln + k*ln + max_deg + j] * wlen_pw[j]
                    arr[offset*ln + k*ln + j] = u + v
                    arr[offset*ln + k*ln + max_deg + j] = u - v
                    
            for q in range(np.floor(np.log2(num_pr)).astype(int)):    
                block_send_id = (pr_id + q + num_pr) % num_pr
                offset_send = block_send_id * block_per_process + min(block_send_id, rem_blocks)
                extra_send = 1 if (block_send_id < rem_blocks) else 0

                block_recv_id = (block_send_id + 1 + num_pr) % num_pr
                offset_recv = block_recv_id * block_per_process + min(block_recv_id, rem_blocks)
                extra_recv = 1 if (block_recv_id < rem_blocks) else 0
	
                x_recv = np.zeros((block_per_process + extra_recv)*ln, dtype=np.complex128)
                reqs_r = comm.Irecv(x_recv, source, 42)
                reqs_s = comm.Isend(arr[offset_send*ln:offset_send*ln + (block_per_process + extra_send)*ln], dest, 42) 
    
                reqs_r.wait()
                reqs_s.wait()
                
                np.copyto(arr[offset_recv*ln:offset_recv*ln + (block_per_process + extra_recv)*ln], x_recv)
        
        ln <<= 1
 

    if (inv):
        arr /= n   
    
    return arr


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    
    pr_id = comm.Get_rank()
    num_pr = comm.Get_size()
    
    inv = True
    n = int(sys.argv[1])
    
    lg_n = np.ceil(np.log2(n))
    n_degr = np.power(2, lg_n).astype(int)
    if (pr_id == 0):
        print("Extension to: ", n_degr)

    arr = np.array([i for i in range(n)] + [0]*(n_degr - n), dtype=np.complex128)
    
    start_time = time.time()
 
    fft(arr, inv, n_degr, comm)

    end_time = time.time()

    if (pr_id == 0):
        print(f"Time in seconds: {end_time - start_time} sec")
        print("Final array: ", arr[:5])

    MPI.Finalize()


