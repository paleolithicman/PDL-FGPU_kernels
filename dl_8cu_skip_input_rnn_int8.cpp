#include "FGPUlib.c"

/*
 * h_t1 = act( A * h_t0 + x )
 * this is a persistent kernel, so size depends on register state
 * this kernel assumes the input weights, input activations, and bias have been precomputed into the x vectors
 * that is, x is the precomputed biased matrix multiplication on the input
 * for simplicity, activation function is ReLU and math is in fixed point
 * in this blocking scheme, there is one thread per row
 * for 2048 threads
 */
#define N_SIZE              2048
#define DATA_SIZE           sizeof(int8_t)
#define NUM_MATS            1

#define N_BYTES             (N_SIZE * DATA_SIZE)
#define N_WORDS             (N_BYTES / sizeof(uint32_t))
#define NUM_WG_THDS         256
#define NUM_VREGS           16
#define N_WRDS_PER_WG_THD   (N_WORDS / NUM_WG_THDS)
#define NUM_VREGS_PER_BLK   (NUM_VREGS / NUM_MATS)
#define NUM_ELEMS_PER_BLK   (NUM_VREGS_PER_BLK * sizeof(uint8x128) / DATA_SIZE)

__kernel void skip_input_rnn_int8(int iters, __global int8_t* h, __global int8_t* x, __global int8_t* A)
{
    __local int8_t* h_shr_ptr = 0;                                      // N_SIZE
    __local int* swp_shr_ptr  = (__local int*) (h_shr_ptr + N_SIZE);    // NUM_WG_THDS

    int gid = get_global_id(0);
    int lid = get_local_id(0);

    __global int8_t* A_ptr = A + (gid * NUM_ELEMS_PER_BLK);
    __global int8_t* x_ptr = x;

    /* bring matrix into registers */
    __global uint8x128* A_vec_ptr = (__global uint8x128*) A_ptr;
    uint8x128 A_vreg_0  = A_vec_ptr[0];
    uint8x128 A_vreg_1  = A_vec_ptr[1];
    uint8x128 A_vreg_2  = A_vec_ptr[2];
    uint8x128 A_vreg_3  = A_vec_ptr[3];
    uint8x128 A_vreg_4  = A_vec_ptr[4];
    uint8x128 A_vreg_5  = A_vec_ptr[5];
    uint8x128 A_vreg_6  = A_vec_ptr[6];
    uint8x128 A_vreg_7  = A_vec_ptr[7];
    uint8x128 A_vreg_8  = A_vec_ptr[8];
    uint8x128 A_vreg_9  = A_vec_ptr[9];
    uint8x128 A_vreg_10 = A_vec_ptr[10];
    uint8x128 A_vreg_11 = A_vec_ptr[11];
    uint8x128 A_vreg_12 = A_vec_ptr[12];
    uint8x128 A_vreg_13 = A_vec_ptr[13];
    uint8x128 A_vreg_14 = A_vec_ptr[14];
    uint8x128 A_vreg_15 = A_vec_ptr[15];

    global_sync();

    for (int i = 0; i < iters; ++i) {
        /* bring vector into shared memory */
        __global uint32_t* h_word_ptr    = (__global uint32_t*) h;
        __local uint32_t* h_shr_word_ptr = (__local uint32_t*) h_shr_ptr;
        for (int j = 0; j < N_WRDS_PER_WG_THD; ++j) {
            h_shr_word_ptr[lid + NUM_WG_THDS * j] = h_word_ptr[lid + NUM_WG_THDS * j];
        }

        local_sync();

        /* dot */
        int res = 0;
        __local uint8x128* h_shr_vec_ptr = (__local uint8x128*) h_shr_ptr;
        dot_int32(res, A_vreg_0,  h_shr_vec_ptr, 0 );
        dot_int32(res, A_vreg_1,  h_shr_vec_ptr, 1 );
        dot_int32(res, A_vreg_2,  h_shr_vec_ptr, 2 );
        dot_int32(res, A_vreg_3,  h_shr_vec_ptr, 3 );
        dot_int32(res, A_vreg_4,  h_shr_vec_ptr, 4 );
        dot_int32(res, A_vreg_5,  h_shr_vec_ptr, 5 );
        dot_int32(res, A_vreg_6,  h_shr_vec_ptr, 6 );
        dot_int32(res, A_vreg_7,  h_shr_vec_ptr, 7 );
        dot_int32(res, A_vreg_8,  h_shr_vec_ptr, 8 );
        dot_int32(res, A_vreg_9,  h_shr_vec_ptr, 9 );
        dot_int32(res, A_vreg_10, h_shr_vec_ptr, 10);
        dot_int32(res, A_vreg_11, h_shr_vec_ptr, 11);
        dot_int32(res, A_vreg_12, h_shr_vec_ptr, 12);
        dot_int32(res, A_vreg_13, h_shr_vec_ptr, 13);
        dot_int32(res, A_vreg_14, h_shr_vec_ptr, 14);
        dot_int32(res, A_vreg_15, h_shr_vec_ptr, 15);

        /* add input */
        res += x_ptr[gid];

        /* activate */
        res = relu_int32(res);

        global_sync();

        /* store result */
        h[gid] = (int8_t) res;

        /* next input */
        x_ptr += N_SIZE;

        global_sync();
    }
}
