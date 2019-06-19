#include "FGPUlib.c"

/*
 * x_t1 = act( W * x_t0 + b )
 * this kernel uses the same size for all layers
 * for simplicity, activation function is ReLU
 * in this blocking scheme, there are two threads per row
 * for 2048 threads
 */
#define N_SIZE              1024
#define DATA_SIZE           sizeof(float)
#define NUM_MATS            1

#define N_BYTES             (N_SIZE * DATA_SIZE)
#define N_WORDS             (N_BYTES / sizeof(uint32_t))
#define NUM_WG_THDS         256
#define NUM_VREGS           16
#define N_WRDS_PER_WG_THD   (N_WORDS / NUM_WG_THDS)
#define NUM_VREGS_PER_BLK   (NUM_VREGS / NUM_MATS)
#define NUM_ELEMS_PER_BLK   (NUM_VREGS_PER_BLK * sizeof(uint8x128) / DATA_SIZE)

__kernel void mlp_fp32(int num_layers, __global float* x, __global float* W, __global float* b)
{
    __local float* x_shr_ptr   = 0;                         // N_SIZE
    __local float* swp_shr_ptr = x_shr_ptr + N_SIZE;        // NUM_WG_THDS

    int gid = get_global_id(0);
    int lid = get_local_id(0);

    int lid_swp_1 = lid ^ 0x0020;
    // bid is the remapped gid that blocks for better global and shared memory coalescing
    int bid = (gid & 0x1fc0) | ((gid & 0x0020) >> 5) | ((gid & 0x001f) << 1);
    int row = bid >> 1;
    int row_pos = bid & 0x0001;

    __global float* W_ptr = W + (bid * NUM_ELEMS_PER_BLK);
    __global float* b_ptr = b;

    for (int i = 0; i < num_layers; ++i) {
        /* bring vector into shared memory */
        for (int j = 0; j < N_WRDS_PER_WG_THD; ++j) {
            x_shr_ptr[lid + NUM_WG_THDS * j] = x[lid + NUM_WG_THDS * j];
        }

        local_sync();

        /* bring matrix into registers */
        __global uint8x128* W_vec_ptr = (__global uint8x128*) W_ptr;
        uint8x128 W_vreg_0  = W_vec_ptr[0];
        uint8x128 W_vreg_1  = W_vec_ptr[1];
        uint8x128 W_vreg_2  = W_vec_ptr[2];
        uint8x128 W_vreg_3  = W_vec_ptr[3];
        uint8x128 W_vreg_4  = W_vec_ptr[4];
        uint8x128 W_vreg_5  = W_vec_ptr[5];
        uint8x128 W_vreg_6  = W_vec_ptr[6];
        uint8x128 W_vreg_7  = W_vec_ptr[7];
        uint8x128 W_vreg_8  = W_vec_ptr[8];
        uint8x128 W_vreg_9  = W_vec_ptr[9];
        uint8x128 W_vreg_10 = W_vec_ptr[10];
        uint8x128 W_vreg_11 = W_vec_ptr[11];
        uint8x128 W_vreg_12 = W_vec_ptr[12];
        uint8x128 W_vreg_13 = W_vec_ptr[13];
        uint8x128 W_vreg_14 = W_vec_ptr[14];
        uint8x128 W_vreg_15 = W_vec_ptr[15];

        /* dot */
        float res = 0.f;
        __local uint8x128* x_shr_vec_ptr = (__local uint8x128*) (x_shr_ptr + (row_pos * NUM_ELEMS_PER_BLK));
        dot_fp32(res, W_vreg_0,  x_shr_vec_ptr, 0 );
        dot_fp32(res, W_vreg_1,  x_shr_vec_ptr, 1 );
        dot_fp32(res, W_vreg_2,  x_shr_vec_ptr, 2 );
        dot_fp32(res, W_vreg_3,  x_shr_vec_ptr, 3 );
        dot_fp32(res, W_vreg_4,  x_shr_vec_ptr, 4 );
        dot_fp32(res, W_vreg_5,  x_shr_vec_ptr, 5 );
        dot_fp32(res, W_vreg_6,  x_shr_vec_ptr, 6 );
        dot_fp32(res, W_vreg_7,  x_shr_vec_ptr, 7 );
        dot_fp32(res, W_vreg_8,  x_shr_vec_ptr, 8 );
        dot_fp32(res, W_vreg_9,  x_shr_vec_ptr, 9 );
        dot_fp32(res, W_vreg_10, x_shr_vec_ptr, 10);
        dot_fp32(res, W_vreg_11, x_shr_vec_ptr, 11);
        dot_fp32(res, W_vreg_12, x_shr_vec_ptr, 12);
        dot_fp32(res, W_vreg_13, x_shr_vec_ptr, 13);
        dot_fp32(res, W_vreg_14, x_shr_vec_ptr, 14);
        dot_fp32(res, W_vreg_15, x_shr_vec_ptr, 15);

        /* reduce */
        // this is within a wavefront so no syncing is necessary
        swp_shr_ptr[lid] = res;
        res += swp_shr_ptr[lid_swp_1];

        /* add bias */
        res += load_if_zero_fp32(0.f, row_pos, &b_ptr[row], 0);

        /* activate */
        res = relu_fp32(res);

        global_sync();

        /* store result */
        store_if_zero_fp32(res, row_pos, &x[row], 0);

        /* next layer */
        W_ptr += N_SIZE * N_SIZE;
        b_ptr += N_SIZE;

        global_sync();
    }
}
