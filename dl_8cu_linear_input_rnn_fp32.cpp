#include "FGPUlib.c"

/*
 * h_t1 = act( Ah * h_t0 + Ax * x + b )
 * this is a persistent kernel, so size depends on register state
 * this kernel computes two matrix multiplications per time step; for the hidden and for the input vectors
 * for simplicity, activation function is ReLU
 * in this blocking scheme, there are four threads per row
 * for 4096 threads
 */
#define N_SIZE              1024
#define DATA_SIZE           sizeof(float)
#define NUM_MATS            2

#define N_BYTES             (N_SIZE * DATA_SIZE)
#define N_WORDS             (N_BYTES / sizeof(uint32_t))
#define NUM_WG_THDS         512
#define NUM_VREGS           16
#define N_WRDS_PER_WG_THD   (N_WORDS / NUM_WG_THDS)
#define NUM_VREGS_PER_BLK   (NUM_VREGS / NUM_MATS)
#define NUM_ELEMS_PER_BLK   (NUM_VREGS_PER_BLK * sizeof(uint8x128) / DATA_SIZE)

__kernel void linear_input_rnn_fp32(
    int iters,
    __global float* h, __global float* x,
    __global float* Ah, __global float* Ax, __global float* b
)
{
    __local float* h_shr_ptr   = 0;                         // N_SIZE
    __local float* x_shr_ptr   = h_shr_ptr + N_SIZE;        // N_SIZE
    __local float* swp_shr_ptr = x_shr_ptr + N_SIZE;        // NUM_WG_THDS

    int gid = get_global_id(0);
    int lid = get_local_id(0);

    int lid_swp_1 = lid ^ 0x0020;
    int lid_swp_2 = lid ^ 0x0010;
    // bid is the remapped gid that blocks for better global and shared memory coalescing
    int bid = (gid & 0x1fc0) | ((gid & 0x0030) >> 4) | ((gid & 0x000f) << 2);
    int row = bid >> 2;
    int row_pos = bid & 0x0003;

    __global float* Ah_ptr = Ah + (bid * NUM_ELEMS_PER_BLK);
    __global float* Ax_ptr = Ax + (bid * NUM_ELEMS_PER_BLK);
    __global float* x_ptr = x;

    /* bring matrix into registers */
    __global uint8x128* Ah_vec_ptr = (__global uint8x128*) Ah_ptr;
    uint8x128 Ah_vreg_0 = Ah_vec_ptr[0];
    uint8x128 Ah_vreg_1 = Ah_vec_ptr[1];
    uint8x128 Ah_vreg_2 = Ah_vec_ptr[2];
    uint8x128 Ah_vreg_3 = Ah_vec_ptr[3];
    uint8x128 Ah_vreg_4 = Ah_vec_ptr[4];
    uint8x128 Ah_vreg_5 = Ah_vec_ptr[5];
    uint8x128 Ah_vreg_6 = Ah_vec_ptr[6];
    uint8x128 Ah_vreg_7 = Ah_vec_ptr[7];
    __global uint8x128* Ax_vec_ptr = (__global uint8x128*) Ax_ptr;
    uint8x128 Ax_vreg_0 = Ax_vec_ptr[0];
    uint8x128 Ax_vreg_1 = Ax_vec_ptr[1];
    uint8x128 Ax_vreg_2 = Ax_vec_ptr[2];
    uint8x128 Ax_vreg_3 = Ax_vec_ptr[3];
    uint8x128 Ax_vreg_4 = Ax_vec_ptr[4];
    uint8x128 Ax_vreg_5 = Ax_vec_ptr[5];
    uint8x128 Ax_vreg_6 = Ax_vec_ptr[6];
    uint8x128 Ax_vreg_7 = Ax_vec_ptr[7];

    global_sync();

    for (int i = 0; i < iters; ++i) {
        /* bring vectors into shared memory */
        for (int j = 0; j < N_WRDS_PER_WG_THD; ++j) {
            h_shr_ptr[lid + NUM_WG_THDS * j] = h[lid + NUM_WG_THDS * j];
            x_shr_ptr[lid + NUM_WG_THDS * j] = x_ptr[lid + NUM_WG_THDS * j];
        }

        local_sync();

        /* dot */
        float res = 0.f;
        __local uint8x128* h_shr_vec_ptr = (__local uint8x128*) (h_shr_ptr + (row_pos * NUM_ELEMS_PER_BLK));
        dot_fp32(res, Ah_vreg_0, h_shr_vec_ptr, 0);
        dot_fp32(res, Ah_vreg_1, h_shr_vec_ptr, 1);
        dot_fp32(res, Ah_vreg_2, h_shr_vec_ptr, 2);
        dot_fp32(res, Ah_vreg_3, h_shr_vec_ptr, 3);
        dot_fp32(res, Ah_vreg_4, h_shr_vec_ptr, 4);
        dot_fp32(res, Ah_vreg_5, h_shr_vec_ptr, 5);
        dot_fp32(res, Ah_vreg_6, h_shr_vec_ptr, 6);
        dot_fp32(res, Ah_vreg_7, h_shr_vec_ptr, 7);
        __local uint8x128* x_shr_vec_ptr = (__local uint8x128*) (x_shr_ptr + (row_pos * NUM_ELEMS_PER_BLK));
        dot_fp32(res, Ax_vreg_0, x_shr_vec_ptr, 0);
        dot_fp32(res, Ax_vreg_1, x_shr_vec_ptr, 1);
        dot_fp32(res, Ax_vreg_2, x_shr_vec_ptr, 2);
        dot_fp32(res, Ax_vreg_3, x_shr_vec_ptr, 3);
        dot_fp32(res, Ax_vreg_4, x_shr_vec_ptr, 4);
        dot_fp32(res, Ax_vreg_5, x_shr_vec_ptr, 5);
        dot_fp32(res, Ax_vreg_6, x_shr_vec_ptr, 6);
        dot_fp32(res, Ax_vreg_7, x_shr_vec_ptr, 7);

        /* reduce */
        // this is within a wavefront so no syncing is necessary
        swp_shr_ptr[lid] = res;
        res += swp_shr_ptr[lid_swp_1];
        swp_shr_ptr[lid] = res;
        res += swp_shr_ptr[lid_swp_2];

        /* add bias */
        res += load_if_zero_fp32(0.f, row_pos, &b[row], 0);

        /* activate */
        res = relu_fp32(res);

        global_sync();

        /* store result */
        store_if_zero_fp32(res, row_pos, &h[row], 0);

        /* next input */
        x_ptr += N_SIZE;

        global_sync();
    }
}
