#include "FGPUlib.c"

/*
 * r = sigmoid( Wrh * h_t0 + Wrx * x + br )
 * z = sigmoid( Wzh * h_t0 + Wzx * x + bz )
 * h_t1 = (1 - z) * h_t0 + z * tanh( Whh * (r * h_t0) + Whx * x + bh )
 * this is a persistent kernel, so size depends on register state
 * this kernel computes six matrix multiplications per time step
 * in this blocking scheme, there are eight threads per row
 * for 4096 threads
 */
#define N_SIZE              512
#define DATA_SIZE           sizeof(float)
#define NUM_MATS            (6 + 2)     // only 3 gates, but padded to the nearest power of 2

#define N_BYTES             (N_SIZE * DATA_SIZE)
#define N_WORDS             (N_BYTES / sizeof(uint32_t))
#define NUM_WG_THDS         512
#define NUM_VREGS           16
#define N_WRDS_PER_WG_THD   (N_WORDS / NUM_WG_THDS)
#define NUM_VREGS_PER_BLK   (NUM_VREGS / NUM_MATS)
#define NUM_ELEMS_PER_BLK   (NUM_VREGS_PER_BLK * sizeof(uint8x128) / DATA_SIZE)

__kernel void gru_fp32(
    int iters,
    __global float* h, __global float* x, __global float* r,
    __global float* Wrh, __global float* Wrx, __global float* br,
    __global float* Wzh, __global float* Wzx, __global float* bz,
    __global float* Whh, __global float* Whx, __global float* bh
)
{
    __local float* h_shr_ptr   = 0;                         // N_SIZE
    __local float* x_shr_ptr   = h_shr_ptr + N_SIZE;        // N_SIZE
    __local float* r_shr_ptr   = x_shr_ptr + N_SIZE;        // N_SIZE
    __local float* swp_shr_ptr = r_shr_ptr + N_SIZE;        // NUM_WG_THDS

    int gid = get_global_id(0);
    int lid = get_local_id(0);

    int lid_swp_1 = lid ^ 0x0020;
    int lid_swp_2 = lid ^ 0x0010;
    int lid_swp_3 = lid ^ 0x0008;
    // bid is the remapped gid that blocks for better global and shared memory coalescing
    int bid = (gid & 0x1fc0) | ((gid & 0x0038) >> 3) | ((gid & 0x0007) << 3);
    int row = bid >> 3;
    int row_pos = bid & 0x0007;

    __global float* Wrh_ptr = Wrh + (bid * NUM_ELEMS_PER_BLK);
    __global float* Wzh_ptr = Wzh + (bid * NUM_ELEMS_PER_BLK);
    __global float* Whh_ptr = Whh + (bid * NUM_ELEMS_PER_BLK);
    __global float* Wrx_ptr = Wrx + (bid * NUM_ELEMS_PER_BLK);
    __global float* Wzx_ptr = Wzx + (bid * NUM_ELEMS_PER_BLK);
    __global float* Whx_ptr = Whx + (bid * NUM_ELEMS_PER_BLK);
    //__global float* br_ptr = br;
    //__global float* bz_ptr = bz;
    //__global float* bh_ptr = bh;
    __global float* x_ptr = x;

    /* bring matrix into registers */
    __global uint8x128* Wrh_vec_ptr = (__global uint8x128*) Wrh_ptr;
    uint8x128 Wrh_vreg_0 = Wrh_vec_ptr[0];
    uint8x128 Wrh_vreg_1 = Wrh_vec_ptr[1];
    __global uint8x128* Wzh_vec_ptr = (__global uint8x128*) Wzh_ptr;
    uint8x128 Wzh_vreg_0 = Wzh_vec_ptr[0];
    uint8x128 Wzh_vreg_1 = Wzh_vec_ptr[1];
    __global uint8x128* Whh_vec_ptr = (__global uint8x128*) Whh_ptr;
    uint8x128 Whh_vreg_0 = Whh_vec_ptr[0];
    uint8x128 Whh_vreg_1 = Whh_vec_ptr[1];
    __global uint8x128* Wrx_vec_ptr = (__global uint8x128*) Wrx_ptr;
    uint8x128 Wrx_vreg_0 = Wrx_vec_ptr[0];
    uint8x128 Wrx_vreg_1 = Wrx_vec_ptr[1];
    __global uint8x128* Wzx_vec_ptr = (__global uint8x128*) Wzx_ptr;
    uint8x128 Wzx_vreg_0 = Wzx_vec_ptr[0];
    uint8x128 Wzx_vreg_1 = Wzx_vec_ptr[1];
    __global uint8x128* Whx_vec_ptr = (__global uint8x128*) Whx_ptr;
    uint8x128 Whx_vreg_0 = Whx_vec_ptr[0];
    uint8x128 Whx_vreg_1 = Whx_vec_ptr[1];

    global_sync();

    for (int i = 0; i < iters; ++i) {
        /* bring vectors into shared memory 1 */
        for (int j = 0; j < N_WRDS_PER_WG_THD; ++j) {
            h_shr_ptr[lid + NUM_WG_THDS * j] = h[lid + NUM_WG_THDS * j];
            x_shr_ptr[lid + NUM_WG_THDS * j] = x_ptr[lid + NUM_WG_THDS * j];
        }

        local_sync();

        /* dot 1 */
        float res_r = 0.f;
        float res_z = 0.f;
        __local uint8x128* h_shr_vec_ptr = (__local uint8x128*) (h_shr_ptr + (row_pos * NUM_ELEMS_PER_BLK));
        dot_fp32(res_r, Wrh_vreg_0, h_shr_vec_ptr, 0);
        dot_fp32(res_r, Wrh_vreg_1, h_shr_vec_ptr, 1);
        dot_fp32(res_z, Wzh_vreg_0, h_shr_vec_ptr, 0);
        dot_fp32(res_z, Wzh_vreg_1, h_shr_vec_ptr, 1);
        __local uint8x128* x_shr_vec_ptr = (__local uint8x128*) (x_shr_ptr + (row_pos * NUM_ELEMS_PER_BLK));
        dot_fp32(res_r, Wrx_vreg_0, x_shr_vec_ptr, 0);
        dot_fp32(res_r, Wrx_vreg_1, x_shr_vec_ptr, 1);
        dot_fp32(res_z, Wzx_vreg_0, x_shr_vec_ptr, 0);
        dot_fp32(res_z, Wzx_vreg_1, x_shr_vec_ptr, 1);

        /* reduce 1 */
        // this is within a wavefront so no syncing is necessary
        swp_shr_ptr[lid] = res_r;
        res_r += swp_shr_ptr[lid_swp_1];
        swp_shr_ptr[lid] = res_r;
        res_r += swp_shr_ptr[lid_swp_2];
        swp_shr_ptr[lid] = res_r;
        res_r += swp_shr_ptr[lid_swp_3];
        swp_shr_ptr[lid] = res_z;
        res_z += swp_shr_ptr[lid_swp_1];
        swp_shr_ptr[lid] = res_z;
        res_z += swp_shr_ptr[lid_swp_2];
        swp_shr_ptr[lid] = res_z;
        res_z += swp_shr_ptr[lid_swp_3];

        /* add bias 1 */
        res_r += load_if_zero_fp32(0.f, row_pos, &br[row], 0);
        res_z += load_if_zero_fp32(0.f, row_pos, &bz[row], 0);

        /* activate 1 */
        res_r = sigmoid_fp32(res_r);
        res_z = sigmoid_fp32(res_z);

        global_sync();

        /* store result 1 */
        store_if_zero_fp32(res_r, row_pos, &r[row], 0);

        global_sync();

        /* bring vectors into shared memory 2 */
        for (int j = 0; j < N_WRDS_PER_WG_THD; ++j) {
            r_shr_ptr[lid + NUM_WG_THDS * j] = r[lid + NUM_WG_THDS * j] * h_shr_ptr[lid + NUM_WG_THDS * j];
        }

        local_sync();

        /* dot 2 */
        float res_h = 0.f;
        __local uint8x128* r_shr_vec_ptr = (__local uint8x128*) (r_shr_ptr + (row_pos * NUM_ELEMS_PER_BLK));
        dot_fp32(res_h, Whh_vreg_0, r_shr_vec_ptr, 0);
        dot_fp32(res_h, Whh_vreg_1, r_shr_vec_ptr, 1);
        //__local uint8x128* x_shr_vec_ptr = (__local uint8x128*) (x_shr_ptr + (row_pos * NUM_ELEMS_PER_BLK));
        dot_fp32(res_h, Whx_vreg_0, x_shr_vec_ptr, 0);
        dot_fp32(res_h, Whx_vreg_1, x_shr_vec_ptr, 1);

        /* reduce 2 */
        // this is within a wavefront so no syncing is necessary
        swp_shr_ptr[lid] = res_h;
        res_h += swp_shr_ptr[lid_swp_1];
        swp_shr_ptr[lid] = res_h;
        res_h += swp_shr_ptr[lid_swp_2];
        swp_shr_ptr[lid] = res_h;
        res_h += swp_shr_ptr[lid_swp_3];

        /* add bias 2 */
        res_h += load_if_zero_fp32(0.f, row_pos, &bh[row], 0);

        /* activate 2 */
        res_h = tanh_fp32(res_h);

        /* put gates together */
        float tmp_h = (1 - res_z) * h_shr_ptr[row] + res_z * res_h;

        global_sync();

        /* store result 2 */
        store_if_zero_fp32(tmp_h, row_pos, &h[row], 0);

        /* next input */
        x_ptr += N_SIZE;

        global_sync();
    }
}
