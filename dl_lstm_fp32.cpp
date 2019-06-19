#include "FGPUlib.c"

/*
 * c_t1 = c_t0
        * sigmoid( Wfh * h_t0 + Wfx * x + bf )
        + sigmoid( Wih * h_t0 + Wix * x + bi ) * tanh( Wch * h_t0 + Wcx * x + bc )
 * h_t1 = tanh( c_t1 ) * sigmoid( Woh * h_t0 + Wox * x + bo )
 * this is a persistent kernel, so size depends on register state
 * this kernel computes eight matrix multiplications per time step
 * in this blocking scheme, there are eight threads per row
 * for 4096 threads
 */
#define N_SIZE              512
#define DATA_SIZE           sizeof(float)
#define NUM_MATS            8

#define MAT_SIZE            (N_SIZE * N_SIZE)
#define N_BYTES             (N_SIZE * DATA_SIZE)
#define N_WORDS             (N_BYTES / sizeof(uint32_t))
#define NUM_WG_THDS         256
#define NUM_VREGS           16
#define N_WRDS_PER_WG_THD   (N_WORDS / NUM_WG_THDS)
#define NUM_VREGS_PER_BLK   (NUM_VREGS / NUM_MATS)
#define NUM_ELEMS_PER_BLK   (NUM_VREGS_PER_BLK * sizeof(uint8x128) / DATA_SIZE)

__kernel void lstm_fp32(
    int iters,
    __global float* h, __global float* x, __global float* c,
    __global float* Whs, __global float* Wxs, __global float* bs
)
{
    __global float* Wfh = Whs + (0 * MAT_SIZE);
    __global float* Wih = Whs + (1 * MAT_SIZE);
    __global float* Wch = Whs + (2 * MAT_SIZE);
    __global float* Woh = Whs + (3 * MAT_SIZE);
    __global float* Wfx = Wxs + (0 * MAT_SIZE);
    __global float* Wix = Wxs + (1 * MAT_SIZE);
    __global float* Wcx = Wxs + (2 * MAT_SIZE);
    __global float* Wox = Wxs + (3 * MAT_SIZE);
    __global float* bf = bs + (0 * N_SIZE);
    __global float* bi = bs + (1 * N_SIZE);
    __global float* bc = bs + (2 * N_SIZE);
    __global float* bo = bs + (3 * N_SIZE);

    __local float* h_shr_ptr   = 0;                         // N_SIZE
    __local float* x_shr_ptr   = h_shr_ptr + N_SIZE;        // N_SIZE
    __local float* swp_shr_ptr = x_shr_ptr + N_SIZE;        // NUM_WG_THDS

    int gid = get_global_id(0);
    int lid = get_local_id(0);

    int lid_swp_1 = lid ^ 0x0020;
    int lid_swp_2 = lid ^ 0x0010;
    int lid_swp_3 = lid ^ 0x0008;
    // bid is the remapped gid that blocks for better global and shared memory coalescing
    int bid = (gid & 0x1fc0) | ((gid & 0x0038) >> 3) | ((gid & 0x0007) << 3);
    int row = bid >> 3;
    int row_pos = bid & 0x0007;

    __global float* Wfh_ptr = Wfh + (bid * NUM_ELEMS_PER_BLK);
    __global float* Wih_ptr = Wih + (bid * NUM_ELEMS_PER_BLK);
    __global float* Wch_ptr = Wch + (bid * NUM_ELEMS_PER_BLK);
    __global float* Woh_ptr = Woh + (bid * NUM_ELEMS_PER_BLK);
    __global float* Wfx_ptr = Wfx + (bid * NUM_ELEMS_PER_BLK);
    __global float* Wix_ptr = Wix + (bid * NUM_ELEMS_PER_BLK);
    __global float* Wcx_ptr = Wcx + (bid * NUM_ELEMS_PER_BLK);
    __global float* Wox_ptr = Wox + (bid * NUM_ELEMS_PER_BLK);
    //__global float* bf_ptr = bf;
    //__global float* bi_ptr = bi;
    //__global float* bc_ptr = bc;
    //__global float* bo_ptr = bo;
    __global float* x_ptr = x;

    /* bring matrix into registers */
    __global uint8x128* Wfh_vec_ptr = (__global uint8x128*) Wfh_ptr;
    uint8x128 Wfh_vreg_0 = Wfh_vec_ptr[0];
    uint8x128 Wfh_vreg_1 = Wfh_vec_ptr[1];
    __global uint8x128* Wih_vec_ptr = (__global uint8x128*) Wih_ptr;
    uint8x128 Wih_vreg_0 = Wih_vec_ptr[0];
    uint8x128 Wih_vreg_1 = Wih_vec_ptr[1];
    __global uint8x128* Wch_vec_ptr = (__global uint8x128*) Wch_ptr;
    uint8x128 Wch_vreg_0 = Wch_vec_ptr[0];
    uint8x128 Wch_vreg_1 = Wch_vec_ptr[1];
    __global uint8x128* Woh_vec_ptr = (__global uint8x128*) Woh_ptr;
    uint8x128 Woh_vreg_0 = Woh_vec_ptr[0];
    uint8x128 Woh_vreg_1 = Woh_vec_ptr[1];
    __global uint8x128* Wfx_vec_ptr = (__global uint8x128*) Wfx_ptr;
    uint8x128 Wfx_vreg_0 = Wfx_vec_ptr[0];
    uint8x128 Wfx_vreg_1 = Wfx_vec_ptr[1];
    __global uint8x128* Wix_vec_ptr = (__global uint8x128*) Wix_ptr;
    uint8x128 Wix_vreg_0 = Wix_vec_ptr[0];
    uint8x128 Wix_vreg_1 = Wix_vec_ptr[1];
    __global uint8x128* Wcx_vec_ptr = (__global uint8x128*) Wcx_ptr;
    uint8x128 Wcx_vreg_0 = Wcx_vec_ptr[0];
    uint8x128 Wcx_vreg_1 = Wcx_vec_ptr[1];
    __global uint8x128* Wox_vec_ptr = (__global uint8x128*) Wox_ptr;
    uint8x128 Wox_vreg_0 = Wox_vec_ptr[0];
    uint8x128 Wox_vreg_1 = Wox_vec_ptr[1];

    global_sync();

    for (int i = 0; i < iters; ++i) {
        /* bring vectors into shared memory */
        for (int j = 0; j < N_WRDS_PER_WG_THD; ++j) {
            h_shr_ptr[lid + NUM_WG_THDS * j] = h[lid + NUM_WG_THDS * j];
            x_shr_ptr[lid + NUM_WG_THDS * j] = x_ptr[lid + NUM_WG_THDS * j];
        }

        local_sync();

        /* dot */
        float res_f = 0.f;
        float res_i = 0.f;
        float res_c = 0.f;
        float res_o = 0.f;
        __local uint8x128* h_shr_vec_ptr = (__local uint8x128*) (h_shr_ptr + (row_pos * NUM_ELEMS_PER_BLK));
        dot_fp32(res_f, Wfh_vreg_0, h_shr_vec_ptr, 0);
        dot_fp32(res_f, Wfh_vreg_1, h_shr_vec_ptr, 1);
        dot_fp32(res_i, Wih_vreg_0, h_shr_vec_ptr, 0);
        dot_fp32(res_i, Wih_vreg_1, h_shr_vec_ptr, 1);
        dot_fp32(res_c, Wch_vreg_0, h_shr_vec_ptr, 0);
        dot_fp32(res_c, Wch_vreg_1, h_shr_vec_ptr, 1);
        dot_fp32(res_o, Woh_vreg_0, h_shr_vec_ptr, 0);
        dot_fp32(res_o, Woh_vreg_1, h_shr_vec_ptr, 1);
        __local uint8x128* x_shr_vec_ptr = (__local uint8x128*) (x_shr_ptr + (row_pos * NUM_ELEMS_PER_BLK));
        dot_fp32(res_f, Wfx_vreg_0, x_shr_vec_ptr, 0);
        dot_fp32(res_f, Wfx_vreg_1, x_shr_vec_ptr, 1);
        dot_fp32(res_i, Wix_vreg_0, x_shr_vec_ptr, 0);
        dot_fp32(res_i, Wix_vreg_1, x_shr_vec_ptr, 1);
        dot_fp32(res_c, Wcx_vreg_0, x_shr_vec_ptr, 0);
        dot_fp32(res_c, Wcx_vreg_1, x_shr_vec_ptr, 1);
        dot_fp32(res_o, Wox_vreg_0, x_shr_vec_ptr, 0);
        dot_fp32(res_o, Wox_vreg_1, x_shr_vec_ptr, 1);

        /* reduce */
        // this is within a wavefront so no syncing is necessary
        swp_shr_ptr[lid] = res_f;
        res_f += swp_shr_ptr[lid_swp_1];
        swp_shr_ptr[lid] = res_f;
        res_f += swp_shr_ptr[lid_swp_2];
        swp_shr_ptr[lid] = res_f;
        res_f += swp_shr_ptr[lid_swp_3];
        swp_shr_ptr[lid] = res_i;
        res_i += swp_shr_ptr[lid_swp_1];
        swp_shr_ptr[lid] = res_i;
        res_i += swp_shr_ptr[lid_swp_2];
        swp_shr_ptr[lid] = res_i;
        res_i += swp_shr_ptr[lid_swp_3];
        swp_shr_ptr[lid] = res_c;
        res_c += swp_shr_ptr[lid_swp_1];
        swp_shr_ptr[lid] = res_c;
        res_c += swp_shr_ptr[lid_swp_2];
        swp_shr_ptr[lid] = res_c;
        res_c += swp_shr_ptr[lid_swp_3];
        swp_shr_ptr[lid] = res_o;
        res_o += swp_shr_ptr[lid_swp_1];
        swp_shr_ptr[lid] = res_o;
        res_o += swp_shr_ptr[lid_swp_2];
        swp_shr_ptr[lid] = res_o;
        res_o += swp_shr_ptr[lid_swp_3];

        /* add bias */
        res_f += load_if_zero_fp32(0.f, row_pos, &bf[row], 0);
        res_i += load_if_zero_fp32(0.f, row_pos, &bi[row], 0);
        res_c += load_if_zero_fp32(0.f, row_pos, &bc[row], 0);
        res_o += load_if_zero_fp32(0.f, row_pos, &bo[row], 0);

        /* activate */
        res_f = sigmoid_fp32(res_f);
        res_i = sigmoid_fp32(res_i);
        res_c =    tanh_fp32(res_c);
        res_o = sigmoid_fp32(res_o);

        /* put gates together */
        float tmp_c = c[row] * res_f + res_i * res_c;
        float tmp_h = tanh_fp32(tmp_c) * res_o;

        global_sync();

        /* store result */
        store_if_zero_fp32(tmp_c, row_pos, &c[row], 0);
        store_if_zero_fp32(tmp_h, row_pos, &h[row], 0);

        /* next input */
        x_ptr += N_SIZE;

        global_sync();
    }
}
