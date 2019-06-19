#include "FGPUlib.c"

/*
 * c_t1 = c_t0
        * sigmoid( Wfh * h_t0 + Wfx * x + bf )
        + sigmoid( Wih * h_t0 + Wix * x + bi ) * tanh( Wch * h_t0 + Wcx * x + bc )
 * h_t1 = tanh( c_t1 ) * sigmoid( Woh * h_t0 + Wox * x + bo )
 * this is a persistent kernel, so size depends on register state
 * this kernel computes eight matrix multiplications per time step
 * each thread computes one gate (2 matrices); gates are interleaved by wavefronts
 * gates are interleaved by the low 2 bits of the wavefront within a workgroup
 * blocking gates by wavefronts allows for fewer matrices per thread reducing reduction operations
 * gates for the same elements are in a single workgroup to take advantage of shared memory
 * control flow for the gates does not cause divergence within the wavefront
 * in this blocking scheme, there are two threads per row
 * for 4096 threads
 */
#define N_SIZE              512
#define DATA_SIZE           sizeof(float)
#define NUM_MATS            2

#define MAT_SIZE            (N_SIZE * N_SIZE)
#define N_BYTES             (N_SIZE * DATA_SIZE)
#define N_WORDS             (N_BYTES / sizeof(uint32_t))
#define NUM_WG_THDS         256
#define NUM_VREGS           16
#define N_WRDS_PER_WG_THD   (N_WORDS / NUM_WG_THDS)
#define NUM_VREGS_PER_BLK   (NUM_VREGS / NUM_MATS)
#define NUM_ELEMS_PER_BLK   (NUM_VREGS_PER_BLK * sizeof(uint8x128) / DATA_SIZE)

#define GATE_FORGET         0
#define GATE_INPUT          1
#define GATE_CANDIDATE      2
#define GATE_OUTPUT         3

__kernel void lstm_gate_blk_fp32(
    int iters,
    __global float* h, __global float* x, __global float* c,
    __global float* Whs, __global float* Wxs, __global float* bs
)
{
    __local float* h_shr_ptr   = 0;                         // N_SIZE
    __local float* x_shr_ptr   = h_shr_ptr + N_SIZE;        // N_SIZE
    __local float* swp_shr_ptr = x_shr_ptr + N_SIZE;        // NUM_WG_THDS

    int gid = get_global_id(0);
    int lid = get_local_id(0);

    // gate computed by this thread
    int gate = (gid & 0x00c0) >> 6;
    //int lid_swp_gate_f = (lid & 0x00c0) | 0x0000;
    int lid_swp_gate_i = (lid & 0x00c0) | 0x0040;
    int lid_swp_gate_c = (lid & 0x00c0) | 0x0080;
    int lid_swp_gate_o = (lid & 0x00c0) | 0x00c0;

    int lid_swp_reduce = lid ^ 0x0020;
    // bid is the remapped gid that excludes gate bits and blocks for better global and shared memory coalescing
    int bid = ((gid & 0x1f00) >> 2) | ((gid & 0x0020) >> 5) | ((gid & 0x001f) << 1);

    int row = bid >> 1;
    int row_pos = bid & 0x0001;

    __global float* Wh_ptr = Whs + (gate * MAT_SIZE) + (bid * NUM_ELEMS_PER_BLK);
    __global float* Wx_ptr = Wxs + (gate * MAT_SIZE) + (bid * NUM_ELEMS_PER_BLK);
    __global float* b_ptr = bs + (gate * N_SIZE);
    __global float* x_ptr = x;

    /* bring matrix into registers */
    __global uint8x128* Wh_vec_ptr = (__global uint8x128*) Wh_ptr;
    uint8x128 Wh_vreg_0 = Wh_vec_ptr[0];
    uint8x128 Wh_vreg_1 = Wh_vec_ptr[1];
    uint8x128 Wh_vreg_2 = Wh_vec_ptr[2];
    uint8x128 Wh_vreg_3 = Wh_vec_ptr[3];
    uint8x128 Wh_vreg_4 = Wh_vec_ptr[4];
    uint8x128 Wh_vreg_5 = Wh_vec_ptr[5];
    uint8x128 Wh_vreg_6 = Wh_vec_ptr[6];
    uint8x128 Wh_vreg_7 = Wh_vec_ptr[7];
    __global uint8x128* Wx_vec_ptr = (__global uint8x128*) Wx_ptr;
    uint8x128 Wx_vreg_0 = Wx_vec_ptr[0];
    uint8x128 Wx_vreg_1 = Wx_vec_ptr[1];
    uint8x128 Wx_vreg_2 = Wx_vec_ptr[2];
    uint8x128 Wx_vreg_3 = Wx_vec_ptr[3];
    uint8x128 Wx_vreg_4 = Wx_vec_ptr[4];
    uint8x128 Wx_vreg_5 = Wx_vec_ptr[5];
    uint8x128 Wx_vreg_6 = Wx_vec_ptr[6];
    uint8x128 Wx_vreg_7 = Wx_vec_ptr[7];

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
        dot_fp32(res, Wh_vreg_0, h_shr_vec_ptr, 0);
        dot_fp32(res, Wh_vreg_1, h_shr_vec_ptr, 1);
        dot_fp32(res, Wh_vreg_2, h_shr_vec_ptr, 2);
        dot_fp32(res, Wh_vreg_3, h_shr_vec_ptr, 3);
        dot_fp32(res, Wh_vreg_4, h_shr_vec_ptr, 4);
        dot_fp32(res, Wh_vreg_5, h_shr_vec_ptr, 5);
        dot_fp32(res, Wh_vreg_6, h_shr_vec_ptr, 6);
        dot_fp32(res, Wh_vreg_7, h_shr_vec_ptr, 7);
        __local uint8x128* x_shr_vec_ptr = (__local uint8x128*) (x_shr_ptr + (row_pos * NUM_ELEMS_PER_BLK));
        dot_fp32(res, Wx_vreg_0, x_shr_vec_ptr, 0);
        dot_fp32(res, Wx_vreg_1, x_shr_vec_ptr, 1);
        dot_fp32(res, Wx_vreg_2, x_shr_vec_ptr, 2);
        dot_fp32(res, Wx_vreg_3, x_shr_vec_ptr, 3);
        dot_fp32(res, Wx_vreg_4, x_shr_vec_ptr, 4);
        dot_fp32(res, Wx_vreg_5, x_shr_vec_ptr, 5);
        dot_fp32(res, Wx_vreg_6, x_shr_vec_ptr, 6);
        dot_fp32(res, Wx_vreg_7, x_shr_vec_ptr, 7);

        /* reduce */
        // this is within a wavefront so no syncing is necessary
        swp_shr_ptr[lid] = res;
        res += swp_shr_ptr[lid_swp_reduce];

        /* add bias */
        res += load_if_zero_fp32(0.f, row_pos, &b_ptr[row], 0);

        /* activate */
        //switch (gate) {
        //    case GATE_FORGET:
        //        res = sigmoid_fp32(res);
        //    break;

        //    case GATE_INPUT:
        //        res = sigmoid_fp32(res);
        //    break;

        //    case GATE_CANDIDATE:
        //        res =    tanh_fp32(res);
        //    break;

        //    case GATE_OUTPUT:
        //        res = sigmoid_fp32(res);
        //    break;
        //}
        if (gate == GATE_CANDIDATE) {
            res =    tanh_fp32(res);
        }
        else {
            res = sigmoid_fp32(res);
        }

        swp_shr_ptr[lid] = res;

        local_sync();

        float tmp_c = 0.f;
        float tmp_h = 0.f;

        // only do this in one gate's set of threads
        if (gate == GATE_FORGET) {
            /* put gates together */
            //float res_f = swp_shr_ptr[lid_swp_gate_f];
            float res_f = res;
            float res_i = swp_shr_ptr[lid_swp_gate_i];
            float res_c = swp_shr_ptr[lid_swp_gate_c];
            float res_o = swp_shr_ptr[lid_swp_gate_o];

            tmp_c = c[row] * res_f + res_i * res_c;
            tmp_h = tanh_fp32(tmp_c) * res_o;
        }

        global_sync();

        if (gate == GATE_FORGET) {
            /* store result */
            store_if_zero_fp32(tmp_c, row_pos, &c[row], 0);
            store_if_zero_fp32(tmp_h, row_pos, &h[row], 0);
        }

        /* next input */
        x_ptr += N_SIZE;

        global_sync();
    }
}
