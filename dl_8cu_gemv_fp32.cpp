#include "FGPUlib.c"

/*
 * y = A * x
 * blocked in 2**12 rows x 2**9 columns
 * in a block, each thread is assigned a different row sharing the same columns
 * must be aligned to block
 * for 4096 threads
 */
#define ROW_BLK_SIZE (1 << 12)      // 2**12 rows per block
#define COL_BLK_SIZE (1 << 9)       // 2**9 cols per block

__kernel void gemv_fp32(__global float* y, __global float* x, __global float* A, int num_row_blks, int num_col_blks)
{
    __local float* x_shr_ptr = 0;   // COL_BLK_SIZE

    int gid = get_global_id(0);
    int lid = get_local_id(0);

    int row_size = COL_BLK_SIZE * num_col_blks;
    __global float* A_ptr = A + gid * row_size;
    __global float* x_ptr = x;
    __global float* y_ptr = y;

    for (int i = 0; i < num_row_blks; ++i) {
        // initialize row sum
        float res = 0.f;

        for (int j = 0; j < num_col_blks; ++j) {
            // load matrix A block
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

            local_sync();

            // load vector x block
            x_shr_ptr[lid] = x_ptr[lid];

            local_sync();

            // dot product
            __local uint8x128* x_shr_vec_ptr = (__local uint8x128*) x_shr_ptr;
            dot_fp32(res, A_vreg_0,  x_shr_vec_ptr, 0 );
            dot_fp32(res, A_vreg_1,  x_shr_vec_ptr, 1 );
            dot_fp32(res, A_vreg_2,  x_shr_vec_ptr, 2 );
            dot_fp32(res, A_vreg_3,  x_shr_vec_ptr, 3 );
            dot_fp32(res, A_vreg_4,  x_shr_vec_ptr, 4 );
            dot_fp32(res, A_vreg_5,  x_shr_vec_ptr, 5 );
            dot_fp32(res, A_vreg_6,  x_shr_vec_ptr, 6 );
            dot_fp32(res, A_vreg_7,  x_shr_vec_ptr, 7 );
            dot_fp32(res, A_vreg_8,  x_shr_vec_ptr, 8 );
            dot_fp32(res, A_vreg_9,  x_shr_vec_ptr, 9 );
            dot_fp32(res, A_vreg_10, x_shr_vec_ptr, 10);
            dot_fp32(res, A_vreg_11, x_shr_vec_ptr, 11);
            dot_fp32(res, A_vreg_12, x_shr_vec_ptr, 12);
            dot_fp32(res, A_vreg_13, x_shr_vec_ptr, 13);
            dot_fp32(res, A_vreg_14, x_shr_vec_ptr, 14);
            dot_fp32(res, A_vreg_15, x_shr_vec_ptr, 15);

            // increment A and x pointers to next column block
            A_ptr += COL_BLK_SIZE;
            x_ptr += COL_BLK_SIZE;
        }

        // store vector y block
        y_ptr[gid] = res;

        // increment A, x, and y pointers to next row block
        A_ptr += (ROW_BLK_SIZE - 1) * row_size;
        x_ptr = x;
        y_ptr += ROW_BLK_SIZE;
    }
}
