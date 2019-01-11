// Computes the matrix product C = A * B. 
//  C(m,n) = A(m,k) X B(k,n)  where k and n must be times of 4
// lda and ldc defines how much memory each row of the matrix will use,it is not necessarily equal to the horizontal dimension of the matrix; in some cases lda could be different.
// https://www.qualcomm.com/news/onq/2016/10/17/matrix-multiply-adreno-gpus-part-2-host-code-and-kernel
// version 1.0
// 2019.1.11   https://github.com/John-Yu/
// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(
// Main body of the matrix-multiplication algorithm. 
__kernel void matmul_8x4_blocks(
                           __global const float *A,
                           const int lda,
                           __global float *C,
                           const int ldc,
                           const int m,
                           const int n,
                           const int k,
                           __read_only image2d_t Bi)
{
    int gx = get_global_id(0);  // col ID of C (0 .. n/4)
    int gy = get_global_id(1);  // row ID of C (0 .. (m+7)/8)
    int gx4 = gx << 2;
    int gy8 = gy << 3;
    //number of y in one y loop, deal with m%8 != 0
    int num_y = 8; 

    if ((gx4 < n) && (gy8 < m))
    {
        float4 a[8];
        float4 b[4];
        float4 c[8];

        if(gy8 + 8 > m)  num_y = m - gy8; 
        //initializes elements of matrix C to zero.
        for (int i = 0; i < num_y; i++)
        {
            c[i] = (float4)(0.0f);
        }
        int A_y_off = gy8 * lda;

        for (int pos = 0; pos < k; pos += 4)
        {
            //read elements of matrix B through the TP/L1 with the read_imagef function.
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                b[i] = read_imagef(Bi, (int2)(gx, pos + i));
            }

            //reads of elements of matrix A from L2 directly
            int A_off = A_y_off + pos;
            #pragma unroll
            for (int i = 0; i < num_y; i++)
            {
                a[i] = vload4(0, A + A_off);
                A_off += lda;
            }

            //calculates partial dot products.
            #pragma unroll
            for (int i = 0; i < num_y; i++)
            {
                c[i] += a[i].x * b[0] + a[i].y * b[1] + a[i].z * b[2] + a[i].w * b[3];
            }
        }

        #pragma unroll
        for (int i = 0; i < num_y; i++)
        {
            int C_offs = (gy8 + i) * ldc + gx4;
            vstore4(c[i], 0, C + C_offs);
        }
    }
}

)"
// End of the C++11 raw string literal

// =================================================================================================
