#define __CL_ENABLE_EXCEPTIONS
#include "openCLGEMM.h"
#include <cstdio>
#include <chrono>
#include <vector>
#include <string>

#include <qml.h>

static const std::string kernelSource =
#include "gemm_8x4.opencl"
;

class COpen_CL {
public:
    cl::Context  m_context;
    cl::CommandQueue  m_queue;
    cl::Kernel  m_kernel;
    bool isReady;
    COpen_CL() { isReady = false;}
    void init();
};
void COpen_CL::init()
{
    // init only once
    if(isReady) return ;
    cl_int err = CL_SUCCESS;
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "Platform size 0\n";
            return;
        }

        cl_context_properties properties[] =
                {CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms[0])(), 0};
        m_context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

        std::vector<cl::Device> devices = m_context.getInfo<CL_CONTEXT_DEVICES>();
        m_queue = cl::CommandQueue(m_context, devices[0], 0, &err);
        if(err != CL_SUCCESS) return;

        cl::Program::Sources source(1, std::make_pair(kernelSource.c_str(),
                                                      kernelSource.length() + 1));
        cl::Program program(m_context, source);
        const char *options = "-cl-fast-relaxed-math";
        program.build(devices, options);

        m_kernel = cl::Kernel(program, "matmul_8x4_blocks", &err);
        if(err != CL_SUCCESS) return;
        isReady = true;
    }
    catch (cl::Error err) {
        LOGE("ERROR: %s\n", err.what());
    }
}

COpen_CL g_open_cl;

static size_t ceilMultiple(size_t a, size_t b)
{
    if (a % b == 0) {
        return a;
    }

    auto ret = a + (b - a % b);
    return ret;
}

static void cl_cblas_sgemm(const bool isRowMajor, const bool isTRANSA, const bool isTRANSB, const size_t M, const size_t N, const size_t K, const float *A,
                        const size_t LDA, const float *B, const size_t LDB, float *C, const size_t LDC)
{
    if(!g_open_cl.isReady) return ;
    const auto k_ceil = ceilMultiple(K, 4);
    const auto n_ceil = ceilMultiple(N, 4);
    const auto A_vm_size =  M  * k_ceil * sizeof(float);
    const auto B_vm_size =  n_ceil  * k_ceil * sizeof(float);
    const auto C_vm_size =  M  * n_ceil * sizeof(float);

    auto A_zeros = std::vector<float>(A_vm_size);
    auto B_zeros = std::vector<float>(B_vm_size);
    auto C_zeros = std::vector<float>(C_vm_size);
    //ZeroPad A
    for(auto row = 0; row< M; row++)
        for(auto col =0; col < K; col++)
        {
            auto i = row * k_ceil + col;
            auto j = row * LDA + col;
            if((isRowMajor && isTRANSA) || (!isRowMajor && !isTRANSA)) {
                    j = col * LDA + row;
            }
            A_zeros[i] = A[j];
        }
        //ZeroPad B
    for(auto row = 0; row< K; row++)
        for(auto col =0; col < N; col++)
        {
            auto i = row * n_ceil + col;
            auto j = row * LDB + col;
            if((isRowMajor && isTRANSB) || (!isRowMajor && !isTRANSB)) {
                j = col * LDB + row;
            }
            B_zeros[i] = B[j];
        }
    // C =  A x B
    cl_int err = CL_SUCCESS;
    try {
/*
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "Platform size 0\n";
            return;
        }

        cl_context_properties properties[] =
                {CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);

        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::CommandQueue queue(context, devices[0], 0, &err);

        cl::Program::Sources source(1, std::make_pair(kernelSource.c_str(),
                                                      kernelSource.length() + 1));
        cl::Program program(context, source);
        const char *options = "-cl-fast-relaxed-math";
        program.build(devices, options);

        cl::Kernel kernel(program, "matmul_8x4_blocks", &err);
*/
        cl::Context & context = g_open_cl.m_context;
        cl::Kernel & kernel = g_open_cl.m_kernel;
        cl::CommandQueue & queue = g_open_cl.m_queue;

        cl::Buffer bufferIn = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         A_vm_size, (void *) A_zeros.data(), &err);
        cl::Buffer bufferOut = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                          C_vm_size, (void *) C_zeros.data(), &err);
        //CL_RGBA has 4 color channles, so use (width / 4)
        cl::Image2D img = cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      cl::ImageFormat(CL_RGBA, CL_FLOAT), n_ceil / 4, k_ceil,
                                      0, (void *)B_zeros.data());

        kernel.setArg(0, bufferIn);  //A
        kernel.setArg(1, k_ceil); //lda
        kernel.setArg(2, bufferOut); //C
        kernel.setArg(3, n_ceil); //ldc
        kernel.setArg(4, M);
        kernel.setArg(5, n_ceil);
        kernel.setArg(6, k_ceil);
        kernel.setArg(7, img); //B

        cl::Event event;

        queue.enqueueNDRangeKernel(kernel,
                                   cl::NullRange,
                                   cl::NDRange(n_ceil/4, (M+7)/8),     //be careful
                                   cl::NullRange,
                                   NULL,
                                   &event);

        queue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, C_vm_size, (void *) C_zeros.data());
        queue.finish();

        for(auto row = 0; row< M; row++)
            for(auto col =0; col < N; col++)
            {
                auto i = row * n_ceil + col;
                auto j = row * LDC + col;
                if(!isRowMajor) {
                    j = col * LDC + row;
                }
                C[j]= C_zeros[i] ;
            }
    }
    catch (cl::Error err) {
        LOGE("ERROR: %s\n", err.what());
    }
}
static void openCLGEMM_test1()
{
    // Example SGEMM arguments
    const size_t m = 224;
    const size_t n = 25;
    const size_t k = 224;
    auto a_ld = k;
    const auto b_ld = n;
    const auto c_ld = n;

    // Populate host matrices with some example data
    auto host_a = std::vector<float>(m*k);
    auto host_b = std::vector<float>(n*k);
    auto host_c = std::vector<float>(m*n);
    //for (auto &item: host_a) { item = 1.0f; }
    //isRowMajor
    for(auto i = 0; i< m; i++)
        for(auto j =0; j<k; j++)
            host_a[i*k+j] = (i+1) * 1.0f;
    for (auto &item: host_b) { item = 2.0f; }
    for (auto &item: host_c) { item = 0.0f; }
    g_open_cl.init();
/*
    // Start the timer
    auto start_time = std::chrono::steady_clock::now();
    for(auto i=0;i<36;i++)
        cl_cblas_sgemm(true, false, false, m, n, k, host_a.data(), a_ld, host_b.data(), b_ld, host_c.data(), c_ld);
    auto elapsed_time = std::chrono::steady_clock::now() - start_time;
    auto time_ms = std::chrono::duration<double,std::milli>(elapsed_time).count();
    LOGI("OpenCL code on the GPU took %.3lf ms\n\n", time_ms);
*/
    //for (auto &item: host_c) { item = 0.0f; }
    {
        // Start the timer
        auto start_time = std::chrono::steady_clock::now();
        for(auto i=0;i<36;i++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, host_a.data(), a_ld, host_b.data(), b_ld, 0.0, host_c.data(), c_ld);
        auto elapsed_time = std::chrono::steady_clock::now() - start_time;
        auto time_ms = std::chrono::duration<double,std::milli>(elapsed_time).count();
        LOGI("QML cblas_sgemm() took %.3lf ms\n\n", time_ms);

    }

    //isTRANSA == true
    for(auto i = 0; i< m; i++)
        for(auto j =0; j<k; j++)
            host_a[i+j*m] = (i+1) * 1.0f;

    for (auto &item: host_c) { item = 0.0f; }
    a_ld = m; //
    cl_cblas_sgemm(true, true, false, m, n, k, host_a.data(), a_ld, host_b.data(), b_ld, host_c.data(), c_ld);

}
//test
static void openCLGEMM_test(unsigned char *bufIn, unsigned char *bufOut, int *info) {

    LOGI("\n\nStart openCLGEMM (i.e., OpenCL on the GPU)");

    int width = 16; //info[0];
    int height = 16; //info[1];
    unsigned int imageSize = width * height *  sizeof(float);
    float A[256], B[256],C[256];
    for(int i =0; i< width; i++)
        for(int j = 0; j < height; j++)
        {
            A[i*height + j] = 1.0f;
            B[i*height + j] = 2.0f;
        }

    cl_int err = CL_SUCCESS;
    try {

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "Platform size 0\n";
            return;
        }

        cl_context_properties properties[] =
                {CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);

        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::CommandQueue queue(context, devices[0], 0, &err);

        cl::Program::Sources source(1, std::make_pair(kernelSource.c_str(),
                                                      kernelSource.length() + 1));
        cl::Program program(context, source);
        const char *options = "-cl-fast-relaxed-math";
        program.build(devices, options);

        cl::Kernel kernel(program, "matmul_8x4_blocks", &err);
        cl::Buffer bufferIn = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         imageSize, (void *) &(A[0]), &err);
        cl::Buffer bufferOut = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                          imageSize, (void *) &C[0], &err);
        //CL_RGBA has 4 color channles, so use (width / 4)
        cl::Image2D img = cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      cl::ImageFormat(CL_RGBA, CL_FLOAT), width / 4, height,
                                      0, (void *)&B[0]);

        kernel.setArg(0, bufferIn);  //A
        kernel.setArg(1, width);
        kernel.setArg(2, bufferOut); //C
        kernel.setArg(3, height);
        kernel.setArg(4, width);
        kernel.setArg(5, height);
        kernel.setArg(6, width);
        kernel.setArg(7, img); //B

        cl::Event event;

        clock_t startTimer1, stopTimer1;
        startTimer1 = clock();

        queue.enqueueNDRangeKernel(kernel,
                                   cl::NullRange,
                                   cl::NDRange(width/4, (height+7)/8),     //be careful
                                   cl::NullRange,
                                   NULL,
                                   &event);


        queue.finish();

        stopTimer1 = clock();
        double elapse = 1000.0 * (double) (stopTimer1 - startTimer1) / (double) CLOCKS_PER_SEC;
        info[2] = (int) elapse;
        LOGI("OpenCL code on the GPU took %g ms\n\n",
             1000.0 * (double) (stopTimer1 - startTimer1) / (double) CLOCKS_PER_SEC);

        queue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, imageSize, (void *) &C[0]);

    }
    catch (cl::Error err) {
        LOGE("ERROR: %s\n", err.what());
    }
    return;
}

void openCLGEMM(unsigned char *bufIn, unsigned char *bufOut, int *info) {

    LOGI("\n\nStart openCLGEMM (i.e., OpenCL on the GPU)");
    openCLGEMM_test1();
    return;

    int width = info[0];
    int height = info[1];
    unsigned int imageSize = width * height * 4 * sizeof(cl_uchar); //number of BYTE

    cl_int err = CL_SUCCESS;
    try {

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "Platform size 0\n";
            return;
        }

        cl_context_properties properties[] =
                {CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);

        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::CommandQueue queue(context, devices[0], 0, &err);

        cl::Program::Sources source(1, std::make_pair(kernelSource.c_str(),
                                                      kernelSource.length() + 1));
        cl::Program program(context, source);
        const char *options = "-cl-fast-relaxed-math";
        program.build(devices, options);

        cl::Kernel kernel(program, "matmul_8x4_blocks", &err);
        cl::Buffer bufferIn = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         imageSize, (void *) &bufIn[0], &err);
        cl::Buffer bufferOut = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                          imageSize, (void *) &bufOut[0], &err);

        cl::Image2D img2D = cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                               cl::ImageFormat(CL_RGBA, CL_FLOAT), width / 4, height,
                                               0, (void *) &bufIn[0], &err);

        kernel.setArg(0, bufferIn);  //A
        kernel.setArg(1, width);
        kernel.setArg(2, bufferOut); //C
        kernel.setArg(3, height);
        kernel.setArg(4, width);
        kernel.setArg(5, height);
        kernel.setArg(6, width);
        kernel.setArg(7, img2D); //B

        cl::Event event;

        clock_t startTimer1, stopTimer1;
        startTimer1 = clock();

// 		one time
        queue.enqueueNDRangeKernel(kernel,
                                   cl::NullRange,
                                   cl::NDRange(width/4, (height+7)/8),
                                   cl::NullRange,
                                   NULL,
                                   &event);


        queue.finish();

        stopTimer1 = clock();
        double elapse = 1000.0 * (double) (stopTimer1 - startTimer1) / (double) CLOCKS_PER_SEC;
        info[2] = (int) elapse;
        LOGI("OpenCL code on the GPU took %g ms\n\n",
             1000.0 * (double) (stopTimer1 - startTimer1) / (double) CLOCKS_PER_SEC);

        queue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, imageSize, bufOut); //C
    }
    catch (cl::Error err) {
        LOGE("ERROR: %s\n", err.what());
    }
    return;
}
