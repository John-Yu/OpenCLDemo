#define __CL_ENABLE_EXCEPTIONS
#include "openCLGEMM.h"

static const std::string kernelSource =
#include "gemm_8x4.opencl"
;

//test
void openCLGEMM_test(unsigned char *bufIn, unsigned char *bufOut, int *info) {

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
