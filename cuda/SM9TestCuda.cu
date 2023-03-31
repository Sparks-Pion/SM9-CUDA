#include "SM9Cuda.cuh"

float SM9TEST_INIT_CUDA(int N, int numBlock, int blockSize, struct SM9_Para *gpu_para)
{
    cudaEvent_t start, end;
    float time, res = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Init_CUDA<<<numBlock, blockSize>>>(N, gpu_para);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nInit_CUDA计时结果: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_GenerateSignKey_CUDA<<<numBlock, blockSize>>>(N, gpu_para);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nGenerateSignKey_CUDA计时结果: %.6f ms\n", time);
    res += time;

    return res;
}

float SM9TEST_SIGN_CUDA(int N, int numBlock, int blockSize, struct SM9_Para *gpu_para)
{
    cudaEvent_t start, end;
    float time, res = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Sign_Init_CUDA<<<numBlock, blockSize>>>(N, gpu_para);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Sign_Init_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Sign_Step1_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step1:g = e(P1, Ppub-s)
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Sign_Step1_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Sign_Step2_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step2:calculate w=g(r)
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Sign_Step2_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Sign_Step3_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step3:calculate h=H2(M||w,N)
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Sign_Step3_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Sign_Step4_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step4:l=(r-h)mod N
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Sign_Step4_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Sign_Step5_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step5:S=[l]dSA=(xS,yS)
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Sign_Step5_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Sign_Finish_CUDA<<<numBlock, blockSize>>>(N, gpu_para);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Sign_Finish_CUDA耗时: %.6f ms\n", time);
    res += time;

    return res;
}

//struct SM9_Para cpu_para[1024];

float SM9TEST_VERIFY_CUDA(int N, int numBlock, int blockSize, struct SM9_Para *gpu_para)
{
    cudaEvent_t start, end;
    float time, res = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Verify_Init_CUDA<<<numBlock, blockSize>>>(N, gpu_para);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Verify_Init_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Verify_Step1_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step 1:test if h in the rangge [1,N-1]
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Verify_Step1_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Verify_Step2_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step 2:test if S is on G1
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Verify_Step2_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Verify_Step3_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step3:g = e(P1, Ppub-s)
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Verify_Step3_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Verify_Step4_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step4:calculate t=g^h
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Verify_Step4_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Verify_Step5_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step5:calculate h1=H1(IDA||hid,N)
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Verify_Step5_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Verify_Step6_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step6:P=[h1]P2+Ppubs
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Verify_Step6_CUDA耗时: %.6f ms\n", time);
    res += time;

//     for (int i = 0; i < N; i++)
//         ParaMemcpy2Host(cpu_para + i, gpu_para + i);
//     cudaDeviceReset();
//     cudaDeviceSetLimit(cudaLimitMallocHeapSize, N * CUDA_HEAP_SIZE);
//     cudaMalloc((void **)&gpu_para, N * sizeof(struct SM9_Para));
//     for (int i = 0; i < N; i++)
//         ParaMemcpy2Device(gpu_para + i, cpu_para + i);

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Verify_Step7_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step7:u=e(S1,P)
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Verify_Step7_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Verify_Step8_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step8:w=u*t
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Verify_Step8_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Verify_Step9_CUDA<<<numBlock, blockSize>>>(N, gpu_para); // Step9:h2=H2(M||w,N)
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Verify_Step9_CUDA耗时: %.6f ms\n", time);
    res += time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    SM9_Verify_Finish_CUDA<<<numBlock, blockSize>>>(N, gpu_para);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("\nSM9_Verify_Finish_CUDA耗时: %.6f ms\n", time);
    res += time;

    return res;
}

int SM9TEST_CUDA(int N, int blockSize, struct SM9_Para *para)
{
//    struct SM9_Para *cpu_para; // malloc(Heap//    cpu_para = (struct SM9_Para *)malloc(N * sizeof(SM9_Para));) instead of cpu_para[N](Stack)

//    for (int i = 0; i < N; i++)
//        memcpy(cpu_para + i, para, sizeof(SM9_Para));

    // 设置显存堆大小为CUDA_HEAP_SIZE
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, N * CUDA_HEAP_SIZE);

    // =====计时开始=====
    cudaEvent_t copy_start, copy_end;
    cudaEventCreate(&copy_start);
    cudaEventCreate(&copy_end);
    cudaEventRecord(copy_start, 0);
    // =====申请显存并拷贝参数开始=====
    struct SM9_Para *gpu_para;
    cudaMalloc((void **)&gpu_para, N * sizeof(struct SM9_Para));
    for (int i = 0; i < N; i++)
         cudaMemcpy(gpu_para + i, para, sizeof(struct SM9_Para), cudaMemcpyHostToDevice);
//        ParaMemcpy2Device(gpu_para + i, cpu_para + i);

    // printf("%s\n", para->SM9_q);
    // =====申请显存并拷贝参数结束=====
    // =====计时结束=====
    cudaEventRecord(copy_end, 0);
    cudaEventSynchronize(copy_end);
    float copy_time;
    cudaEventElapsedTime(&copy_time, copy_start, copy_end);
    printf("\ngpu显存拷贝耗时: %.6f ms\n", copy_time);

    int numBlock = (N + blockSize - 1) / blockSize;

    { // SM9TEST_INIT_CUDA
        SM9TEST_INIT_CUDA(N, numBlock, blockSize, gpu_para);
    }

//    for (int i = 0; i < N; i++)
//        ParaMemcpy2Host(cpu_para + i, gpu_para + i);
//    cudaDeviceReset();
//    cudaDeviceSetLimit(cudaLimitMallocHeapSize, N * CUDA_HEAP_SIZE);
//    cudaMalloc((void **)&gpu_para, N * sizeof(struct SM9_Para));
//    for (int i = 0; i < N; i++)
//        ParaMemcpy2Device(gpu_para + i, cpu_para + i);
    
//    Print_Para(cpu_para);

    { // SM9TEST_SIGN_CUDA
        float time = SM9TEST_SIGN_CUDA(N, numBlock, blockSize, gpu_para);

        printf("\nsign_cuda计时结果: %.6f ms\n", time);
    }

//     for (int i = 0; i < N; i++)
//         ParaMemcpy2Host(cpu_para + i, gpu_para + i);
//
////     Print_Para(cpu_para + i);
//
//     cudaDeviceReset();
//     cudaDeviceSetLimit(cudaLimitMallocHeapSize, N * CUDA_HEAP_SIZE);
//     cudaMalloc((void **)&gpu_para, N * sizeof(struct SM9_Para));
//     for (int i = 0; i < N; i++)
//         ParaMemcpy2Device(gpu_para + i, cpu_para + i);

    { // SM9TEST_VERIFY_CUDA
        float time = SM9TEST_VERIFY_CUDA(N, numBlock, blockSize, gpu_para);

        printf("\nverify_cuda计时结果: %.6f ms\n", time);
    }
    // ParaMemcpy(&cpu_para, gpu_para, cudaMemcpyDeviceToHost);
    // cudaDeviceReset();

    return 0;
}
