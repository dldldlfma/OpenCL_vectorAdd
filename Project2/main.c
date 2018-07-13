#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

const char* getErrorString(cl_int error); // https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
void error_checker(char* func_name, cl_int status);
char* readKernelFile(const char* filename, long* _size);

int main() {
	printf("Hello OpenCL ! \n");

	int* A = NULL;
	int* B = NULL;
	int* C = NULL;

	const int Elements = 2048; //Elements를 상수로 선언

	size_t datasize = sizeof(int)*Elements;

	A = (int*)malloc(datasize);
	B = (int*)malloc(datasize);
	C = (int*)malloc(datasize);

	int i;
	for (i = 0; i < Elements; i++) {
		A[i] = i;
		B[i] = i;
	}

	//각 API 호출의 출력체크를 위해 사용함
	cl_int status;

	// 플랫폼 수를 가져옴
	cl_uint numPlatforms = 0;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	error_checker("clGetPlatformIDs", status);

	//각 플랫폼을 위한 충분한 공간을 할당
	cl_platform_id* platforms = NULL;
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatforms);

	//플랫폼 정보를 가져다가 platforms에 담음
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	error_checker("clGetPlatformIDs", status);

	//각 디바이스의 수를 가져옴
	cl_uint numDevices = 0;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	error_checker("clGeDeviceIDs", status);

	//각 디바이스를 위한 충분한 공간을 할당
	cl_device_id* devices = NULL;
	devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);

	//디바이스 정보를 가져옴
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	error_checker("clGeDeviceIDs", status);

	//컨택스트를 생성하고 디바이스와 연결시킴
	//컨택스트가 뭔지
	//CreateContext 함수에 들어가야 하는 변수 값의 의미
	//HW적으로 어떻게 존재하는가
	cl_context context;
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	error_checker("clCreateContext", status);

	//명령어 큐를 새성하고 디바이스와 연결시킴
	//CreateQueue에 들어가는 입력 값의 의미
	//Queue를 선언하기 위해서 필요한것, Context와의 관계
	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
	error_checker("clCreateCommandQueue", status);

	//호스트 배열 A로 부터 데이터를 포함하는 버퍼 객체를 생성
	cl_mem bufA;
	bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
	error_checker("clCreateBuffer for A", status);

	//호스트 배열 B로부터 데이터를 포함하는 버퍼 객체를 생성
	cl_mem bufB;
	bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
	error_checker("clCreateBuffer for B", status);

	//호스트 배열 C로부터 데이터를 포함하는 버퍼 객체 생성

	cl_mem bufC;
	bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	error_checker("clCreateBuffer for C", status);

	//입력 배열 A,B를 디바이스 버퍼 bufA,bufB에 작성
	status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_FALSE, 0, datasize, A, 0, NULL, NULL);
	error_checker("clEnqueueWriteBuffer for A", status);
	status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_FALSE, 0, datasize, B, 0, NULL, NULL);
	error_checker("clEnqueueWriteBuffer for B", status);

	// vectorAdd.cl 을 읽어서 source에 저장함 
	//readKernelFile이라는 함수를 만들어서 사용
	long sizeSource;
	const char* source = readKernelFile("vectorAdd.cl", &sizeSource);

	//program을 Source에서 받아서 생성
	cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &status);
	error_checker("clCreateProgramWithSource", status);
	
	//생성한 프로그램을 빌드함
	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

	//벡터 덧셈 커널을 생성
	cl_kernel kernel;
	kernel = clCreateKernel(program, "vecadd", &status);
	error_checker("clCreateKernel", status);

	//입력 및 출력 버퍼와 커널을 연결
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

	error_checker("clSetKernelArg", status);
	//실행을 위해 워크아이템의 인덱스 공간(글로벌 워크사이즈)를 정의함
	//워크그룹 크기(로컬 워크 사이즈)가 필요하지는 않지만 사용될 수 있다. 
	size_t globalWorkSize[1];

	//워크아이템의 '항목'이 있다. 
	globalWorkSize[0] = Elements;

	//커널을 실행함
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	error_checker("clEnqueueNDRangeKernel", status);

	//디바이스 출력 버퍼에서 호스트 출력 버퍼로 읽어옴
	clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);
	error_checker("clEnqueueReadBuffer", status);

	int result = 1;
	for (i = 0; i < Elements; i++) {
		if (C[i] != i + i){
			result = 0;
			break;
		}
	}

	if (result) {
		printf("Output is Correct\n");
	} else {
		printf("Output is incorrect\n");
	}

	//OpenCL 리소스 해제
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufA);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufC);
	clReleaseContext(context);

	//호스트 리소스 해제
	free(A);
	free(B);
	free(C);
	free(platforms);
	free(devices);

	return 0;
}


const char *getErrorString(cl_int error)
{
	switch (error) {
		// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}

void error_checker(char* func_name, cl_int status) {
	printf("%s error is : %s \n", func_name, getErrorString(status));
}

char* readKernelFile(const char* filename, long* _size) {

	// Open the file
	FILE* file = fopen(filename, "r");
	if (!file) {
		printf("-- Error opening file %s\n", filename);
		exit(1);
	}

	// Get its size
	fseek(file, 0, SEEK_END);
	long size = ftell(file);
	rewind(file);

	// Read the kernel code as a string
	char* source = (char *)malloc((size + 1) * sizeof(char));
	fread(source, 1, size * sizeof(char), file);
	source[size] = '\0';
	printf("\n\n %s \n\n", source);
	fclose(file);

	// Save the size and return the source string
	*_size = (size + 1);
	return source;
}
