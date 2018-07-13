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

	const int Elements = 2048; //Elements�� ����� ����

	size_t datasize = sizeof(int)*Elements;

	A = (int*)malloc(datasize);
	B = (int*)malloc(datasize);
	C = (int*)malloc(datasize);

	int i;
	for (i = 0; i < Elements; i++) {
		A[i] = i;
		B[i] = i;
	}

	//�� API ȣ���� ���üũ�� ���� �����
	cl_int status;

	// �÷��� ���� ������
	cl_uint numPlatforms = 0;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	error_checker("clGetPlatformIDs", status);

	//�� �÷����� ���� ����� ������ �Ҵ�
	cl_platform_id* platforms = NULL;
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatforms);

	//�÷��� ������ �����ٰ� platforms�� ����
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	error_checker("clGetPlatformIDs", status);

	//�� ����̽��� ���� ������
	cl_uint numDevices = 0;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	error_checker("clGeDeviceIDs", status);

	//�� ����̽��� ���� ����� ������ �Ҵ�
	cl_device_id* devices = NULL;
	devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);

	//����̽� ������ ������
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	error_checker("clGeDeviceIDs", status);

	//���ý�Ʈ�� �����ϰ� ����̽��� �����Ŵ
	//���ý�Ʈ�� ����
	//CreateContext �Լ��� ���� �ϴ� ���� ���� �ǹ�
	//HW������ ��� �����ϴ°�
	cl_context context;
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	error_checker("clCreateContext", status);

	//��ɾ� ť�� �����ϰ� ����̽��� �����Ŵ
	//CreateQueue�� ���� �Է� ���� �ǹ�
	//Queue�� �����ϱ� ���ؼ� �ʿ��Ѱ�, Context���� ����
	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
	error_checker("clCreateCommandQueue", status);

	//ȣ��Ʈ �迭 A�� ���� �����͸� �����ϴ� ���� ��ü�� ����
	cl_mem bufA;
	bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
	error_checker("clCreateBuffer for A", status);

	//ȣ��Ʈ �迭 B�κ��� �����͸� �����ϴ� ���� ��ü�� ����
	cl_mem bufB;
	bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
	error_checker("clCreateBuffer for B", status);

	//ȣ��Ʈ �迭 C�κ��� �����͸� �����ϴ� ���� ��ü ����

	cl_mem bufC;
	bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	error_checker("clCreateBuffer for C", status);

	//�Է� �迭 A,B�� ����̽� ���� bufA,bufB�� �ۼ�
	status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_FALSE, 0, datasize, A, 0, NULL, NULL);
	error_checker("clEnqueueWriteBuffer for A", status);
	status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_FALSE, 0, datasize, B, 0, NULL, NULL);
	error_checker("clEnqueueWriteBuffer for B", status);

	// vectorAdd.cl �� �о source�� ������ 
	//readKernelFile�̶�� �Լ��� ���� ���
	long sizeSource;
	const char* source = readKernelFile("vectorAdd.cl", &sizeSource);

	//program�� Source���� �޾Ƽ� ����
	cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &status);
	error_checker("clCreateProgramWithSource", status);
	
	//������ ���α׷��� ������
	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

	//���� ���� Ŀ���� ����
	cl_kernel kernel;
	kernel = clCreateKernel(program, "vecadd", &status);
	error_checker("clCreateKernel", status);

	//�Է� �� ��� ���ۿ� Ŀ���� ����
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

	error_checker("clSetKernelArg", status);
	//������ ���� ��ũ�������� �ε��� ����(�۷ι� ��ũ������)�� ������
	//��ũ�׷� ũ��(���� ��ũ ������)�� �ʿ������� ������ ���� �� �ִ�. 
	size_t globalWorkSize[1];

	//��ũ�������� '�׸�'�� �ִ�. 
	globalWorkSize[0] = Elements;

	//Ŀ���� ������
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	error_checker("clEnqueueNDRangeKernel", status);

	//����̽� ��� ���ۿ��� ȣ��Ʈ ��� ���۷� �о��
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

	//OpenCL ���ҽ� ����
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufA);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufC);
	clReleaseContext(context);

	//ȣ��Ʈ ���ҽ� ����
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
