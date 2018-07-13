__kernel 
void vecadd(__global int *A, __global int *B, __global int *C)
{
	//워크 아이템의 유니크 ID를 얻는다.
	int idx = get_global_id(0);

	//A와 B의 해당 위치의 데이터를 덧셈하여 C에 저장함
	C[idx] = A[idx] + B[idx];
}