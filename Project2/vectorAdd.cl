__kernel 
void vecadd(__global int *A, __global int *B, __global int *C)
{
	//��ũ �������� ����ũ ID�� ��´�.
	int idx = get_global_id(0);

	//A�� B�� �ش� ��ġ�� �����͸� �����Ͽ� C�� ������
	C[idx] = A[idx] + B[idx];
}