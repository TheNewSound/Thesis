__global__ void vmin4(
	int32_t &d,
	unsigned const &A,
	unsigned const &B,
	int32_t const &c){
	#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300))
		asm volatile("vmin4.s32.s32.s32.add %0, %1, %2, %3;"
                 : "=r"(d)
                 : "r"(A), "r"(B), "r"(c));
	#endif
}

__global__ void vmin4_intrinsic(
	int32_t &d,
	unsigned const &A,
	unsigned const &B,
	int32_t const &c){
	#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300))
		int8_t t[4];
		unsigned &T = reinterpret_cast<unsigned &>(t);
		T = __vmins4(A, B);
		d += t[0] + t[1] + t[2] + t[3];
	#endif
}

__global__ void vmin4_intrinsic_dp4a(
	int32_t &d,
	unsigned const &A,
	unsigned const &B,
	int32_t const &c){
	#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610))
		int32_t T = __vmins4(A, B);
		d = __dp4a(T, 0x01010101, c);
	#endif
}

__global__ void dp4a(
    int32_t &d,
    unsigned const &A,
    unsigned const &B,
    int32_t const &c){
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610))
        asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                 : "=r"(d)
                 : "r"(A), "r"(B), "r"(c));
#endif
}

__global__ void loop_unroll(
    int32_t &d,
    int8_t const a[],
    int8_t const b[],
    int32_t const &c){
    d = c;

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      d += a[k] * b[k];
    }
}


