#include <musa_runtime.h>
#include <musa_bf16.h>
#include <cmath>
#include <cstdint>
using B = __mt_bfloat16;

__global__ void gather(const B* a,const uint32_t* ids,B* o,int n,int d){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*d)o[i]=a[ids[i/d]*d+i%d];}
extern "C" musaError_t apxinf_musa_portable_embedding(const void*a,const void*p,void*o,int n,int d,void*stream){gather<<<(n*d+255)/256,256,0,(musaStream_t)stream>>>((const B*)a,(const uint32_t*)p,(B*)o,n,d);return musaGetLastError();}
