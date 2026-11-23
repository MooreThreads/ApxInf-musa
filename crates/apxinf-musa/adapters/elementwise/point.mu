#include <musa_runtime.h>
#include <musa_bf16.h>
#include <cmath>
#include <cstdint>
using B = __mt_bfloat16;

__global__ void point(const B* a,const B* b,B* o,int n,int op,float f) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float x=__bfloat162float(a[i]);float y=b?__bfloat162float(b[i]):0;
 o[i]=__float2bfloat16(op==0?x+y:op==1?x*y:x*f);
}
extern "C" musaError_t apxinf_musa_portable_point(const void*a,const void*b,void*o,int n,int op,float f,void* stream){point<<<(n+255)/256,256,0,(musaStream_t)stream>>>((const B*)a,(const B*)b,(B*)o,n,op,f);return musaGetLastError();}
