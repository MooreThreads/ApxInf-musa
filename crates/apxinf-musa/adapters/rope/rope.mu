#include <musa_runtime.h>
#include <musa_bf16.h>
#include <cmath>
#include <cstdint>
using B = __mt_bfloat16;

__global__ void rotate(const B* a,B* o,const uint32_t* p,int n,int h,int d,float theta,int mode,int sh,int sw){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n/2)return;
 int pair=i%(d/2), base=(i/(d/2))*d, token=i/(h*d/2),axis=0, freq=pair;
 if(mode==3) {if(pair%3==1 && pair<sh*3)axis=1;else if(pair%3==2 && pair<sw*3)axis=2;}
 if(mode==2){axis=pair/(d/4);freq=pair%(d/4);}
 float angle=p[token*mode+axis]*powf(theta,-2.f*freq/(mode==2?d/2:d));
 float c=cosf(angle),s=sinf(angle),x=__bfloat162float(a[base+pair]),y=__bfloat162float(a[base+pair+d/2]);
 o[base+pair]=__float2bfloat16(x*c-y*s);o[base+pair+d/2]=__float2bfloat16(x*s+y*c);
}
extern "C" musaError_t apxinf_musa_portable_rope(const void*a,void*o,const void*p,int n,int h,int d,float theta,int mode,int sh,int sw,void*stream){rotate<<<(n/2+255)/256,256,0,(musaStream_t)stream>>>((const B*)a,(B*)o,(const uint32_t*)p,n,h,d,theta,mode,sh,sw);return musaGetLastError();}
