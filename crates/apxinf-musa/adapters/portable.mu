#include <musa_runtime.h>
#include <musa_bf16.h>
#include <cmath>
#include <cstdint>
namespace {
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
// One block per query/head. Shared reduction avoids assumptions about MP22
// hardware warp size. Online softmax needs no quadratic score allocation.
__global__ void attend(const B* q,const B* k,const B* v,B* o,int nq,int nk,int h,int kh,int d,int causal){
 int t=blockIdx.x,head=blockIdx.y,j=threadIdx.x;
 __shared__ float dots[128];__shared__ float alpha,beta,denom,maximum;
 float acc=0; if(j==0){denom=0;maximum=-INFINITY;}__syncthreads();
 int end=causal?nk-nq+t+1:nk;
 for(int s=0;s<end;s++){
  int off=(s*kh+head/(h/kh))*d+j;
  dots[j]=j<d?__bfloat162float(q[(t*h+head)*d+j])*__bfloat162float(k[off]):0;__syncthreads();
  for(int stride=64;stride;stride/=2){if(j<stride)dots[j]+=dots[j+stride];__syncthreads();}
  if(j==0){float score=dots[0]/sqrtf(float(d));float m=fmaxf(maximum,score);alpha=expf(maximum-m);beta=expf(score-m);denom=denom*alpha+beta;maximum=m;}__syncthreads();
  if(j<d)acc=acc*alpha+beta*__bfloat162float(v[off]);__syncthreads();
 }
 if(j<d)o[(t*h+head)*d+j]=__float2bfloat16(acc/denom);
}
}
extern "C" musaError_t apxinf_musa_portable_rope(const void*a,void*o,const void*p,int n,int h,int d,float theta,int mode,int sh,int sw,void*stream){rotate<<<(n/2+255)/256,256,0,(musaStream_t)stream>>>((const B*)a,(B*)o,(const uint32_t*)p,n,h,d,theta,mode,sh,sw);return musaGetLastError();}
extern "C" musaError_t apxinf_musa_portable_attention(const void*q,const void*k,const void*v,void*o,int nq,int nk,int h,int kh,int d,int causal,void*stream){attend<<<dim3(nq,h),128,0,(musaStream_t)stream>>>((const B*)q,(const B*)k,(const B*)v,(B*)o,nq,nk,h,kh,d,causal);return musaGetLastError();}
