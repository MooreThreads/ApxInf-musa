// Native muDNN operators. Descriptors and scratch belong to a device/stream;
// no torch objects, host fallback, or hot-path allocation is involved.
//
// Contract
// --------
// Every entry point returns **0 on success** and non-zero on failure, and
// `apxinf_musa_mudnn_last_error()` names the reason for the last failure on this
// thread. Two vocabularies meet in that `int`, and they are kept apart by
// construction rather than by luck: the vendor's `md::Status` values are
// whatever the enum says, and this adapter's own conditions are `kStatus*`
// below, starting at 1000. Before that, "the scratch buffer is the wrong size"
// reached Rust as the bare number 1005 with nothing anywhere to say what it
// meant -- the Rust side formats the code it is handed, so an unnamed code is an
// unreadable error.
//
// `mublas_adapter.mu` and the two FA adapters return their own status types;
// this one returns `int` because muDNN's own Run() does, and translating would
// mean inventing a mapping for codes this file does not own.
#include <mudnncxx/mudnn.h>
#include <musa_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "adapter_common.h"

namespace {
namespace md = musa::dnn;


// This adapter's own failure codes. 1000 and up, so they cannot collide with a
// `md::Status` value, and named, so a caller reading `status 1004` can find out
// what it means.
enum {
  kStatusCaptureStatusUnreadable = 1002,
  kStatusCaptureInProgress = 1003,
  kStatusAllocationFailed = 1004,
  kStatusScratchTooSmall = 1005,
  kStatusInvalidArgument = 1006,
  kStatusUnknown = 1007,
  kStatusDeviceQueryFailed = 1008,
};

struct StatusError { int status; };
void check(md::Status status) {
  if (status != md::Status::SUCCESS) throw StatusError{int(status)};
}
bool capturing(musaStream_t stream) {
  return apxinf_adapters::capturing(stream);
}
struct Scratch {
  std::vector<void*> allocations;
  std::vector<size_t> sizes;
  size_t cursor = 0;
  musaStream_t stream;
  explicit Scratch(musaStream_t s) : stream(s) {}
  ~Scratch() { for (void* p : allocations) musaFree(p); }
  md::MemoryHandler get(size_t bytes) {
    const size_t slot = cursor++;
    if (slot >= allocations.size()) {
      if (capturing(stream)) throw StatusError{kStatusCaptureInProgress};
      void* p = nullptr;
      if (musaMalloc(&p, bytes) != musaSuccess) throw StatusError{kStatusAllocationFailed};
      allocations.push_back(p); sizes.push_back(bytes);
    } else if (bytes > sizes[slot]) {
      // A later call for the same shape wanted more scratch than the first did.
      // This used to throw and leave the shape permanently unusable, because the
      // entry is keyed on the shape and never retried -- an intermittent muDNN
      // failure that could not heal. Reallocating is what the size is for.
      //
      // Refused during capture, for the reason every other first-touch
      // allocation is: the graph baked the pointer this would replace.
      if (capturing(stream)) throw StatusError{kStatusCaptureInProgress};
      void* p = nullptr;
      if (musaMalloc(&p, bytes) != musaSuccess) throw StatusError{kStatusAllocationFailed};
      musaFree(allocations[slot]);
      allocations[slot] = p;
      sizes[slot] = bytes;
    }
    return md::MemoryHandler(allocations[slot], [](void*) {});
  }
};
struct Matmul {
  md::Tensor a, w, y;
  md::MatMul operation;
  Scratch scratch;
  Matmul(int m,int n,int k,musaStream_t stream) : scratch(stream) {
    for(auto p:{&a,&w,&y}) check(p->SetType(md::Tensor::Type::BFLOAT16));
    check(a.SetNdInfo({m,k})); check(w.SetNdInfo({k,n})); check(y.SetNdInfo({m,n}));
    check(operation.SetTranspose(false,false));
    check(operation.SetComputeMode(md::MatMul::ComputeMode::TENSOR));
    check(operation.SetAlpha(1)); check(operation.SetBeta(0));
    check(operation.SetDeterministic(true));
  }
};
struct MatmulFp8 {
  md::Tensor a, w, y;
  md::MatMul operation;
  Scratch scratch;
  MatmulFp8(int m,int n,int k,musaStream_t stream) : scratch(stream) {
    check(a.SetType(md::Tensor::Type::FP8_E4M3));
    check(w.SetType(md::Tensor::Type::FP8_E4M3));
    check(y.SetType(md::Tensor::Type::BFLOAT16));
    check(a.SetNdInfo({m,k})); check(w.SetNdInfo({k,n})); check(y.SetNdInfo({m,n}));
    check(operation.SetTranspose(false,false));
    check(operation.SetComputeMode(md::MatMul::ComputeMode::TENSOR));
    check(operation.SetBeta(0));
    check(operation.SetDeterministic(true));
  }
};
// FP8 GEMM with the muDNN bias epilogue:
//   d = alpha * a * b + beta * residual + gamma * bias
// `c` is the residual operand. The bias-only entry point points `c` at the
// output and uses the inplace overload with beta = 0, so no zero buffer is
// needed. `alpha` is the static-FP8 `activation_scale * weight_scale`; the
// epilogue does bias and residual in f32 and rounds once to BF16.
struct MatmulFp8Bias {
  md::Tensor a, w, y, c, bias;
  md::MatMul operation;
  Scratch scratch;
  MatmulFp8Bias(int m,int n,int k,musaStream_t stream) : scratch(stream) {
    check(a.SetType(md::Tensor::Type::FP8_E4M3));
    check(w.SetType(md::Tensor::Type::FP8_E4M3));
    check(y.SetType(md::Tensor::Type::BFLOAT16));
    check(c.SetType(md::Tensor::Type::BFLOAT16));
    check(bias.SetType(md::Tensor::Type::BFLOAT16));
    check(a.SetNdInfo({m,k})); check(w.SetNdInfo({k,n}));
    check(y.SetNdInfo({m,n})); check(c.SetNdInfo({m,n}));
    check(bias.SetNdInfo({n}));
    check(operation.SetTranspose(false,false));
    check(operation.SetComputeMode(md::MatMul::ComputeMode::TENSOR));
    check(operation.SetDeterministic(true));
  }
};
struct Flash {
  md::Tensor q,k,v,y,lse,mask,dropout;
  md::ScaledDotProductAttention operation;
  Scratch scratch;
  void* lse_buffer = nullptr;
  Flash(int batch,int heads,int tokens,int dim,musaStream_t stream) : scratch(stream) {
    for(auto p:{&q,&k,&v,&y,&mask}) check(p->SetType(md::Tensor::Type::BFLOAT16));
    for(auto p:{&q,&k,&v,&y}) check(p->SetNdInfo({batch,heads,tokens,dim},
      {tokens*heads*dim,dim,heads*dim,1}));
    check(lse.SetType(md::Tensor::Type::FLOAT)); check(lse.SetNdInfo({batch,heads,tokens}));
    check(mask.SetNdInfo({0}));
    check(dropout.SetType(md::Tensor::Type::FLOAT)); check(dropout.SetNdInfo({0}));
    check(operation.SetBatchSize(batch)); check(operation.SetHeadsNum(heads));
    check(operation.SetEmbedDim(heads*dim)); check(operation.SetScale(1./std::sqrt(double(dim))));
    check(operation.SetCausal(false)); check(operation.SetMaskMode(false));
    check(operation.SetTraining(false)); check(operation.SetDropoutP(0));
    // Pinned, like Matmul above and for the same reason: a captured graph is
    // only worth capturing if replaying it twice gives one answer twice. The
    // vendor default is not documented as reproducible, and one run of this
    // binary against a fixed fixture has disagreed with three others at the
    // sixth significant digit of the end-to-end cosine with no input differing.
    check(operation.SetIsDeterministic(true));
    // Last, so no later `check` can throw past an owned allocation. A throwing
    // constructor does not run `~Flash` (only the members'), and a rejected
    // shape re-runs the constructor on every call -- `make_unique` never
    // inserts it -- so a leak here would repeat rather than happen once.
    if(musaMalloc(&lse_buffer,size_t(batch)*heads*tokens*sizeof(float))!=musaSuccess)
      throw StatusError{kStatusAllocationFailed};
    check(lse.SetAddr(lse_buffer));
  }
  ~Flash() { if(lse_buffer) musaFree(lse_buffer); }
};

// General BF16 flash-attention descriptor with independent Q/K lengths, GQA
// head counts, causal selection and an optional packed-QKV physical layout.
// Keep it separate from `Flash`, whose established D72 path is used by PI0.5.
struct GeneralFlash {
  md::Tensor q,k,v,y,lse,mask,dropout;
  md::ScaledDotProductAttention operation;
  Scratch scratch;
  void* lse_buffer = nullptr;
  GeneralFlash(int batch,int query_heads,int kv_heads,int query_tokens,int key_tokens,
               int dim,bool causal,bool packed_qkv,musaStream_t stream) : scratch(stream) {
    for(auto p:{&q,&k,&v,&y,&mask}) check(p->SetType(md::Tensor::Type::BFLOAT16));
    check(y.SetNdInfo({batch,query_heads,query_tokens,dim},
      {query_tokens*query_heads*dim,dim,query_heads*dim,1}));
    if(packed_qkv) {
      const int token_stride=3*query_heads*dim;
      for(auto p:{&q,&k,&v}) check(p->SetNdInfo(
        {batch,query_heads,query_tokens,dim},
        {query_tokens*token_stride,dim,token_stride,1}));
    } else {
      check(q.SetNdInfo({batch,query_heads,query_tokens,dim},
        {query_tokens*query_heads*dim,dim,query_heads*dim,1}));
      for(auto p:{&k,&v}) check(p->SetNdInfo({batch,kv_heads,key_tokens,dim},
        {key_tokens*kv_heads*dim,dim,kv_heads*dim,1}));
    }
    check(lse.SetType(md::Tensor::Type::FLOAT));
    check(lse.SetNdInfo({batch,query_heads,query_tokens}));
    check(mask.SetNdInfo({0}));
    check(dropout.SetType(md::Tensor::Type::FLOAT)); check(dropout.SetNdInfo({0}));
    check(operation.SetBatchSize(batch)); check(operation.SetHeadsNum(query_heads));
    check(operation.SetEmbedDim(query_heads*dim));
    check(operation.SetScale(1./std::sqrt(double(dim))));
    check(operation.SetCausal(causal)); check(operation.SetMaskMode(false));
    check(operation.SetMaxSeqlenQ(query_tokens));
    check(operation.SetMaxSeqlenK(key_tokens));
    check(operation.SetTraining(false)); check(operation.SetDropoutP(0));
    check(operation.SetIsDeterministic(true));
    if(musaMalloc(&lse_buffer,size_t(batch)*query_heads*query_tokens*sizeof(float))!=musaSuccess)
      throw StatusError{kStatusAllocationFailed};
    check(lse.SetAddr(lse_buffer));
  }
  ~GeneralFlash() { if(lse_buffer) musaFree(lse_buffer); }
};

// Exact fallback for Fast-WAM's block mask. Unlike Flash, muDNN Math SDPA
// consumes an explicit BOOL [Tq,Tk] mask and materializes attention
// probabilities. Q/K/V remain physically [B,T,H,D], described logically as
// [B,H,T,D], matching Torch-MUSA's MuDNNMathSDPAFwd wrapper.
struct MaskedMathAttention {
  md::Tensor q,k,v,y,mask,attn_probs,dropout;
  md::ScaledDotProductAttention operation;
  Scratch scratch;
  void* attn_probs_buffer = nullptr;
  MaskedMathAttention(int batch,int heads,int query_tokens,int key_tokens,
                      int dim,musaStream_t stream) : scratch(stream) {
    for(auto p:{&q,&k,&v,&y,&attn_probs}) check(p->SetType(md::Tensor::Type::BFLOAT16));
    check(q.SetNdInfo({batch,heads,query_tokens,dim},
      {heads*query_tokens*dim,query_tokens*dim,dim,1}));
    check(y.SetNdInfo({batch,heads,query_tokens,dim},
      {heads*query_tokens*dim,query_tokens*dim,dim,1}));
    for(auto p:{&k,&v}) check(p->SetNdInfo({batch,heads,key_tokens,dim},
      {heads*key_tokens*dim,key_tokens*dim,dim,1}));
    check(mask.SetType(md::Tensor::Type::BOOL));
    check(mask.SetNdInfo({query_tokens,key_tokens}));
    check(attn_probs.SetNdInfo({batch,heads,query_tokens,key_tokens}));
    check(dropout.SetType(md::Tensor::Type::BOOL));
    check(dropout.SetNdInfo({0}));
    check(operation.SetComputeMode(md::ScaledDotProductAttention::ComputeMode::TENSOR));
    check(operation.SetBatchSize(batch));
    check(operation.SetHeadsNum(heads));
    check(operation.SetEmbedDim(heads*dim));
    check(operation.SetScale(1./std::sqrt(double(dim))));
    check(operation.SetMaskMode(false));
    check(operation.SetCausal(false));
    check(operation.SetTraining(false));
    check(operation.SetDropoutP(0));
    check(operation.SetIsDeterministic(true));
    const size_t bytes=size_t(batch)*heads*query_tokens*key_tokens*sizeof(uint16_t);
    if(musaMalloc(&attn_probs_buffer,bytes)!=musaSuccess)
      throw StatusError{kStatusAllocationFailed};
    check(attn_probs.SetAddr(attn_probs_buffer));
  }
  ~MaskedMathAttention() { if(attn_probs_buffer) musaFree(attn_probs_buffer); }
};

md::Tensor::Format channel_first_format(int rank) {
  switch (rank) {
    case 3: return md::Tensor::Format::NCW;
    case 4: return md::Tensor::Format::NCHW;
    case 5: return md::Tensor::Format::NCDHW;
    default: throw StatusError{kStatusInvalidArgument};
  }
}

void configure_bf16_tensor(md::Tensor& tensor, const int64_t* dims, int rank) {
  if (!dims || rank < 1 || rank > 5) throw StatusError{kStatusInvalidArgument};
  for (int i = 0; i < rank; ++i) {
    if (dims[i] <= 0) throw StatusError{kStatusInvalidArgument};
  }
  check(tensor.SetType(md::Tensor::Type::BFLOAT16));
  check(tensor.SetNdInfo(rank, dims));
  if (rank >= 3) check(tensor.SetFormat(channel_first_format(rank)));
}

void configure_empty_bf16_tensor(md::Tensor& tensor) {
  check(tensor.SetType(md::Tensor::Type::BFLOAT16));
  check(tensor.SetNdInfo({0}));
}

void configure_contiguous_bf16_tensor(md::Tensor& tensor,const int64_t* dims,int rank) {
  if(!dims||rank<1||rank>8) throw StatusError{kStatusInvalidArgument};
  for(int i=0;i<rank;++i) if(dims[i]<=0) throw StatusError{kStatusInvalidArgument};
  check(tensor.SetType(md::Tensor::Type::BFLOAT16));
  check(tensor.SetNdInfo(rank,dims));
}

struct LayerNormOp {
  md::Tensor input,weight,bias,output,mean,inv_var;
  md::LayerNorm operation;
  Scratch scratch;
  void* statistics=nullptr;
  bool affine;
  LayerNormOp(int rows,int cols,double eps,bool use_affine,musaStream_t stream)
      : scratch(stream),affine(use_affine) {
    const int64_t matrix_dims[2]={rows,cols};
    for(auto p:{&input,&output}) configure_contiguous_bf16_tensor(*p,matrix_dims,2);
    if(affine) {
      const int64_t affine_dims[1]={cols};
      configure_contiguous_bf16_tensor(weight,affine_dims,1);
      configure_contiguous_bf16_tensor(bias,affine_dims,1);
    } else {
      configure_empty_bf16_tensor(weight);
      configure_empty_bf16_tensor(bias);
    }
    for(auto p:{&mean,&inv_var}) {
      check(p->SetType(md::Tensor::Type::FLOAT));
      check(p->SetNdInfo({rows}));
    }
    check(operation.SetAxis({1}));
    check(operation.SetEpsilon(eps));
    check(operation.SetVarMode(md::LayerNorm::VarMode::DIRECT));
    const size_t bytes=size_t(rows)*2*sizeof(float);
    if(musaMalloc(&statistics,bytes)!=musaSuccess) throw StatusError{kStatusAllocationFailed};
    check(mean.SetAddr(statistics));
    check(inv_var.SetAddr(static_cast<unsigned char*>(statistics)+size_t(rows)*sizeof(float)));
  }
  ~LayerNormOp(){if(statistics)musaFree(statistics);}
};

struct RmsNormOp {
  md::Tensor input,weight,output,inv_var;
  md::RMSNorm operation;
  void* statistics=nullptr;
  RmsNormOp(int rows,int cols,double eps,musaStream_t) {
    const int64_t matrix_dims[2]={rows,cols};
    for(auto p:{&input,&output}) configure_contiguous_bf16_tensor(*p,matrix_dims,2);
    configure_empty_bf16_tensor(weight);
    check(inv_var.SetType(md::Tensor::Type::FLOAT));
    check(inv_var.SetNdInfo({rows}));
    check(operation.SetAxis({1}));
    check(operation.SetEpsilon(eps));
    check(operation.SetVarMode(md::RMSNorm::VarMode::DIRECT));
    if(musaMalloc(&statistics,size_t(rows)*sizeof(float))!=musaSuccess)
      throw StatusError{kStatusAllocationFailed};
    check(inv_var.SetAddr(statistics));
  }
  ~RmsNormOp(){if(statistics)musaFree(statistics);}
};

// Adjacent-pair RoPE implemented entirely with muDNN tensor operators. The
// strided BF16 descriptors expose even and odd lanes without a gather kernel;
// the five cached F32 planes preserve the reference's single final BF16 cast.
struct PairwiseRopeOp {
  md::Tensor input_even,input_odd,output_even,output_odd,cos,sin;
  md::Tensor even_f32,odd_f32,product_a,product_b,result;
  md::Unary cast;
  md::Binary multiply,add,subtract;
  void* workspace=nullptr;
  size_t plane_bytes=0;

  PairwiseRopeOp(int batch,int tokens,int heads,int head_dim,musaStream_t) {
    const int pairs=head_dim/2;
    const int64_t pair_dims[4]={batch,tokens,heads,pairs};
    // The model tensor is contiguous [B,H,T,D]; descriptors expose the
    // strided even/odd planes as logical [B,T,H,D/2].
    const int64_t pair_strides[4]={
      int64_t(heads)*tokens*head_dim,
      head_dim,
      int64_t(tokens)*head_dim,
      2,
    };
    for(auto p:{&input_even,&input_odd,&output_even,&output_odd}) {
      check(p->SetType(md::Tensor::Type::BFLOAT16));
      check(p->SetNdInfo(4,pair_dims,pair_strides));
    }
    for(auto p:{&even_f32,&odd_f32,&product_a,&product_b,&result}) {
      check(p->SetType(md::Tensor::Type::FLOAT));
      check(p->SetNdInfo(4,pair_dims));
    }
    const int64_t table_dims[4]={1,tokens,1,pairs};
    for(auto p:{&cos,&sin}) {
      check(p->SetType(md::Tensor::Type::FLOAT));
      check(p->SetNdInfo(4,table_dims));
    }
    check(cast.SetMode(md::Unary::Mode::CAST));
    check(multiply.SetMode(md::Binary::Mode::MUL));
    check(add.SetMode(md::Binary::Mode::ADD));
    check(subtract.SetMode(md::Binary::Mode::SUB));

    plane_bytes=size_t(batch)*tokens*heads*pairs*sizeof(float);
    if(musaMalloc(&workspace,5*plane_bytes)!=musaSuccess)
      throw StatusError{kStatusAllocationFailed};
    auto* base=static_cast<unsigned char*>(workspace);
    check(even_f32.SetAddr(base));
    check(odd_f32.SetAddr(base+plane_bytes));
    check(product_a.SetAddr(base+2*plane_bytes));
    check(product_b.SetAddr(base+3*plane_bytes));
    check(result.SetAddr(base+4*plane_bytes));
  }
  ~PairwiseRopeOp(){if(workspace)musaFree(workspace);}
};

// Convolution is used by Fast-WAM's VAE and video patch embedding.  muDNN's
// recommended algorithm and scratch allocation are shape/stream resources;
// constructing them inside stream capture triggers runtime error 900.  Cache
// both on the warmup traversal, just like Matmul and attention above.
struct ConvolutionOp {
  md::Tensor input,weight,bias,output,add;
  md::Convolution operation;
  md::Convolution::Algorithm algorithm;
  md::Convolution::FusedActivationDesc activation;
  Scratch scratch;
  bool has_bias;
  ConvolutionOp(const int64_t* input_dims,const int64_t* weight_dims,
                const int64_t* output_dims,int rank,const int* padding,
                const int* stride,const int* dilation,int groups,bool use_bias,
                md::Handle& handle,musaStream_t stream)
      : scratch(stream),has_bias(use_bias) {
    const int spatial=rank-2;
    configure_bf16_tensor(input,input_dims,rank);
    configure_bf16_tensor(weight,weight_dims,rank);
    configure_bf16_tensor(output,output_dims,rank);
    if(has_bias) {
      const int64_t bias_dims[1]={weight_dims[0]};
      configure_bf16_tensor(bias,bias_dims,1);
    } else {
      configure_empty_bf16_tensor(bias);
    }
    configure_empty_bf16_tensor(add);
    check(operation.SetGroups(groups));
    check(operation.SetNdInfo(spatial,padding,stride,dilation));
    check(operation.SetComputeMode(md::Convolution::ComputeMode::TENSOR));
    check(operation.GetRecommendForwardAlgorithm(handle,algorithm,output,input,weight));
    check(activation.SetMode(md::Convolution::FusedActivationDesc::Mode::IDENTITY));
  }
};

struct Stream {
  md::Handle handle;
  std::map<std::tuple<int,int,int>,std::unique_ptr<Matmul>> matmuls;
  std::map<std::tuple<int,int,int>,std::unique_ptr<MatmulFp8>> fp8_matmuls;
  std::map<std::tuple<int,int,int>,std::unique_ptr<MatmulFp8Bias>> fp8_bias_matmuls;
  std::map<std::tuple<int,int,int,int>,std::unique_ptr<Flash>> flashes;
  std::map<std::tuple<int,int,int,int,int,int,bool,bool>,std::unique_ptr<GeneralFlash>> general_flashes;
  std::map<std::tuple<int,int,int,int,int>,std::unique_ptr<MaskedMathAttention>> masked_math_attention;
  std::map<std::tuple<int,int,double,bool>,std::unique_ptr<LayerNormOp>> layer_norms;
  std::map<std::tuple<int,int,double>,std::unique_ptr<RmsNormOp>> rms_norms;
  std::map<std::tuple<int,int,int,int>,std::unique_ptr<PairwiseRopeOp>> pairwise_ropes;
  std::map<std::vector<int64_t>,std::unique_ptr<ConvolutionOp>> convolutions;
  Stream(int device,musaStream_t stream) : handle(device) {
    check(handle.SetStream(stream)); check(handle.SetAllowTF32(false));
  }
};
thread_local std::map<std::pair<int,musaStream_t>,std::unique_ptr<Stream>> streams;
Stream& state(musaStream_t stream) {
  const int device=apxinf_adapters::stream_device(stream);
  if(device<0) throw StatusError{kStatusDeviceQueryFailed};
  auto key=std::make_pair(device,stream); auto it=streams.find(key);
  if(it==streams.end()) {
    if(capturing(stream)) throw StatusError{kStatusCaptureInProgress};
    it=streams.emplace(key,std::make_unique<Stream>(device,stream)).first;
  }
  return *it->second;
}

/// Name the reason the last call on this thread failed.
///
/// `set_error` writes the buffer; the two entry points below are the only
/// writers, and each writes on the way out. A caller that reads this after a
/// non-zero return gets a sentence; one that reads it after a zero return gets
/// whatever the previous failure said, which is why the Rust side only asks
/// when the status is non-zero.
void record_failure(int status, const char* what) {
  switch (status) {
    case kStatusDeviceQueryFailed:
      apxinf_adapters::set_error(
          "%s: the stream\'s device could not be named, so the allocation is refused "
          "rather than guessed at", what);
      break;
    case kStatusCaptureStatusUnreadable:
      apxinf_adapters::set_error(
          "%s: musaStreamIsCapturing did not answer, so whether this stream is capturing "
          "is unknown. The allocation is refused rather than guessed at", what);
      break;
    case kStatusCaptureInProgress:
      apxinf_adapters::set_error(
          "%s: the muDNN descriptors for this shape were first needed during graph capture; "
          "a captured graph bakes the pointers it recorded, so every shape must be seen once "
          "on the warmup traversal before musaStreamBeginCapture", what);
      break;
    case kStatusAllocationFailed:
      apxinf_adapters::set_error("%s: musaMalloc failed", what);
      break;
    case kStatusScratchTooSmall:
      apxinf_adapters::set_error("%s: the muDNN scratch buffer for this shape is too small", what);
      break;
    case kStatusInvalidArgument:
      apxinf_adapters::set_error("%s: a pointer was null or a dimension was not positive", what);
      break;
    case kStatusUnknown:
      apxinf_adapters::set_error("%s: an exception this adapter does not name", what);
      break;
    default:
      // A vendor `md::Status`: it has its own names, and inventing one here
      // would be a second vocabulary for the same number.
      apxinf_adapters::set_error("%s: muDNN returned status %d", what, status);
      break;
  }
}
}

extern "C" int apxinf_musa_mudnn_bf16_gemm(const void* a,const void* w,void* y,
 int m,int n,int k,musaStream_t stream) {
 if(!a||!w||!y||m<=0||n<=0||k<=0) {
   record_failure(kStatusInvalidArgument, "muDNN BF16 GEMM");
   return kStatusInvalidArgument;
 }
 try {
  auto& s=state(stream); auto key=std::make_tuple(m,n,k); auto it=s.matmuls.find(key);
  if(it==s.matmuls.end()) {
    if(capturing(stream)) {
      record_failure(kStatusCaptureInProgress, "muDNN BF16 GEMM");
      return kStatusCaptureInProgress;
    }
    it=s.matmuls.emplace(key,std::make_unique<Matmul>(m,n,k,stream)).first;
  }
  auto& p=*it->second; check(p.a.SetAddr(const_cast<void*>(a)));
  check(p.w.SetAddr(const_cast<void*>(w))); check(p.y.SetAddr(y)); p.scratch.cursor=0;
  const int status = int(p.operation.Run(s.handle,p.y,p.a,p.w,[&](size_t n){return p.scratch.get(n);}));
  if (status != 0) record_failure(status, "muDNN BF16 GEMM");
  return status;
 } catch(const StatusError& e) {
   record_failure(e.status, "muDNN BF16 GEMM");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown, "muDNN BF16 GEMM");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_fp8_gemm(const void* a,const void* w,void* y,
 int m,int n,int k,float alpha,musaStream_t stream) {
 if(!a||!w||!y||m<=0||n<=0||k<=0||!std::isfinite(alpha)) {
   record_failure(kStatusInvalidArgument, "muDNN FP8 GEMM");
   return kStatusInvalidArgument;
 }
 try {
  auto& s=state(stream); auto key=std::make_tuple(m,n,k); auto it=s.fp8_matmuls.find(key);
  if(it==s.fp8_matmuls.end()) {
    if(capturing(stream)) {
      record_failure(kStatusCaptureInProgress, "muDNN FP8 GEMM");
      return kStatusCaptureInProgress;
    }
    it=s.fp8_matmuls.emplace(key,std::make_unique<MatmulFp8>(m,n,k,stream)).first;
  }
  auto& p=*it->second; check(p.a.SetAddr(const_cast<void*>(a)));
  check(p.w.SetAddr(const_cast<void*>(w))); check(p.y.SetAddr(y));
  check(p.operation.SetAlpha(static_cast<double>(alpha)));
  p.scratch.cursor=0;
  const int status = int(p.operation.Run(s.handle,p.y,p.a,p.w,[&](size_t n){return p.scratch.get(n);}));
  if (status != 0) record_failure(status, "muDNN FP8 GEMM");
  return status;
 } catch(const StatusError& e) {
   record_failure(e.status, "muDNN FP8 GEMM");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown, "muDNN FP8 GEMM");
   return kStatusUnknown;
 }
}

// d = alpha * a * b + residual + bias in one muDNN call. The residual path
// uses the outplace overload (c is a separate BF16 [m,n]); the bias-only path
// points c at the output and uses the inplace overload with beta = 0.
extern "C" int apxinf_musa_mudnn_fp8_gemm_bias_residual(const void* a,const void* w,void* y,
 const void* bias,const void* residual,int m,int n,int k,float alpha,musaStream_t stream) {
 if(!a||!w||!y||!bias||!residual||m<=0||n<=0||k<=0||!std::isfinite(alpha)) {
   record_failure(kStatusInvalidArgument, "muDNN FP8 GEMM bias+residual");
   return kStatusInvalidArgument;
 }
 try {
  auto& s=state(stream); auto key=std::make_tuple(m,n,k); auto it=s.fp8_bias_matmuls.find(key);
  if(it==s.fp8_bias_matmuls.end()) {
    if(capturing(stream)) {
      record_failure(kStatusCaptureInProgress, "muDNN FP8 GEMM bias+residual");
      return kStatusCaptureInProgress;
    }
    it=s.fp8_bias_matmuls.emplace(key,std::make_unique<MatmulFp8Bias>(m,n,k,stream)).first;
  }
  auto& p=*it->second;
  check(p.a.SetAddr(const_cast<void*>(a)));
  check(p.w.SetAddr(const_cast<void*>(w)));
  check(p.y.SetAddr(y));
  check(p.c.SetAddr(const_cast<void*>(residual)));
  check(p.bias.SetAddr(const_cast<void*>(bias)));
  check(p.operation.SetAlpha(static_cast<double>(alpha)));
  check(p.operation.SetBeta(1.0));
  check(p.operation.SetGamma(1.0));
  p.scratch.cursor=0;
  const int status = int(p.operation.RunWithBiasAdd(
      s.handle,p.y,p.a,p.w,p.c,p.bias,[&](size_t n){return p.scratch.get(n);}));
  if (status != 0) record_failure(status, "muDNN FP8 GEMM bias+residual");
  return status;
 } catch(const StatusError& e) {
   record_failure(e.status, "muDNN FP8 GEMM bias+residual");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown, "muDNN FP8 GEMM bias+residual");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_fp8_gemm_bias(const void* a,const void* w,void* y,
 const void* bias,int m,int n,int k,float alpha,musaStream_t stream) {
 if(!a||!w||!y||!bias||m<=0||n<=0||k<=0||!std::isfinite(alpha)) {
   record_failure(kStatusInvalidArgument, "muDNN FP8 GEMM bias");
   return kStatusInvalidArgument;
 }
 try {
  auto& s=state(stream); auto key=std::make_tuple(m,n,k); auto it=s.fp8_bias_matmuls.find(key);
  if(it==s.fp8_bias_matmuls.end()) {
    if(capturing(stream)) {
      record_failure(kStatusCaptureInProgress, "muDNN FP8 GEMM bias");
      return kStatusCaptureInProgress;
    }
    it=s.fp8_bias_matmuls.emplace(key,std::make_unique<MatmulFp8Bias>(m,n,k,stream)).first;
  }
  auto& p=*it->second;
  check(p.a.SetAddr(const_cast<void*>(a)));
  check(p.w.SetAddr(const_cast<void*>(w)));
  check(p.y.SetAddr(y));
  check(p.c.SetAddr(y));
  check(p.bias.SetAddr(const_cast<void*>(bias)));
  check(p.operation.SetAlpha(static_cast<double>(alpha)));
  check(p.operation.SetBeta(0.0));
  check(p.operation.SetGamma(1.0));
  p.scratch.cursor=0;
  const int status = int(p.operation.RunWithBiasAdd(
      s.handle,p.c,p.a,p.w,p.bias,[&](size_t n){return p.scratch.get(n);}));
  if (status != 0) record_failure(status, "muDNN FP8 GEMM bias");
  return status;
 } catch(const StatusError& e) {
   record_failure(e.status, "muDNN FP8 GEMM bias");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown, "muDNN FP8 GEMM bias");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_convolution_bf16(
 const void* input,const void* weight,const void* bias,void* output,
 const int64_t* input_dims,const int64_t* weight_dims,const int64_t* output_dims,
 int rank,const int* padding,const int* stride,const int* dilation,int groups,
 musaStream_t stream) {
 const int spatial=rank-2;
 if(!input||!weight||!output||!input_dims||!weight_dims||!output_dims||
    !padding||!stride||!dilation||(rank!=4&&rank!=5)||groups<=0) {
   record_failure(kStatusInvalidArgument, "muDNN BF16 convolution");
   return kStatusInvalidArgument;
 }
 try {
   auto& s=state(stream);
   std::vector<int64_t> key;
   key.reserve(size_t(rank)*3+size_t(spatial)*3+3);
   key.push_back(rank); key.push_back(groups); key.push_back(bias?1:0);
   key.insert(key.end(),input_dims,input_dims+rank);
   key.insert(key.end(),weight_dims,weight_dims+rank);
   key.insert(key.end(),output_dims,output_dims+rank);
   for(int i=0;i<spatial;++i) key.push_back(padding[i]);
   for(int i=0;i<spatial;++i) key.push_back(stride[i]);
   for(int i=0;i<spatial;++i) key.push_back(dilation[i]);
   auto it=s.convolutions.find(key);
   if(it==s.convolutions.end()) {
     if(capturing(stream)) {
       record_failure(kStatusCaptureInProgress,"muDNN BF16 convolution");
       return kStatusCaptureInProgress;
     }
     it=s.convolutions.emplace(key,std::make_unique<ConvolutionOp>(
       input_dims,weight_dims,output_dims,rank,padding,stride,dilation,groups,
       bias!=nullptr,s.handle,stream)).first;
   }
   auto& p=*it->second;
   check(p.input.SetAddr(input)); check(p.weight.SetAddr(weight));
   check(p.output.SetAddr(output));
   if(p.has_bias) check(p.bias.SetAddr(bias));
   p.scratch.cursor=0;
   const int status=int(p.operation.RunFusion(
     s.handle,p.output,p.input,p.weight,p.bias,p.add,p.activation,p.algorithm,
     [&](size_t n){return p.scratch.get(n);}));
   if(status!=0) record_failure(status,"muDNN BF16 convolution");
   return status;
 } catch(const StatusError& e) {
   record_failure(e.status,"muDNN BF16 convolution");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown,"muDNN BF16 convolution");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_interpolate_nearest_exact_bf16(
 const void* input,void* output,const int64_t* input_dims,const int64_t* output_dims,
 int rank,const float* scales,musaStream_t stream) {
 if(!input||!output||!input_dims||!output_dims||!scales||rank<3||rank>5) {
   record_failure(kStatusInvalidArgument,"muDNN BF16 nearest-exact interpolate");
   return kStatusInvalidArgument;
 }
 try {
   auto& s=state(stream);
   md::Tensor x,y;
   configure_bf16_tensor(x,input_dims,rank);
   configure_bf16_tensor(y,output_dims,rank);
   check(x.SetAddr(input)); check(y.SetAddr(output));
   md::Interpolate op;
   check(op.SetMode(md::Interpolate::Mode::NEAREST_EXACT));
   check(op.SetScaleInfo(rank-2,const_cast<float*>(scales)));
   const int status=int(op.Run(s.handle,y,x));
   if(status!=0) record_failure(status,"muDNN BF16 nearest-exact interpolate");
   return status;
 } catch(const StatusError& e) {
   record_failure(e.status,"muDNN BF16 nearest-exact interpolate");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown,"muDNN BF16 nearest-exact interpolate");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_unary_bf16(
 const void* input,void* output,const int64_t* dims,int rank,int mode,
 musaStream_t stream) {
 if(!input||!output||!dims||rank<1||rank>5||mode<0||mode>1) {
   record_failure(kStatusInvalidArgument,"muDNN BF16 unary");
   return kStatusInvalidArgument;
 }
 try {
   auto& s=state(stream);
   md::Tensor x,y;
   configure_contiguous_bf16_tensor(x,dims,rank);
   configure_contiguous_bf16_tensor(y,dims,rank);
   check(x.SetAddr(input)); check(y.SetAddr(output));
   md::Unary op;
   check(op.SetMode(mode==0?md::Unary::Mode::GELU_TANH:md::Unary::Mode::SILU));
   const int status=int(op.Run(s.handle,y,x));
   if(status!=0) record_failure(status,"muDNN BF16 unary");
   return status;
 } catch(const StatusError& e) {
   record_failure(e.status,"muDNN BF16 unary");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown,"muDNN BF16 unary");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_binary_bf16(
 const void* left,const int64_t* left_dims,int left_rank,
 const void* right,const int64_t* right_dims,int right_rank,
 void* output,const int64_t* output_dims,int output_rank,int mode,
 musaStream_t stream) {
 if(!left||!right||!output||!left_dims||!right_dims||!output_dims||
    left_rank<1||left_rank>5||right_rank<1||right_rank>5||
    output_rank<1||output_rank>5||mode<0||mode>3) {
   record_failure(kStatusInvalidArgument,"muDNN BF16 binary");
   return kStatusInvalidArgument;
 }
 try {
   auto& s=state(stream);
   md::Tensor l,r,y;
   configure_contiguous_bf16_tensor(l,left_dims,left_rank);
   configure_contiguous_bf16_tensor(r,right_dims,right_rank);
   configure_contiguous_bf16_tensor(y,output_dims,output_rank);
   check(l.SetAddr(left)); check(r.SetAddr(right)); check(y.SetAddr(output));
   md::Binary op;
   const auto op_mode = mode == 0 ? md::Binary::Mode::ADD
                      : mode == 1 ? md::Binary::Mode::MUL
                      : mode == 2 ? md::Binary::Mode::MAX
                                  : md::Binary::Mode::MIN;
   check(op.SetMode(op_mode));
   const int status=int(op.Run(s.handle,y,l,r));
   if(status!=0) record_failure(status,"muDNN BF16 binary");
   return status;
 } catch(const StatusError& e) {
   record_failure(e.status,"muDNN BF16 binary");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown,"muDNN BF16 binary");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_layer_norm_bf16(
 const void* input,const void* weight,const void* bias,void* output,
 int rows,int cols,double eps,musaStream_t stream) {
 if(!input||!output||rows<=0||cols<=0||!std::isfinite(eps)||eps<=0||
    ((weight==nullptr)!=(bias==nullptr))) {
   record_failure(kStatusInvalidArgument,"muDNN BF16 layer norm");
   return kStatusInvalidArgument;
 }
 try {
   auto& s=state(stream);
   const bool affine=weight!=nullptr;
   auto key=std::make_tuple(rows,cols,eps,affine);
   auto it=s.layer_norms.find(key);
   if(it==s.layer_norms.end()) {
     if(capturing(stream)) {
       record_failure(kStatusCaptureInProgress,"muDNN BF16 layer norm");
       return kStatusCaptureInProgress;
     }
     it=s.layer_norms.emplace(
       key,std::make_unique<LayerNormOp>(rows,cols,eps,affine,stream)).first;
   }
   auto& p=*it->second;
   check(p.input.SetAddr(input)); check(p.output.SetAddr(output));
   if(affine) { check(p.weight.SetAddr(weight)); check(p.bias.SetAddr(bias)); }
   p.scratch.cursor=0;
   const int status=int(p.operation.Run(s.handle,p.output,p.mean,p.inv_var,p.input,
     p.weight,p.bias,[&](size_t n){return p.scratch.get(n);}));
   if(status!=0) record_failure(status,"muDNN BF16 layer norm");
   return status;
 } catch(const StatusError& e) {
   record_failure(e.status,"muDNN BF16 layer norm");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown,"muDNN BF16 layer norm");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_rms_norm_bf16(
 const void* input,void* output,int rows,int cols,double eps,musaStream_t stream) {
 if(!input||!output||rows<=0||cols<=0||!std::isfinite(eps)||eps<=0) {
   record_failure(kStatusInvalidArgument,"muDNN BF16 RMS norm");
   return kStatusInvalidArgument;
 }
 try {
   auto& s=state(stream);
   auto key=std::make_tuple(rows,cols,eps);
   auto it=s.rms_norms.find(key);
   if(it==s.rms_norms.end()) {
     if(capturing(stream)) {
       record_failure(kStatusCaptureInProgress,"muDNN BF16 RMS norm");
       return kStatusCaptureInProgress;
     }
     it=s.rms_norms.emplace(
       key,std::make_unique<RmsNormOp>(rows,cols,eps,stream)).first;
   }
   auto& p=*it->second;
   check(p.input.SetAddr(input)); check(p.output.SetAddr(output));
   const int status=int(p.operation.Run(s.handle,p.output,p.inv_var,p.input,p.weight));
   if(status!=0) record_failure(status,"muDNN BF16 RMS norm");
   return status;
 } catch(const StatusError& e) {
   record_failure(e.status,"muDNN BF16 RMS norm");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown,"muDNN BF16 RMS norm");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_pairwise_rope_bf16(
 const void* input,const void* cos,const void* sin,void* output,
 int batch,int tokens,int heads,int head_dim,musaStream_t stream) {
 if(!input||!cos||!sin||!output||batch<=0||tokens<=0||heads<=0||
    head_dim<=0||head_dim%2!=0) {
   record_failure(kStatusInvalidArgument,"muDNN BF16 pairwise RoPE");
   return kStatusInvalidArgument;
 }
 try {
   auto& s=state(stream);
   auto key=std::make_tuple(batch,tokens,heads,head_dim);
   auto it=s.pairwise_ropes.find(key);
   if(it==s.pairwise_ropes.end()) {
     if(capturing(stream)) {
       record_failure(kStatusCaptureInProgress,"muDNN BF16 pairwise RoPE");
       return kStatusCaptureInProgress;
     }
     it=s.pairwise_ropes.emplace(
       key,std::make_unique<PairwiseRopeOp>(batch,tokens,heads,head_dim,stream)).first;
   }
   auto& p=*it->second;
   auto* input_bytes=static_cast<const unsigned char*>(input);
   auto* output_bytes=static_cast<unsigned char*>(output);
   check(p.input_even.SetAddr(input_bytes));
   check(p.input_odd.SetAddr(input_bytes+sizeof(uint16_t)));
   check(p.output_even.SetAddr(output_bytes));
   check(p.output_odd.SetAddr(output_bytes+sizeof(uint16_t)));
   check(p.cos.SetAddr(cos));
   check(p.sin.SetAddr(sin));

   check(p.cast.Run(s.handle,p.even_f32,p.input_even));
   check(p.cast.Run(s.handle,p.odd_f32,p.input_odd));
   check(p.multiply.Run(s.handle,p.product_a,p.even_f32,p.cos));
   check(p.multiply.Run(s.handle,p.product_b,p.odd_f32,p.sin));
   check(p.subtract.Run(s.handle,p.result,p.product_a,p.product_b));
   check(p.cast.Run(s.handle,p.output_even,p.result));
   check(p.multiply.Run(s.handle,p.product_a,p.even_f32,p.sin));
   check(p.multiply.Run(s.handle,p.product_b,p.odd_f32,p.cos));
   check(p.add.Run(s.handle,p.result,p.product_a,p.product_b));
   const int status=int(p.cast.Run(s.handle,p.output_odd,p.result));
   if(status!=0) record_failure(status,"muDNN BF16 pairwise RoPE");
   return status;
 } catch(const StatusError& e) {
   record_failure(e.status,"muDNN BF16 pairwise RoPE");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown,"muDNN BF16 pairwise RoPE");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_concat_rows_bf16(
 const void* first,const void* second,void* output,
 int first_rows,int second_rows,int cols,musaStream_t stream) {
 if(!first||!second||!output||first_rows<=0||second_rows<=0||cols<=0) {
   record_failure(kStatusInvalidArgument,"muDNN BF16 row concatenation");
   return kStatusInvalidArgument;
 }
 try {
   auto& s=state(stream);
   md::Tensor inputs[2];
   md::Tensor y;
   const int64_t first_dims[2]={first_rows,cols};
   const int64_t second_dims[2]={second_rows,cols};
   const int64_t output_dims[2]={first_rows+second_rows,cols};
   configure_contiguous_bf16_tensor(inputs[0],first_dims,2);
   configure_contiguous_bf16_tensor(inputs[1],second_dims,2);
   configure_contiguous_bf16_tensor(y,output_dims,2);
   check(inputs[0].SetAddr(first));
   check(inputs[1].SetAddr(second));
   check(y.SetAddr(output));
   md::Concat op;
   check(op.SetAxis(0));
   const int status=int(op.Run(s.handle,y,2,inputs));
   if(status!=0) record_failure(status,"muDNN BF16 row concatenation");
   return status;
 } catch(const StatusError& e) {
   record_failure(e.status,"muDNN BF16 row concatenation");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown,"muDNN BF16 row concatenation");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_permute_bf16(
 const void* input,void* output,const int64_t* input_dims,
 const int64_t* permutation,int rank,musaStream_t stream) {
 if(!input||!output||!input_dims||!permutation||rank<1||rank>8) {
   record_failure(kStatusInvalidArgument,"muDNN BF16 permute");
   return kStatusInvalidArgument;
 }
 try {
   std::vector<int64_t> output_dims(rank);
   std::vector<bool> seen(rank,false);
   for(int i=0;i<rank;++i) {
     if(input_dims[i]<=0||permutation[i]<0||permutation[i]>=rank||seen[permutation[i]])
       throw StatusError{kStatusInvalidArgument};
     seen[permutation[i]]=true;
     output_dims[i]=input_dims[permutation[i]];
   }
   auto& s=state(stream);
   md::Tensor x,y;
   configure_contiguous_bf16_tensor(x,input_dims,rank);
   configure_contiguous_bf16_tensor(y,output_dims.data(),rank);
   check(x.SetAddr(input));
   check(y.SetAddr(output));
   check(md::Permute::ConfigDimStride(y,x,rank,permutation));
   md::Permute op;
   const int status=int(op.Run(s.handle,y,x));
   if(status!=0) record_failure(status,"muDNN BF16 permute");
   return status;
 } catch(const StatusError& e) {
   record_failure(e.status,"muDNN BF16 permute");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown,"muDNN BF16 permute");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_flash_bf16(const void* q,const void* k,const void* v,
 void* y,int batch,int heads,int tokens,int dim,musaStream_t stream) {
 if(!q||!k||!v||!y||batch<=0||heads<=0||tokens<=0||dim<=0) {
   record_failure(kStatusInvalidArgument, "muDNN flash attention");
   return kStatusInvalidArgument;
 }
 try {
  auto& s=state(stream); auto key=std::make_tuple(batch,heads,tokens,dim); auto it=s.flashes.find(key);
  if(it==s.flashes.end()) {
    if(capturing(stream)) {
      record_failure(kStatusCaptureInProgress, "muDNN flash attention");
      return kStatusCaptureInProgress;
    }
    it=s.flashes.emplace(key,std::make_unique<Flash>(batch,heads,tokens,dim,stream)).first;
  }
  auto& p=*it->second; check(p.q.SetAddr(const_cast<void*>(q)));check(p.k.SetAddr(const_cast<void*>(k)));
  check(p.v.SetAddr(const_cast<void*>(v)));check(p.y.SetAddr(y));p.scratch.cursor=0;
  const int status = int(p.operation.RunFlash(s.handle,p.y,p.lse,p.q,p.k,p.v,p.mask,p.dropout,
    [&](size_t n){return p.scratch.get(n);}));
  if (status != 0) record_failure(status, "muDNN flash attention");
  return status;
 } catch(const StatusError& e) {
   record_failure(e.status, "muDNN flash attention");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown, "muDNN flash attention");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_general_flash_bf16(
 const void* q,const void* k,const void* v,void* y,int batch,
 int query_heads,int kv_heads,int query_tokens,int key_tokens,int dim,
 bool causal,musaStream_t stream) {
 if(!q||!k||!v||!y||batch<=0||query_heads<=0||kv_heads<=0||
    query_tokens<=0||key_tokens<=0||dim<=0||query_heads%kv_heads!=0) {
   record_failure(kStatusInvalidArgument, "muDNN general flash attention");
   return kStatusInvalidArgument;
 }
 try {
  auto& s=state(stream);
  auto key=std::make_tuple(batch,query_heads,kv_heads,query_tokens,key_tokens,dim,causal,false);
  auto it=s.general_flashes.find(key);
  if(it==s.general_flashes.end()) {
    if(capturing(stream)) {
      record_failure(kStatusCaptureInProgress, "muDNN general flash attention");
      return kStatusCaptureInProgress;
    }
    it=s.general_flashes.emplace(
      key,std::make_unique<GeneralFlash>(batch,query_heads,kv_heads,query_tokens,
                                         key_tokens,dim,causal,false,stream)).first;
  }
  auto& p=*it->second;
  check(p.q.SetAddr(const_cast<void*>(q))); check(p.k.SetAddr(const_cast<void*>(k)));
  check(p.v.SetAddr(const_cast<void*>(v))); check(p.y.SetAddr(y)); p.scratch.cursor=0;
  const int status = int(p.operation.RunFlash(s.handle,p.y,p.lse,p.q,p.k,p.v,p.mask,p.dropout,
    [&](size_t n){return p.scratch.get(n);}));
  if (status != 0) record_failure(status, "muDNN general flash attention");
  return status;
 } catch(const StatusError& e) {
   record_failure(e.status, "muDNN general flash attention");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown, "muDNN general flash attention");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_masked_math_attention_bf16(
 const void* q,const void* k,const void* v,const void* mask,void* y,
 int batch,int heads,int query_tokens,int key_tokens,int dim,musaStream_t stream) {
 if(!q||!k||!v||!mask||!y||batch<=0||heads<=0||query_tokens<=0||
    key_tokens<=0||dim<=0) {
   record_failure(kStatusInvalidArgument,"muDNN masked math attention");
   return kStatusInvalidArgument;
 }
 try {
  auto& s=state(stream);
  auto key=std::make_tuple(batch,heads,query_tokens,key_tokens,dim);
  auto it=s.masked_math_attention.find(key);
  if(it==s.masked_math_attention.end()) {
    if(capturing(stream)) {
      record_failure(kStatusCaptureInProgress,"muDNN masked math attention");
      return kStatusCaptureInProgress;
    }
    it=s.masked_math_attention.emplace(
      key,std::make_unique<MaskedMathAttention>(batch,heads,query_tokens,
                                                key_tokens,dim,stream)).first;
  }
  auto& p=*it->second;
  check(p.q.SetAddr(const_cast<void*>(q))); check(p.k.SetAddr(const_cast<void*>(k)));
  check(p.v.SetAddr(const_cast<void*>(v))); check(p.mask.SetAddr(const_cast<void*>(mask)));
  check(p.y.SetAddr(y)); p.scratch.cursor=0;
  const int status=int(p.operation.RunMath(s.handle,p.y,p.attn_probs,p.q,p.k,p.v,
    p.mask,p.dropout,[&](size_t n){return p.scratch.get(n);}));
  if(status!=0) record_failure(status,"muDNN masked math attention");
  return status;
 } catch(const StatusError& e) {
   record_failure(e.status,"muDNN masked math attention");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown,"muDNN masked math attention");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_packed_qkv_flash_bf16(
 const void* qkv,void* y,int batch,int heads,int tokens,int dim,musaStream_t stream) {
 if(!qkv||!y||batch<=0||heads<=0||tokens<=0||dim<=0) {
   record_failure(kStatusInvalidArgument, "muDNN packed-QKV flash attention");
   return kStatusInvalidArgument;
 }
 try {
  auto& s=state(stream);
  auto key=std::make_tuple(batch,heads,heads,tokens,tokens,dim,false,true);
  auto it=s.general_flashes.find(key);
  if(it==s.general_flashes.end()) {
    if(capturing(stream)) {
      record_failure(kStatusCaptureInProgress, "muDNN packed-QKV flash attention");
      return kStatusCaptureInProgress;
    }
    it=s.general_flashes.emplace(
      key,std::make_unique<GeneralFlash>(batch,heads,heads,tokens,tokens,
                                         dim,false,true,stream)).first;
  }
  auto& p=*it->second;
  const auto* base=static_cast<const unsigned char*>(qkv);
  const size_t projection_bytes=size_t(heads)*size_t(dim)*sizeof(uint16_t);
  check(p.q.SetAddr(const_cast<unsigned char*>(base)));
  check(p.k.SetAddr(const_cast<unsigned char*>(base+projection_bytes)));
  check(p.v.SetAddr(const_cast<unsigned char*>(base+2*projection_bytes)));
  check(p.y.SetAddr(y)); p.scratch.cursor=0;
  const int status = int(p.operation.RunFlash(s.handle,p.y,p.lse,p.q,p.k,p.v,p.mask,p.dropout,
    [&](size_t n){return p.scratch.get(n);}));
  if (status != 0) record_failure(status, "muDNN packed-QKV flash attention");
  return status;
 } catch(const StatusError& e) {
   record_failure(e.status, "muDNN packed-QKV flash attention");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown, "muDNN packed-QKV flash attention");
   return kStatusUnknown;
 }
}

/// Why the last call on this thread failed. A static string.
extern "C" const char* apxinf_musa_mudnn_last_error(void) {
  return apxinf_adapters::error_buffer();
}


extern "C" void apxinf_musa_mudnn_release_stream(musaStream_t stream) {
 for(auto it=streams.begin();it!=streams.end();) {
  if(it->first.second==stream) {
   apxinf_adapters::select_device(it->first.first);
   it=streams.erase(it);
  } else ++it;
 }
}
