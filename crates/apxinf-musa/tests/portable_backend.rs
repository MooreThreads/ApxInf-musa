use apxinf_core::{Backend, Tensor};
use apxinf_musa::MusaBackend;
fn host(shape: Vec<usize>, x: &[f32]) -> Tensor {
    Tensor::from_bf16(shape, &x.iter().map(|&v| half::bf16::from_f32(v)).collect::<Vec<_>>()).unwrap()
}
#[test]
fn portable_elementwise_and_rejections() {
    let b = MusaBackend::new(0).unwrap();
    let x = b.to_device(&host(vec![2, 3], &[1., -2., 3., 4., 0., -1.])).unwrap();
    let y = b.mul(&x, &x).unwrap();
    assert_eq!(b.to_cpu(&y).unwrap().to_f32_vec().unwrap(), vec![1.,4.,9.,16.,0.,1.]);
    let z = b.add(&x, &x).unwrap();
    assert_eq!(b.to_cpu(&z).unwrap().to_f32_vec().unwrap(), vec![2.,-4.,6.,8.,0.,-2.]);
    assert!(b.mul(&x, &b.to_device(&host(vec![3], &[1.;3])).unwrap()).is_err());
    assert!(b.begin_capture().is_err());
    assert!(b.embedding(&x, &[2]).is_err());
}
#[test]
fn eager_gemm_survives_following_calls() {
    let b = MusaBackend::new(0).unwrap();
    let x = b.to_device(&host(vec![2, 2], &[1.,2.,3.,4.])).unwrap();
    let a = b.matmul(&x, &x).unwrap();
    let _other = b.matmul(&x, &x).unwrap();
    assert_eq!(b.to_cpu(&a).unwrap().to_f32_vec().unwrap(), vec![7.,10.,15.,22.]);
}
