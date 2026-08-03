use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use ruvector_diskann::distance::{
    inner_product, l2_squared, pq_asymmetric_distance, scalar_l2_squared,
};

const DIMS: [usize; 4] = [128, 384, 768, 1536];
const SEED: u64 = 0x5EED_1234_ABCD_EF01;
const PQ_CODEBOOK_SIZE: usize = 256;

fn random_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn bench_l2_squared(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_squared");
    let mut rng = StdRng::seed_from_u64(SEED);

    for &dim in DIMS.iter() {
        let a = random_vector(&mut rng, dim);
        let b = random_vector(&mut rng, dim);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| black_box(l2_squared(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

fn bench_scalar_l2_squared(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_l2_squared");
    let mut rng = StdRng::seed_from_u64(SEED);

    for &dim in DIMS.iter() {
        let a = random_vector(&mut rng, dim);
        let b = random_vector(&mut rng, dim);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| black_box(scalar_l2_squared(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("inner_product");
    let mut rng = StdRng::seed_from_u64(SEED);

    for &dim in DIMS.iter() {
        let a = random_vector(&mut rng, dim);
        let b = random_vector(&mut rng, dim);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| black_box(inner_product(black_box(&a), black_box(&b))));
        });
    }

    group.finish();
}

fn bench_pq_asymmetric_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_asymmetric_distance");
    let mut rng = StdRng::seed_from_u64(SEED);

    for &dim in DIMS.iter() {
        let codes: Vec<u8> = (0..dim)
            .map(|_| rng.gen_range(0..PQ_CODEBOOK_SIZE) as u8)
            .collect();
        let table: Vec<f32> = (0..dim * PQ_CODEBOOK_SIZE)
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| {
                black_box(pq_asymmetric_distance(
                    black_box(&codes),
                    black_box(&table),
                    black_box(PQ_CODEBOOK_SIZE),
                ))
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_l2_squared,
    bench_scalar_l2_squared,
    bench_inner_product,
    bench_pq_asymmetric_distance
);
criterion_main!(benches);
