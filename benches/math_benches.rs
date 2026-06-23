use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::Rng;

use parsecdb::math::distance::{cosine_similarity, normalize_in_place};
use parsecdb::storage::buffer::SoABuffer;

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    normalize_in_place(&mut vec);
    vec
}

fn bench_simd_distance(c: &mut Criterion) {
    let dim = 768;

    let mut buffer = SoABuffer::new(dim, 2);

    let v1 = generate_random_vector(dim);
    let v2 = generate_random_vector(dim);

    buffer.insert(1, &v1).unwrap();
    buffer.insert(2, &v2).unwrap();

    let a = buffer.get_vector(0).unwrap();
    let b = buffer.get_vector(1).unwrap();

    let mut group = c.benchmark_group("Vector Math");

    group.bench_function("cosine_similarity_768d_AVX2", |b_iter| {
        b_iter.iter(|| cosine_similarity(black_box(a), black_box(b)))
    });

    group.finish();
}

criterion_group!(benches, bench_simd_distance);
criterion_main!(benches);
