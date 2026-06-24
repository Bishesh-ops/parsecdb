use parsecdb::core::types::DistanceMetric;
use parsecdb::index::flat::FlatIndex;
use parsecdb::math::distance::normalize_in_place;

#[test]
fn test_flat_index_cosine() {
    let dim = 128;
    let capacity = 100;

    let mut index = FlatIndex::new(dim, capacity, DistanceMetric::Cosine);

    let mut target = vec![0.5; dim];
    normalize_in_place(&mut target);

    for i in 0..50 {
        let mut noise = vec![0.0; dim];

        noise[i % dim] = 1.0;

        normalize_in_place(&mut noise);
        index
            .insert(i as u64, &noise)
            .expect("Failed to insert noise");
    }

    index.insert(999, &target).expect("Failed to insert target");

    let results = index.search(&target, 3);

    assert_eq!(results.len(), 3, "Should return exactly K results");

    assert_eq!(results[0].id, 999, "Top result did not match expected ID");

    assert!(
        results[0].distance < 0.0001,
        "Perfect match distance should be ~0.0"
    );

    assert!(results[0].distance <= results[1].distance);
    assert!(results[1].distance <= results[2].distance);
}

#[test]
fn test_flat_index_l2_squared() {
    let dim = 4;
    let capacity = 10;

    let mut index = FlatIndex::new(dim, capacity, DistanceMetric::L2Squared);

    let target = vec![1.0, 1.0, 1.0, 1.0];

    index.insert(1, &[0.0, 0.0, 0.0, 0.0]).unwrap(); // Diff: 1^2 * 4 = 4.0
    index.insert(2, &[1.0, 1.0, 1.0, 1.0]).unwrap(); // Exact match = 0.0
    index.insert(3, &[2.0, 2.0, 2.0, 2.0]).unwrap(); // Diff: 1^2 * 4 = 4.0
    index.insert(4, &[1.0, 1.0, 1.0, 0.0]).unwrap(); // Diff: 1^2 * 1 = 1.0

    let results = index.search(&target, 2);

    assert_eq!(results.len(), 2, "Should return K=2 results");

    assert_eq!(results[0].id, 2);
    assert_eq!(results[0].distance, 0.0);

    // Second best result should be ID 4 (Distance of 1.0)
    assert_eq!(results[1].id, 4);
    assert_eq!(results[1].distance, 1.0);
}
