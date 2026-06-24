use serde::{Deserialize, Serialize};
use std::alloc::{Layout, alloc, dealloc, handle_alloc_error, realloc};
use std::fs::File;
use std::io::{Read, Write};
use std::ptr::{self, NonNull};

use crate::core::error::{ParsecError, Result};
use crate::core::types::{Scalar, VectorId};

/// Hardware AVX2 registers require 32-byte memory alignment.
const SIMD_ALIGNMENT: usize = 32;

/// A custom highly -optimized vector that manually controls its own memory
/// to gurantee perfect hardware alignment for SIMD instructions.
#[derive(Debug)]
struct AlignedFloatVec {
    ptr: NonNull<Scalar>,
    capacity: usize,
    len: usize,
}

impl AlignedFloatVec {
    /// Casts the raw float memory into a byte for zero-copy disk writes.
    fn as_byte_slice(&self) -> &[u8] {
        let byte_len = self.len * std::mem::size_of::<Scalar>();
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, byte_len) }
    }

    /// Casts the raw memory into a mutable byte slice to read directly from disk.
    fn as_mut_byte_slice(&mut self, new_len: usize) -> &mut [u8] {
        if new_len > self.capacity {
            self.grow(new_len);
        }
        self.len = new_len;
        let byte_len = self.len * std::mem::size_of::<Scalar>();
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut u8, byte_len) }
    }

    /// Allocates raw, uninitialized memory directly from the OS.
    fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                ptr: NonNull::dangling(),
                capacity: 0,
                len: 0,
            };
        }
        // Custom Memory Layout: Array of f32s, strictly 32-byte aligned.
        let layout = Layout::array::<Scalar>(capacity)
            .expect("Capacity overflow")
            .align_to(SIMD_ALIGNMENT)
            .expect("Alignment failed");

        // Raw memory Pointer.
        let raw_ptr = unsafe { alloc(layout) as *mut Scalar };
        // Unwrap the raw-pointer with error handling as well.
        let ptr = NonNull::new(raw_ptr).unwrap_or_else(|| handle_alloc_error(layout));

        Self {
            ptr,
            capacity,
            len: 0,
        }
    }

    /// Appends a slice of floats to the end of our custom buffer.
    fn extend_from_slice(&mut self, slice: &[Scalar]) {
        let new_len = self.len + slice.len();

        if new_len > self.capacity {
            self.grow(new_len);
        }

        unsafe {
            // Get the memory address of where we should start writing.
            let write_ptr = self.ptr.as_ptr().add(self.len);
            // Copy the slice data into our buffer.
            ptr::copy_nonoverlapping(slice.as_ptr(), write_ptr, slice.len());
        }
        self.len = new_len;
    }

    /// Reallocates memory if we run out of capacity.
    fn grow(&mut self, required_len: usize) {
        let new_capacity = std::cmp::max(self.capacity * 2, required_len);
        let old_layout = Layout::array::<Scalar>(self.capacity)
            .unwrap()
            .align_to(SIMD_ALIGNMENT)
            .unwrap();

        let new_layout = Layout::array::<Scalar>(new_capacity)
            .unwrap()
            .align_to(SIMD_ALIGNMENT)
            .unwrap();
        // Requesting the OS to move our data to a larger chink of memory.
        let raw_ptr = unsafe {
            realloc(self.ptr.as_ptr() as *mut u8, old_layout, new_layout.size()) as *mut Scalar
        };

        self.ptr = NonNull::new(raw_ptr).unwrap_or_else(|| handle_alloc_error(new_layout));
        self.capacity = new_capacity;
    }

    /// Safely expose the raw memory as a standard Rust Slice.
    fn as_slice(&self) -> &[Scalar] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

/// The Drop trait acts like a C++ Destructor.
/// Because we bypassed Vec, we MUST manually free the memory when this struct dies,
/// otherwise ParsecDB will have a massive memory leak.
impl Drop for AlignedFloatVec {
    fn drop(&mut self) {
        if self.capacity > 0 {
            let layout = Layout::array::<Scalar>(self.capacity)
                .unwrap()
                .align_to(SIMD_ALIGNMENT)
                .unwrap();
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

#[derive(Debug)]
pub struct SoABuffer {
    dimension: usize,
    ids: Vec<VectorId>,
    data: AlignedFloatVec,
}

impl SoABuffer {
    pub fn new(dimension: usize, capacity: usize) -> Self {
        Self {
            dimension,
            ids: Vec::with_capacity(capacity),
            data: AlignedFloatVec::with_capacity(capacity * dimension),
        }
    }

    pub fn insert(&mut self, id: VectorId, vector: &[Scalar]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(ParsecError::DimensionMismatch {
                expected: self.dimension,
                found: vector.len(),
            });
        }

        self.ids.push(id);
        self.data.extend_from_slice(vector);

        Ok(())
    }

    pub fn get_vector(&self, index: usize) -> Option<&[Scalar]> {
        if index >= self.ids.len() {
            return None;
        }
        let start = index * self.dimension;
        let end = start + self.dimension;
        Some(&self.data.as_slice()[start..end])
    }

    // ... (len, is_empty, dimension, get_id remain exactly the same as before)
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn get_id(&self, index: usize) -> Option<VectorId> {
        self.ids.get(index).copied()
    }

    pub fn save(&self, file: &mut File) -> std::io::Result<()> {
        let metadata = (self.dimension, &self.ids);
        let encoded_meta = bincode::serialize(&metadata).unwrap();

        let meta_len = encoded_meta.len() as u64;
        file.write_all(&meta_len.to_le_bytes())?;
        file.write_all(&encoded_meta)?;

        let float_bytes = self.data.as_byte_slice();
        file.write_all(float_bytes)?;
        Ok(())
    }
    pub fn load(file: &mut File) -> std::io::Result<Self> {
        let mut meta_len_buf = [0u8; 8];
        file.read_exact(&mut meta_len_buf)?;
        let meta_len = u64::from_le_bytes(meta_len_buf) as usize;

        let mut meta_buf = vec![0u8; meta_len];
        file.read_exact(&mut meta_buf)?;
        let (dimension, ids): (usize, Vec<VectorId>) = bincode::deserialize(&meta_buf).unwrap();

        let total_floats = ids.len() * dimension;
        let mut data = AlignedFloatVec::with_capacity(total_floats);

        let float_bytes = data.as_mut_byte_slice(total_floats);
        file.read_exact(float_bytes)?;

        Ok(Self {
            dimension,
            ids,
            data,
        })
    }
}
