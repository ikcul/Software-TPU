#pragma once
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iostream>

class MemoryArena {
private:
  uint8_t *raw_buffer = nullptr;
  uint8_t *aligned_buffer = nullptr;
  size_t buffer_size = 0;
  size_t current_offset;

public:
  MemoryArena(size_t capacity_bytes) : current_offset(0) {
    capacity_bytes = (capacity_bytes + 63) & ~63;
    buffer_size = capacity_bytes;
    raw_buffer = reinterpret_cast<uint8_t *>(std::malloc(capacity_bytes + 64));
    if (raw_buffer == nullptr)
      exit(1);

    uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw_buffer);
    uintptr_t aligned_addr = (raw_addr + 63) & ~63;
    aligned_buffer = reinterpret_cast<uint8_t *>(aligned_addr);
  }
  ~MemoryArena() { std::free(raw_buffer); }

  void *allocate(size_t bytes, size_t alignment = 64) {
    uintptr_t current_ptr =
        reinterpret_cast<uintptr_t>(aligned_buffer) + current_offset;
    uintptr_t aligned_ptr = (current_ptr + alignment - 1) & ~(alignment - 1);

    size_t aligned_padding = aligned_ptr - current_ptr;
    size_t total_required = current_offset + aligned_padding + bytes;

    if (total_required > buffer_size) {
      return nullptr;
    }

    void *block = reinterpret_cast<void *>(aligned_ptr);

    current_offset = total_required;

    return block;
  }
  void reset(bool zero_out = false) {
    // Non-Temporal AVX2 SIMD zeroing: bypasses L1/L2 cache to prevent cache
    // thrashing
    if (zero_out && aligned_buffer && buffer_size > 0) {
      __m256i zero_vec = _mm256_setzero_si256();
      size_t i = 0;
      for (; i + 32 <= buffer_size; i += 32) {
        _mm256_stream_si256(reinterpret_cast<__m256i *>(aligned_buffer + i),
                            zero_vec);
      }
      for (; i < buffer_size; ++i) {
        aligned_buffer[i] = 0;
      }
      _mm_sfence();
    }
    current_offset = 0;
  }

  size_t get_used_bytes() const { return current_offset; }
  size_t get_capacity() const { return buffer_size; }
};
