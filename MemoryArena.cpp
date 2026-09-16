#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

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
  ~MemoryArena() { free(raw_buffer); }

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
  void reset() { current_offset = 0; }
};
