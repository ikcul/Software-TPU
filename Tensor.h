#include "MemoryArena.cpp"
#include <cstddef>

template <typename T> class Tensor {
private:
  size_t rows, cols;
  MemoryArena &arena;
  T *data;

public:
  Tensor(size_t rows, size_t cols, MemoryArena &arena)
      : rows(rows), cols(cols), arena(arena) {
    data = reinterpret_cast<T *>(arena.allocate(rows * cols * sizeof(T), 64));
  }

  T &operator()(size_t r, size_t c) { return data[r * cols + c]; }

  const T &operator()(size_t r, size_t c) const { return data[r * cols + c]; }

  size_t get_rows() const { return rows; }
  size_t get_cols() const { return cols; }
  T *get_data() { return data; }
  const T *get_data() const { return data; }
};