#include "Tensor.h"
#include <cassert>
#include <iostream>

int main() {
  std::cout << "========================================\n";
  std::cout << " Step 1: Memory Arena & Tensor Setup\n";
  std::cout << "========================================\n\n";

  // 1. Create 1MB Memory Arena
  MemoryArena arena(1024 * 1024);
  std::cout << "[+] Memory Arena initialized (1 MB capacity).\n";

  // 2. Allocate Tensors
  Tensor<float> A(64, 64, arena);
  Tensor<float> B(64, 64, arena);

  // 3. Verify 64-byte alignment of tensor data pointers
  uintptr_t addr_A = reinterpret_cast<uintptr_t>(A.get_data());
  uintptr_t addr_B = reinterpret_cast<uintptr_t>(B.get_data());

  std::cout << "[+] Tensor A address: 0x" << std::hex << addr_A << std::dec
            << "\n";
  std::cout << "[+] Tensor B address: 0x" << std::hex << addr_B << std::dec
            << "\n";

  assert(addr_A % 64 == 0 && "Tensor A is NOT 64-byte aligned!");
  assert(addr_B % 64 == 0 && "Tensor B is NOT 64-byte aligned!");
  std::cout << "[SUCCESS] Both tensors are 64-byte aligned!\n\n";

  // 4. Test writing and reading using 2D indexing operator()
  A(0, 0) = 42.0f;
  A(63, 63) = 100.5f;

  std::cout << "[+] A(0, 0)   = " << A(0, 0) << "\n";
  std::cout << "[+] A(63, 63) = " << A(63, 63) << "\n";
  assert(A(0, 0) == 42.0f);
  assert(A(63, 63) == 100.5f);

  std::cout << "\n========================================\n";
  std::cout << " STEP 1 COMPLETED SUCCESSFULLY!\n";
  std::cout << "========================================\n";

  return 0;
}
