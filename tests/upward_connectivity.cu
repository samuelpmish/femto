#include "femto/assert.hpp"
#include "femto/mesh.hpp"
#include "misc/macros.hpp"
#include "misc/timer.hpp"

#include <cuda_runtime.h>

#if __has_include(<cub/cub.cuh>)
#include <cub/cub.cuh>
#elif __has_include(<cccl/cub/cub.cuh>)
#include <cccl/cub/cub.cuh>
#else
#error "CUB headers were not found"
#endif

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

using namespace femto;

namespace {

constexpr uint32_t vertices_per_tet = 4;
constexpr int block_size = 256;

struct GpuTimer {
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
  }

  ~GpuTimer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }

  template <typename callable>
  double time_ms(callable f) {
    CUDA_CHECK(cudaEventRecord(start_event));
    f();
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
    return double(ms);
  }

  cudaEvent_t start_event{};
  cudaEvent_t stop_event{};
};

struct Method1Result {
  nd::array<uint64_t, 1, memory::space::gpu> sorted_pairs;
  nd::array<uint32_t, 1, memory::space::gpu> offsets;
  uint32_t vertex_bits = 0;
  uint32_t elem_bits = 0;
  uint32_t radix_bits = 0;
  double pack_ms = 0.0;
  double sort_ms = 0.0;
  double lower_bound_ms = 0.0;

  double total_ms() const {
    return pack_ms + sort_ms + lower_bound_ms;
  }
};

struct Method2Result {
  nd::array<uint32_t, 1, memory::space::gpu> offsets;
  nd::array<uint32_t, 1, memory::space::gpu> elems;
  nd::array<uint32_t, 1, memory::space::gpu> sorted_elems;
  double count_ms = 0.0;
  double scan_ms = 0.0;
  double cursor_copy_ms = 0.0;
  double scatter_ms = 0.0;
  double segmented_sort_ms = 0.0;

  double total_unsorted_ms() const {
    return count_ms + scan_ms + cursor_copy_ms + scatter_ms;
  }

  double total_sorted_ms() const {
    return total_unsorted_ms() + segmented_sort_ms;
  }
};

__global__ void pack_vertex_element_pairs(
  const uint32_t * tet_to_vertex,
  uint64_t * pairs,
  uint32_t num_tets,
  uint32_t elem_bits) {

  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t num_pairs = num_tets * vertices_per_tet;
  if (tid >= num_pairs) {
    return;
  }

  uint32_t elem_id = tid / vertices_per_tet;
  uint32_t vertex_id = tet_to_vertex[tid];
  pairs[tid] = (uint64_t(vertex_id) << elem_bits) | uint64_t(elem_id);
}

__global__ void build_offsets_by_lower_bound(
  const uint64_t * sorted_pairs,
  uint32_t * offsets,
  uint32_t num_vertices,
  uint32_t num_pairs,
  uint32_t elem_bits) {

  uint32_t vertex_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (vertex_id > num_vertices) {
    return;
  }

  uint64_t key = uint64_t(vertex_id) << elem_bits;
  uint32_t lo = 0;
  uint32_t hi = num_pairs;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (sorted_pairs[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  offsets[vertex_id] = lo;
}

__global__ void count_tet_vertices(
  const uint32_t * tet_to_vertex,
  uint32_t * counts,
  uint32_t num_tets) {

  uint32_t elem_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (elem_id >= num_tets) {
    return;
  }

  uint32_t base = elem_id * vertices_per_tet;
  for (uint32_t i = 0; i < vertices_per_tet; i++) {
    atomicAdd(&counts[tet_to_vertex[base + i]], 1u);
  }
}

__global__ void set_last_offset(
  uint32_t * offsets,
  uint32_t num_vertices,
  uint32_t num_pairs) {
  offsets[num_vertices] = num_pairs;
}

__global__ void scatter_tet_vertices(
  const uint32_t * tet_to_vertex,
  uint32_t * cursors,
  uint32_t * elems,
  uint32_t num_tets) {

  uint32_t elem_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (elem_id >= num_tets) {
    return;
  }

  uint32_t base = elem_id * vertices_per_tet;
  for (uint32_t i = 0; i < vertices_per_tet; i++) {
    uint32_t vertex_id = tet_to_vertex[base + i];
    uint32_t dst = atomicAdd(&cursors[vertex_id], 1u);
    elems[dst] = elem_id;
  }
}

bool cuda_device_available() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return device_count > 0;
}

uint32_t grid_size(uint32_t n) {
  return (n + block_size - 1) / block_size;
}

uint32_t bits_for_ids(uint32_t count) {
  if (count <= 1) {
    return 0;
  }

  uint32_t max_id = count - 1;
  uint32_t bits = 0;
  while (max_id > 0) {
    max_id >>= 1;
    bits++;
  }
  return bits;
}

uint64_t low_bits_mask(uint32_t bits) {
  if (bits == 0) {
    return 0;
  }
  if (bits == 64) {
    return ~uint64_t(0);
  }
  return (uint64_t(1) << bits) - 1;
}

template <typename cub_call>
nd::array<char, 1, memory::space::gpu> allocate_cub_temp(cub_call call) {
  size_t bytes = 0;
  CUDA_CHECK(call(nullptr, bytes));
  return nd::array<char, 1, memory::space::gpu>({static_cast<uint32_t>(bytes)});
}

Method1Result run_sort_lower_bound_method(
  const nd::array<uint32_t, 2, memory::space::gpu> & tet_to_vertex,
  uint32_t num_vertices) {

  uint32_t num_tets = tet_to_vertex.shape[0];
  uint32_t num_pairs = tet_to_vertex.size();

  Method1Result result;
  result.vertex_bits = bits_for_ids(num_vertices);
  result.elem_bits = bits_for_ids(num_tets);
  result.radix_bits = result.vertex_bits + result.elem_bits;
  FEMTO_ASSERT(result.radix_bits <= 64, "packed vertex-element pair needs more than 64 bits");
  result.sorted_pairs.resize(num_pairs);
  result.offsets.resize(num_vertices + 1);

  nd::array<uint64_t, 1, memory::space::gpu> unsorted_pairs({num_pairs});

  nd::array<uint64_t, 1, memory::space::gpu> sort_scratch({num_pairs});
  auto sort_call = [&](void * temp_storage, size_t & temp_bytes) {
    return cub::DeviceRadixSort::SortKeys(
      temp_storage,
      temp_bytes,
      unsorted_pairs.data(),
      sort_scratch.data(),
      int(num_pairs),
      0,
      int(result.radix_bits));
  };
  nd::array<char, 1, memory::space::gpu> sort_temp = allocate_cub_temp(sort_call);

  GpuTimer timer;

  result.pack_ms = timer.time_ms([&]() {
    pack_vertex_element_pairs<<<grid_size(num_pairs), block_size>>>(
      tet_to_vertex.data(),
      unsorted_pairs.data(),
      num_tets,
      result.elem_bits);
    CUDA_CHECK(cudaGetLastError());
  });

  result.sort_ms = timer.time_ms([&]() {
    CUDA_CHECK(sort_call(sort_temp.data(), sort_temp.sz));
  });

  std::swap(result.sorted_pairs, sort_scratch);

  result.lower_bound_ms = timer.time_ms([&]() {
    build_offsets_by_lower_bound<<<grid_size(num_vertices + 1), block_size>>>(
      result.sorted_pairs.data(),
      result.offsets.data(),
      num_vertices,
      num_pairs,
      result.elem_bits);
    CUDA_CHECK(cudaGetLastError());
  });

  return result;
}

Method2Result run_atomic_scan_method(
  const nd::array<uint32_t, 2, memory::space::gpu> & tet_to_vertex,
  uint32_t num_vertices) {

  uint32_t num_tets = tet_to_vertex.shape[0];
  uint32_t num_pairs = tet_to_vertex.size();

  Method2Result result;
  result.offsets.resize(num_vertices + 1);
  result.elems.resize(num_pairs);
  result.sorted_elems.resize(num_pairs);

  nd::array<uint32_t, 1, memory::space::gpu> counts({num_vertices});
  nd::array<uint32_t, 1, memory::space::gpu> cursors({num_vertices + 1});

  auto scan_call = [&](void * temp_storage, size_t & temp_bytes) {
    return cub::DeviceScan::ExclusiveSum(
      temp_storage,
      temp_bytes,
      counts.data(),
      result.offsets.data(),
      int(num_vertices));
  };
  nd::array<char, 1, memory::space::gpu> scan_temp = allocate_cub_temp(scan_call);

  auto segmented_sort_call = [&](void * temp_storage, size_t & temp_bytes) {
    return cub::DeviceSegmentedSort::SortKeys(
      temp_storage,
      temp_bytes,
      result.elems.data(),
      result.sorted_elems.data(),
      int(num_pairs),
      int(num_vertices),
      result.offsets.data(),
      result.offsets.data() + 1);
  };
  nd::array<char, 1, memory::space::gpu> segmented_sort_temp = allocate_cub_temp(segmented_sort_call);

  GpuTimer timer;

  result.count_ms = timer.time_ms([&]() {
    CUDA_CHECK(cudaMemset(counts.data(), 0, counts.sz * sizeof(uint32_t)));
    count_tet_vertices<<<grid_size(num_tets), block_size>>>(
      tet_to_vertex.data(),
      counts.data(),
      num_tets);
    CUDA_CHECK(cudaGetLastError());
  });

  result.scan_ms = timer.time_ms([&]() {
    CUDA_CHECK(scan_call(scan_temp.data(), scan_temp.sz));
    set_last_offset<<<1, 1>>>(result.offsets.data(), num_vertices, num_pairs);
    CUDA_CHECK(cudaGetLastError());
  });

  result.cursor_copy_ms = timer.time_ms([&]() {
    CUDA_CHECK(cudaMemcpy(
      cursors.data(),
      result.offsets.data(),
      cursors.sz * sizeof(uint32_t),
      cudaMemcpyDeviceToDevice));
  });

  result.scatter_ms = timer.time_ms([&]() {
    scatter_tet_vertices<<<grid_size(num_tets), block_size>>>(
      tet_to_vertex.data(),
      cursors.data(),
      result.elems.data(),
      num_tets);
    CUDA_CHECK(cudaGetLastError());
  });

  result.segmented_sort_ms = timer.time_ms([&]() {
    CUDA_CHECK(segmented_sort_call(segmented_sort_temp.data(), segmented_sort_temp.sz));
  });

  return result;
}

nd::array<uint32_t, 2, memory::space::cpu> compact_tet_to_vertex(const Mesh<> & mesh) {
  FEMTO_ASSERT(mesh.tet.shape[1] >= vertices_per_tet, "tet connectivity has fewer than four columns");

  nd::array<uint32_t, 2, memory::space::cpu> tet_to_vertex({mesh.tet.shape[0], vertices_per_tet});
  for (uint32_t elem_id = 0; elem_id < mesh.tet.shape[0]; elem_id++) {
    for (uint32_t i = 0; i < vertices_per_tet; i++) {
      tet_to_vertex(elem_id, i) = mesh.tet(elem_id, i).index;
    }
  }
  return tet_to_vertex;
}

bool compare_results(
  const Method1Result & method1,
  const Method2Result & method2,
  uint32_t num_vertices,
  uint32_t num_pairs) {

  nd::array<uint32_t, 1, memory::space::cpu> method1_offsets = method1.offsets;
  nd::array<uint32_t, 1, memory::space::cpu> method2_offsets = method2.offsets;
  nd::array<uint64_t, 1, memory::space::cpu> method1_pairs = method1.sorted_pairs;
  nd::array<uint32_t, 1, memory::space::cpu> method2_elems = method2.sorted_elems;

  if (method1_offsets.size() != num_vertices + 1 || method2_offsets.size() != num_vertices + 1) {
    std::cerr << "offset size mismatch" << std::endl;
    return false;
  }

  for (uint32_t i = 0; i <= num_vertices; i++) {
    if (method1_offsets(i) != method2_offsets(i)) {
      std::cerr << "offset mismatch at vertex " << i
                << ": method1 = " << method1_offsets(i)
                << ", method2 = " << method2_offsets(i)
                << std::endl;
      return false;
    }
  }

  if (method1_pairs.size() != num_pairs || method2_elems.size() != num_pairs) {
    std::cerr << "element list size mismatch" << std::endl;
    return false;
  }

  for (uint32_t vertex_id = 0; vertex_id < num_vertices; vertex_id++) {
    uint32_t begin = method1_offsets(vertex_id);
    uint32_t end = method1_offsets(vertex_id + 1);
    uint64_t elem_mask = low_bits_mask(method1.elem_bits);
    for (uint32_t p = begin; p < end; p++) {
      uint32_t method1_vertex = uint32_t(method1_pairs(p) >> method1.elem_bits);
      uint32_t method1_elem = uint32_t(method1_pairs(p) & elem_mask);
      if (method1_vertex != vertex_id || method1_elem != method2_elems(p)) {
        std::cerr << "connectivity mismatch at vertex " << vertex_id
                  << ", entry " << p
                  << ": method1 pair = {" << method1_vertex << ", " << method1_elem << "}"
                  << ", method2 elem = " << method2_elems(p)
                  << std::endl;
        return false;
      }
    }
  }

  return true;
}

void print_timing(const std::string & label, double ms) {
  std::cout << "  " << std::left << std::setw(28) << label
            << std::right << std::setw(10) << std::fixed << std::setprecision(3)
            << ms << " ms" << std::endl;
}

} // namespace

int main() {
  if (!cuda_device_available()) {
    std::cout << "No CUDA-capable device is available; skipping upward connectivity benchmark." << std::endl;
    return 0;
  }

  int device = 0;
  CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  std::cout << "CUDA device: " << props.name << std::endl;

  std::cout << "loading mesh ... " << std::flush;
  Mesh<> mesh = Mesh<>::load(FEMTO_DATA_DIR "/meshes/ball1000000.msh");
  std::cout << "finished" << std::endl;

  std::cout << "mesh has:" << std::endl;
  std::cout << "  " << mesh.vert.shape[0] << " vertices" << std::endl;
  std::cout << "  " << mesh.edge.shape[0] << " edges" << std::endl;
  std::cout << "  " << mesh.tri.shape[0] << " triangles" << std::endl;
  std::cout << "  " << mesh.quad.shape[0] << " quadrilaterals" << std::endl;
  std::cout << "  " << mesh.tet.shape[0] << " tetrahedra" << std::endl;
  std::cout << "  " << mesh.hex.shape[0] << " hexahedra" << std::endl;

  femto::timer host_timer;
  host_timer.start();
  nd::array<uint32_t, 2, memory::space::cpu> h_tet_to_vertex = compact_tet_to_vertex(mesh);
  host_timer.stop();
  double compact_ms = host_timer.elapsed() * 1000.0;

  GpuTimer gpu_timer;
  nd::array<uint32_t, 2, memory::space::gpu> d_tet_to_vertex;
  double h2d_ms = gpu_timer.time_ms([&]() {
    d_tet_to_vertex = h_tet_to_vertex;
  });

  uint32_t num_vertices = mesh.vert.shape[0];
  uint32_t num_tets = h_tet_to_vertex.shape[0];
  uint32_t num_pairs = h_tet_to_vertex.size();

  std::cout << "input:" << std::endl;
  std::cout << "  " << num_pairs << " vertex-element incidences" << std::endl;
  print_timing("host compact tets", compact_ms);
  print_timing("copy tets to GPU", h2d_ms);

  CUDA_CHECK(cudaDeviceSynchronize());

  std::cout << "\nmethod 1: sort {vertex_id, elem_id} + lower bound" << std::endl;
  Method1Result method1 = run_sort_lower_bound_method(d_tet_to_vertex, num_vertices);
  std::cout << "  packed bits: vertex_id = " << method1.vertex_bits
            << ", elem_id = " << method1.elem_bits
            << ", radix sort = [0, " << method1.radix_bits << ")" << std::endl;
  print_timing("pack/copy pairs", method1.pack_ms);
  print_timing("radix sort pairs", method1.sort_ms);
  print_timing("lower-bound offsets", method1.lower_bound_ms);
  print_timing("total", method1.total_ms());

  std::cout << "\nmethod 2: atomics + scan + atomics" << std::endl;
  Method2Result method2 = run_atomic_scan_method(d_tet_to_vertex, num_vertices);
  print_timing("atomic count", method2.count_ms);
  print_timing("exclusive scan offsets", method2.scan_ms);
  print_timing("copy offsets to cursors", method2.cursor_copy_ms);
  print_timing("atomic scatter", method2.scatter_ms);
  print_timing("total unsorted", method2.total_unsorted_ms());
  print_timing("segmented sort elems", method2.segmented_sort_ms);
  print_timing("total sorted", method2.total_sorted_ms());

  bool ok = compare_results(method1, method2, num_vertices, num_pairs);
  std::cout << "\ncorrectness: " << (ok ? "passed" : "failed") << std::endl;

  return ok ? 0 : 1;
}
