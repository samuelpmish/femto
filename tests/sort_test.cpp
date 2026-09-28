#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
  std::size_t n = 100000;
  std::size_t m = 3;
  std::size_t threads = 1;
  uint64_t seed = 0;
};

std::size_t parse_size(const char * text, const char * flag) {
  char * end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0') {
    std::cerr << "error: expected an unsigned integer after " << flag << std::endl;
    std::exit(1);
  }
  return static_cast<std::size_t>(value);
}

void print_usage(const char * program) {
  std::cout << "usage: " << program << " [-n tuple-count] [-m tuple-size] [-t threads] [-s seed]" << std::endl;
}

Options parse_args(int argc, char ** argv) {
  Options options;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    }

    auto next_arg = [&]() {
      if (++i >= argc) {
        std::cerr << "error: expected an argument after " << arg << std::endl;
        std::exit(1);
      }
      return argv[i];
    };

    if (arg == "-n") {
      options.n = parse_size(next_arg(), "-n");
    } else if (arg == "-m") {
      options.m = parse_size(next_arg(), "-m");
    } else if (arg == "-t") {
      options.threads = parse_size(next_arg(), "-t");
    } else if (arg == "-s") {
      options.seed = parse_size(next_arg(), "-s");
    } else {
      std::cerr << "error: unknown argument " << arg << std::endl;
      print_usage(argv[0]);
      std::exit(1);
    }
  }

  if (options.m == 0) {
    std::cerr << "error: tuple size must be positive" << std::endl;
    std::exit(1);
  }

  if (options.threads == 0) {
    std::cerr << "error: thread count must be positive" << std::endl;
    std::exit(1);
  }

  return options;
}

template <typename T>
double elapsed_ms(T start, T stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

template <std::size_t M>
uint64_t checksum(const std::vector<std::array<uint32_t, M>> & tuples) {
  uint64_t sum = 0;
  for (const auto & tuple : tuples) {
    for (uint32_t value : tuple) {
      sum = (sum * 1315423911u) ^ value;
    }
  }
  return sum;
}

template <std::size_t M>
std::size_t sort_tuples(std::vector<std::array<uint32_t, M>> & tuples, std::size_t requested_threads) {
  if (requested_threads == 1 || tuples.size() < 2) {
    std::sort(tuples.begin(), tuples.end());
    return 1;
  }

  std::size_t thread_count = std::min(requested_threads, tuples.size());
  std::vector<std::size_t> chunk_begin(thread_count + 1);
  for (std::size_t i = 0; i <= thread_count; i++) {
    chunk_begin[i] = (i * tuples.size()) / thread_count;
  }

  std::vector<std::thread> workers;
  workers.reserve(thread_count);
  for (std::size_t i = 0; i < thread_count; i++) {
    workers.emplace_back([&, i]() {
      std::sort(tuples.begin() + chunk_begin[i], tuples.begin() + chunk_begin[i + 1]);
    });
  }
  for (auto & worker : workers) {
    worker.join();
  }

  for (std::size_t width = 1; width < thread_count; width *= 2) {
    workers.clear();
    for (std::size_t i = 0; i + width < thread_count; i += 2 * width) {
      std::size_t left = chunk_begin[i];
      std::size_t mid = chunk_begin[i + width];
      std::size_t right = chunk_begin[std::min(i + 2 * width, thread_count)];
      workers.emplace_back([&, left, mid, right]() {
        std::inplace_merge(tuples.begin() + left, tuples.begin() + mid, tuples.begin() + right);
      });
    }
    for (auto & worker : workers) {
      worker.join();
    }
  }

  return thread_count;
}

template <std::size_t M>
int run_sort_test(const Options & options) {
  std::mt19937 rng(options.seed);
  std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(options.n));

  std::vector<std::array<uint32_t, M>> tuples(options.n);
  for (auto & tuple : tuples) {
    for (uint32_t & value : tuple) {
      value = dist(rng);
    }
  }

  auto start = std::chrono::steady_clock::now();
  std::size_t threads_used = sort_tuples(tuples, options.threads);
  auto stop = std::chrono::steady_clock::now();

  std::cout << "sorted " << options.n << " " << M << "-tuples" << std::endl;
  std::cout << "threads: " << threads_used << std::endl;
  std::cout << "sort time: " << elapsed_ms(start, stop) << " ms" << std::endl;
  std::cout << "checksum: " << checksum(tuples) << std::endl;

  return 0;
}

} // namespace

int main(int argc, char ** argv) {
  Options options = parse_args(argc, argv);

  switch (options.m) {
    case 1: return run_sort_test<1>(options);
    case 2: return run_sort_test<2>(options);
    case 3: return run_sort_test<3>(options);
    case 4: return run_sort_test<4>(options);
    case 5: return run_sort_test<5>(options);
    case 6: return run_sort_test<6>(options);
    default:
      std::cerr << "error: tuple size must be between 1 and 6" << std::endl;
      return 1;
  }
}
