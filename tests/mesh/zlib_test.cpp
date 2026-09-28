#include "zlib.h"

#include "misc/timer.hpp"

#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>

static std::vector<uint8_t> compress(const std::vector<uint8_t>& uncompressed_data) {
  unsigned long uncompressed_bytes = uncompressed_data.size();
  unsigned long compressed_bytes = compressBound(uncompressed_bytes);
  std::vector<uint8_t> compressed_data(compressed_bytes); // allocate enough size for output buffer
  int error = compress((Bytef *)&compressed_data[0], &compressed_bytes, 
                       (Bytef *)&uncompressed_data[0], uncompressed_bytes);

  compressed_data.resize(compressed_bytes);
  return compressed_data;
}

void compression_test() {

  std::vector<uint8_t> bytes = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,63,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,63,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,63,0,0,0,0,0,0,240,63,0,0,0,0,0,0,240,63,0,0,0,0,0,0,240,63,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,64,0,0,0,0,0,0,240,63,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,240,63,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,64,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,64,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,64,0,0,0,0,0,0,240,63,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,64,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,64,0,0,0,0,0,0,240,63,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,64,0,0,0,0,0,0,0,64};
  auto compressed = compress(bytes);

  std::cout << compressed.size() << " bytes" << std::endl;
  int count = 0;
  for (auto c : compressed) {
    std::cout << std::dec << count++ << ": " << std::hex << uint16_t(c) << std::endl;
  }

}

void performance_test() {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::binomial_distribution<uint8_t> distr(4, 0.7);

  std::cout << "buffer size (bytes) | compression (MBps) | file io (MBps) " << std::endl;

  femto::timer stopwatch;
  for (int i = 15; i < 24; i++) {
    std::size_t num_bytes = 1 << i;
    double MB = double(num_bytes) / (1 << 20);
    std::vector< uint8_t > bytes(num_bytes);
    for (int j = 0; j < num_bytes; j++) {
      bytes[j] = distr(gen);
    }

    std::cout << std::setw(19) << num_bytes << " | ";

    stopwatch.start();
    auto compressed = compress(bytes);
    stopwatch.stop();
    std::cout << std::setw(18) << MB / stopwatch.elapsed()  << " | ";

    std::ofstream tmp("tmp");
    stopwatch.start();
    tmp.write((const char *)bytes.data(), num_bytes);
    stopwatch.stop();
    tmp.close();
    std::cout << std::setw(14) << MB / stopwatch.elapsed()  << std::endl;
  }

}

int main() {
  //compression_test();
  performance_test();
}