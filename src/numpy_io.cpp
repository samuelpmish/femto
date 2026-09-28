#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "containers/ndarray.hpp"

std::string dtype_string(float) { return "<f4"; }
std::string dtype_string(double) { return "<f8"; }
std::string dtype_string(uint64_t) { return "<u8"; }

template < typename T, uint64_t rank > 
nd::cpu_array<T, rank> load(std::string npy_file) {
  std::ifstream infile(npy_file);
  
  if (!infile) {
    std::cout << "error: file not found : " << npy_file << std::endl;
    exit(1);
  }

  std::string header;
  getline(infile, header);

  // verify that this is actually a NUMPY-formatted file
  if (header.substr(1, 5) != "NUMPY") {
    std::cout << "error: improperly formatted data file" << std::endl;
    exit(1);
  };

  // check that this file contains data of the expected type
  if (header.find(dtype_string(T{})) == std::string::npos) {
    std::cout << "error: incorrect dtype" << std::endl;
    exit(1);
  }

  // parse the shape of the array: first find out where
  // the "shape" key appears in the dictionary
  uint64_t size = 1;
  stack::array<uint32_t, rank> shape;
  auto shape_pos = header.find("shape");
  if (shape_pos == std::string::npos) {
    std::cout << "error: incorrect dtype" << std::endl;
    exit(1);
  } else {
   
    // then look for the following parenthetical expression
    auto open_paren_pos = header.find("(", shape_pos);
    auto close_paren_pos = header.find(")", shape_pos);

    std::stringstream csv(header.substr(open_paren_pos + 1, close_paren_pos - open_paren_pos - 1));

    //  split the comma separated values and each value to an integer
    int count = 0;
    std::string value;
    while (count < rank) {
      getline(csv, value, ',');
      size *= shape[count++] = stoi(value);
    }

    if (count != rank) {
      std::cout << "error: incorrect shape" << std::endl;
      exit(1);
    }

  }

  // the rest of the file should just be the binary data,
  // so we see how many bytes are left in the file, to ensure
  // that it is in agreement with the shape parsed earlier
  auto current_pos = infile.tellg();
  infile.seekg(0, std::ios::end);
  auto end_pos = infile.tellg();
  auto nbytes = end_pos - current_pos;
  if (nbytes != size * sizeof(T)) {
    std::cout << "error: reported shape inconsistent with actual data" << std::endl;
    exit(1);
  }

  // go back to the location where the data section starts
  infile.seekg(current_pos);

  // and read that binary into a buffer of the appropriate size
  nd::cpu_array<T, rank> output(shape);
  infile.read((char*)output.data(), nbytes);

  return output;
}

template nd::cpu_array<double,1> load<double, 1>(std::string);
template nd::cpu_array<double,2> load<double, 2>(std::string);
template nd::cpu_array<double,3> load<double, 3>(std::string);
template nd::cpu_array<double,4> load<double, 4>(std::string);

template nd::cpu_array<uint64_t,1> load<uint64_t, 1>(std::string);
template nd::cpu_array<uint64_t,2> load<uint64_t, 2>(std::string);
template nd::cpu_array<uint64_t,3> load<uint64_t, 3>(std::string);
template nd::cpu_array<uint64_t,4> load<uint64_t, 4>(std::string);

template < uint32_t n >
std::string csv(const stack::array<uint32_t, n> & arr) {
  std::stringstream ss;
  for (int i = 0; i < n; i++) {
    ss << arr[i];
    if (i != n - 1) ss << ", ";
  }
  return ss.str();
}

template < typename T, uint64_t rank > 
void save(std::string npy_file, nd::view<const T, rank> arr) {

  std::ofstream outfile(npy_file);
  
  if (!outfile) {
    std::cout << "error: unable to open file " << npy_file << std::endl;
    exit(1);
  }

  // The number of bytes on the first line needs to be divisible by 64.
  // This header line will always be greater than 66 bytes, and in theory
  // it can be longer than 128, but not in practice for arrays of basic data types).
  // As a result, we'll just hardcode the size as 128 bytes.
  constexpr char initial_value[129] = 
  "\x93NUMPY\x01\x00\x76\00                                               "
  "                                                                      \n";

  std::vector < char > header(initial_value, initial_value + 128);

  constexpr int descr_offset = 10;
  constexpr char descr_format[65] = 
  "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }";

  std::string shape = csv(arr.shape);
  uint16_t last = snprintf(&header[descr_offset], 128 - descr_offset, descr_format, dtype_string(T{}).c_str(), shape.c_str());
  header[descr_offset + last] = ' ';

  outfile.write(&header[0], 128);
  outfile.write((char*)arr.data(), sizeof(T) * arr.size());
  outfile.close();
}

template void save<double, 1>(std::string, nd::view<const double, 1>);
template void save<double, 2>(std::string, nd::view<const double, 2>);
template void save<double, 3>(std::string, nd::view<const double, 3>);
template void save<double, 4>(std::string, nd::view<const double, 4>);

template void save<uint64_t, 1>(std::string, nd::view<const uint64_t, 1>);
template void save<uint64_t, 2>(std::string, nd::view<const uint64_t, 2>);
template void save<uint64_t, 3>(std::string, nd::view<const uint64_t, 3>);
template void save<uint64_t, 4>(std::string, nd::view<const uint64_t, 4>);
