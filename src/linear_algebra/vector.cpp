#include "linear_algebra/vector.hpp"

#include "femto/assert.hpp"

#include <fstream>

namespace femto {

  using u32 = uint32_t;

  vector ones(int n) {
    vector a(n);
    for (int i = 0; i < n; i++) {
      a[i] = 1.0;
    }
    return a;
  }

  vector zeros(int n) {
    vector a(n);
    for (int i = 0; i < n; i++) {
      a[i] = 0.0;
    }
    return a;
  }

  vector import_vector(std::string filename) {
    std::ifstream infile(filename, std::ios::binary);

    if (infile) {
      // std::cout << "file found: " << filename << std::endl;
      infile.seekg(0, std::ios::end);
      std::streampos filesize = infile.tellg();
      infile.seekg(0, std::ios::beg);

      vector buffer(filesize / sizeof(double));
      infile.read((char*)&buffer[0], filesize);
      infile.close();
      return buffer;
    } else {
      std::cout << "file not found: " << filename << std::endl;
      exit(1);
    }
  }

}