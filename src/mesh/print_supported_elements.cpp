#include "femto/mesh/io.hpp"

#include <iostream>

namespace io{

void print_supported_elements(){
  std::cout << "supported element types are: " << std::endl;
  std::cout << "  Line2" << std::endl;
  std::cout << "  Line3" << std::endl;
  std::cout << "  Tri3" << std::endl;
  std::cout << "  Tri6" << std::endl;
  std::cout << "  Quad4" << std::endl;
  std::cout << "  Quad8" << std::endl;
  std::cout << "  Quad9" << std::endl;
  std::cout << "  Tet4" << std::endl;
  std::cout << "  Tet10" << std::endl;
  std::cout << "  Pyr5" << std::endl;
  std::cout << "  Pyr13" << std::endl;
  std::cout << "  Pyr14" << std::endl;
  std::cout << "  Prism6" << std::endl;
  std::cout << "  Prism15" << std::endl;
  std::cout << "  Prism18" << std::endl;
  std::cout << "  Hex8" << std::endl;
  std::cout << "  Hex20" << std::endl;
  std::cout << "  Hex27" << std::endl;
}

}