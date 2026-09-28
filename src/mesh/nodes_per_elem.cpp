#include "femto/mesh/io.hpp"

namespace io{

int nodes_per_elem(Element::Type type){
  switch (type) {
    case Element::Type::Line2:       return 2;
    case Element::Type::Line3:       return 3;
    case Element::Type::Tri3:        return 3;
    case Element::Type::Tri6:        return 6;
    case Element::Type::Quad4:       return 4;
    case Element::Type::Quad8:       return 8;
    case Element::Type::Quad9:       return 9;
    case Element::Type::Tet4:        return 4;
    case Element::Type::Tet10:       return 10;
    case Element::Type::Pyr5:        return 5;
    case Element::Type::Pyr13:       return 13;
    case Element::Type::Pyr14:       return 14;
    case Element::Type::Prism6:      return 6;
    case Element::Type::Prism15:     return 15;
    case Element::Type::Prism18:     return 18;
    case Element::Type::Hex8:        return 8;
    case Element::Type::Hex20:       return 20;
    case Element::Type::Hex27:       return 27;
    case Element::Type::Unsupported: return -1;
  }
  return -1;
}

}