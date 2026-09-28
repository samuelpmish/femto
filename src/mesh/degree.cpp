#include "femto/mesh/io.hpp"

namespace io{

int degree(Element::Type type){
  switch (type) {
    case Element::Type::Line2:       return 1;
    case Element::Type::Line3:       return 2;
    case Element::Type::Tri3:        return 1;
    case Element::Type::Tri6:        return 2;
    case Element::Type::Quad4:       return 1;
    case Element::Type::Quad8:       return 2;
    case Element::Type::Quad9:       return 2;
    case Element::Type::Tet4:        return 1;
    case Element::Type::Tet10:       return 2;
    case Element::Type::Pyr5:        return 1;
    case Element::Type::Pyr13:       return 2;
    case Element::Type::Pyr14:       return 2;
    case Element::Type::Prism6:      return 1;
    case Element::Type::Prism15:     return 2;
    case Element::Type::Prism18:     return 2;
    case Element::Type::Hex8:        return 1;
    case Element::Type::Hex20:       return 2;
    case Element::Type::Hex27:       return 2;
    case Element::Type::Unsupported: return -1;
  }
  return -1;
}

}