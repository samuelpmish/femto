#pragma once

#include <array>
#include <random>
#include "containers/ndarray.hpp"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "fm/operations/random.hpp"

#include "femto/mesh/io.hpp"
#include "femto/ndarray_extensions.hpp"

namespace femto {

static constexpr double constexpr_random_values[128] = {
  0.103453,  0.308101,  0.150106, 0.801639,  0.383286,  0.866077,  0.81112,
  0.72845,   0.231025,  0.61333,  0.0621193, 0.771198,  0.11241,   0.447218,
  0.0642695, 0.937815,  0.535583, 0.861039,  0.14165,   0.630969,  0.172921,
  0.863976,  0.438401,  0.299618, 0.0342435, 0.97485,   0.537538,  0.00516285,
  0.383621,  0.944619,  0.364518, 0.712134,  0.74895,   0.571662,  0.0368889,
  0.632958,  0.394983,  0.957917, 0.972249,  0.643788,  0.73541,   0.882818,
  0.286461,  0.497239,  0.83221,  0.614699,  0.345489,  0.0748839, 0.878876,
  0.207424,  0.909473,  0.51932,  0.798878,  0.729999,  0.718204,  0.273397,
  0.559312,  0.0162799, 0.887746, 0.0457282, 0.752254,  0.233506,  0.17606,
  0.717628,  0.0814569, 0.307863, 0.922314,  0.306939,  0.182486,  0.782653,
  0.539217,  0.6081,    0.913474, 0.875857,  0.781578,  0.765896,  0.884575,
  0.867097,  0.0210844, 0.995452, 0.685812,  0.0649577, 0.787939,  0.474384,
  0.4181,    0.0172197, 0.181456, 0.362906,  0.72326,   0.72763,   0.261708,
  0.202196,  0.166486,  0.513558, 0.754321,  0.72149,   0.623714,  0.31561,
  0.15383,   0.876683,  0.43576,  0.0725313, 0.81812,   0.793922,  0.893233,
  0.288383,  0.504056,  0.840426, 0.236487,  0.0665008, 0.71119,   0.526868,
  0.521341,  0.361478,  0.907656, 0.537346,  0.164165,  0.826701,  0.237305,
  0.543286,  0.387996,  0.674291, 0.303275,  0.14184,   0.0847145, 0.425915,
  0.211271,  0.715746
};

double random() {
  static std::default_random_engine generator;
  static std::uniform_real_distribution< double > dist(0.0, 1.0);
  return dist(generator);
}

double random(double) {
  static std::default_random_engine generator;
  static std::uniform_real_distribution< double > dist(0.0, 1.0);
  return dist(generator);
}

template < uint32_t n >
nd::array< double, n, memory::space::cpu > random(stack::array< uint32_t, n > dimensions) {
  std::default_random_engine generator;
  std::uniform_real_distribution< double > dist(0.0, 1.0);
  
  std::size_t size = nd::product(dimensions);
  nd::array< double, n, memory::space::cpu > output(dimensions);
  for (int i = 0; i < size; i++) {
    output.values[i] = dist(generator);
  }
  return output;
}

}

constexpr float h = 0.5;

// note: the node position arrays below are not
// the official reference-element positions!

//  2           
//  |`\.        
//  |  `\.      
//  5    `4     
//  |      `\.  
//  |        `\.
//  0-----3-----1
constexpr fm::vec3 triangle_nodes[6] = {
  {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, 
  {h, 0, 0}, {h, h, 0}, {0, h, 0}
};

//  3-----6-----2 
//  |           | 
//  |           | 
//  7     8     5 
//  |           | 
//  |           | 
//  0-----4-----1 
constexpr fm::vec3 quadrilateral_nodes[9] = {
  {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
  {h, 0, 0}, {1, h, 0}, {h, 1, 0}, {0, h, 0},
  {h, h, 0}
};

//                          
//             2            
//           ,/|`\.         
//         ,/  |  `\.       
//       ,6    '.   `5      
//     ,/       8     `\.   
//   ,/         |       `\. 
//  0--------4--'.---------1 
//   `\.         |      ,/' 
//      `\.      |    ,9    
//         `7.   '. ,/      
//            `\. |/        
//               `3         
//
constexpr fm::vec3 tetrahedron_nodes[10] = {
  {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
  {h, 0, 0}, {h, h, 0}, {0, h, 0}, {0, 0, h}, {0, h, h}, {h, 0, h}
};

//                4              
//              ,/|\.            
//            ,/ .'|\.           
//          ,/   | | \.          
//        ,/    .' | `.          
//      ,7      |  12  \.        
//    ,/       .'   |   \.       
//  ,/         9    |    11      
// 0--------6-.'----3    `.      
//   `\        |      `\    \.   
//     `5     .'(13)    10   \.  
//       `\   |           `\  \. 
//         `\.'             `\`  
//            1--------8-------2 
constexpr fm::vec3 pyramid_nodes[14] = {
  {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.1, 0.2, 1},
  {h, 0, 0}, {0, h, 0}, {0, 0, h}, {1, h, 0}, {h, 0, h}, {h, 1, 0}, {h, h, h}, {0, h, h},
  {h, h, 0}
};

//          3          
//        ,/|`\.       
//      12  |  13      
//    ,/    |    `\.   
//   4------14-----5   
//   |      8      |   
//   |    ,/|`\    |   
//   |  15  |  16  |   
//   |,/    |    `\|   
//   10-----17-----11  
//   |      0      | 
//   |    ,/ `\    | 
//   |  ,6     `7  | 
//   |,/         `\| 
//   1------9------2 
constexpr fm::vec3 prism_nodes[18] = {
  {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1},
  {h, 0, 0}, {0, h, 0}, {0, 0, h}, {h, h, 0}, {1, 0, h}, {0, 1, h},
  {h, 0, 1}, {0, h, 1}, {h, h, 1}, {h, 0, h}, {0, h, h}, {h, h, h}  
};

//                     
//  3----13----2       
//  |\         |\. 
//  |15    24  | 14    
//  9  \ 20    11 \.  
//  |   7----19+---6   
//  |22 |  26  | 23|   
//  0---+-8----1   |   
//   \ 17    25 \  18  
//   10 |  21    12|   
//     \|         \|   
//      4----16----5   
//
constexpr fm::vec3 hexahedron_nodes[27] = {
  {0, 0, 0}, {1.3, 0, 0}, {1, 1, 0}, {0, 1, 0}, 
  {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
  {h, 0, 0}, {0, h, 0}, {0, 0, h}, {1, h, 0},
  {1, 0, h}, {h, 1, 0}, {1, 1, h}, {0, 1, h},
  {h, 0, 1}, {0, h, 1}, {1, h, 1}, {h, 1, 1},
  {h, h, 0}, {h, 0, h}, {0, h, h}, {1, h, h},
  {h, 1, h}, {h, h, 1}, {h, h, h}
};

inline std::vector<int> range(int n) {
  std::vector<int> values(n);
  for (int i = 0; i < n; i++) { values[i] = i; }
  return values;
}

inline io::Mesh single_element_mesh(io::Element::Type type) {
  int n = nodes_per_elem(type);
  switch (type) {
    case io::Element::Type::Tri3:
    case io::Element::Type::Tri6:
      return io::Mesh{
        std::vector<fm::vec3>(triangle_nodes, triangle_nodes + n),
        std::vector<io::Element>{{type, range(n)}}
      };
    case io::Element::Type::Quad4:
    case io::Element::Type::Quad8:
    case io::Element::Type::Quad9:
      return io::Mesh{
        std::vector<fm::vec3>(quadrilateral_nodes, quadrilateral_nodes + n),
        std::vector<io::Element>{{type, range(n)}}
      };
    case io::Element::Type::Tet4:
    case io::Element::Type::Tet10:
      return io::Mesh{
        std::vector<fm::vec3>(tetrahedron_nodes, tetrahedron_nodes + n),
        std::vector<io::Element>{{type, range(n)}}
      };
    case io::Element::Type::Pyr5:
    case io::Element::Type::Pyr13:
    case io::Element::Type::Pyr14:
      return io::Mesh{
        std::vector<fm::vec3>(pyramid_nodes, pyramid_nodes + n),
        std::vector<io::Element>{{type, range(n)}}
      };
    case io::Element::Type::Prism6:
    case io::Element::Type::Prism15:
    case io::Element::Type::Prism18:
      return io::Mesh{
        std::vector<fm::vec3>(prism_nodes, prism_nodes + n),
        std::vector<io::Element>{{type, range(n)}}
      };
    case io::Element::Type::Hex8:
    case io::Element::Type::Hex20:
    case io::Element::Type::Hex27:
      return io::Mesh{
        std::vector<fm::vec3>(hexahedron_nodes, hexahedron_nodes + n),
        std::vector<io::Element>{{type, range(n)}}
      };

    case io::Element::Type::Line2:
    case io::Element::Type::Line3:
    case io::Element::Type::Unsupported:
      return io::Mesh{};
  }

  return io::Mesh{};
}