#pragma once

#include "util.hpp"

namespace gmsh {

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                    gmsh node numbering conventions                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Line2:                 Line3:                                             //
//                                                                           //
//       v                                                                   //
//       ^                                                                   //
//       |                                                                   //
//       |                                                                   //
// 0-----+-----1 --> u    0----2----1                                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Tri3:                   Tri6:                                             //
//                                                                           //
// v                                                                         //
// ^                                                                         //
// |                                                                         //
// 2                       2                                                 //
// |`\                     |`\                                               //
// |  `\                   |  `\                                             //
// |    `\                 5    `4                                           //
// |      `\               |      `\                                         //
// |        `\             |        `\                                       //
// 0----------1--> u       0-----3----1                                      //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Quad4:                 Quad8:                  Quad9:                     //
//                                                                           //
//       v                                                                   //
//       ^                                                                   //
//       |                                                                   //
// 3-----------2          3-----6-----2           3-----6-----2              //
// |     |     |          |           |           |           |              //
// |     |     |          |           |           |           |              //
// |     +---- | --> u    7           5           7     8     5              //
// |           |          |           |           |           |              //
// |           |          |           |           |           |              //
// 0-----------1          0-----4-----1           0-----4-----1              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Tetrahedron:                          Tetrahedron10:                      //
//                                                                           //
//                   v                                                       //
//                 .                                                         //
//               ,/                                                          //
//              /                                                            //
//            2                                     2                        //
//          ,/|`\                                 ,/|`\                      //
//        ,/  |  `\                             ,/  |  `\                    //
//      ,/    '.   `\                         ,6    '.   `5                  //
//    ,/       |     `\                     ,/       8     `\                //
//  ,/         |       `\                 ,/         |       `\              //
// 0-----------'.--------1 --> u         0--------4--'.--------1             //
//  `\.         |      ,/                 `\.         |      ,/              //
//     `\.      |    ,/                      `\.      |    ,9                //
//        `\.   '. ,/                           `7.   '. ,/                  //
//           `\. |/                                `\. |/                    //
//              `3                                    `3                     //
//                 `\.                                                       //
//                    ` w                                                    //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Hex8:                  Hex20:                  Hex27:                     //
//                                                                           //
//        v                                                                  //
// 3----------2            3----13----2           3----13----2               //
// |\     ^   |\           |\         |\          |\         |\              //
// | \    |   | \          | 15       | 14        | 15   24  | 14            //
// |  \   |   |  \         9  \       11 \        9  \ 20    11 \            //
// |   7------+---6        |   7----19+---6       |   7----19+---6           //
// |   |  +-- |-- | -> u   |   |      |   |       |22 |  26  | 23|           //
// 0---+---\--1   |        0---+-8----1   |       0---+-8----1   |           //
//  \  |    \  \  |         \  17      \  18       \ 17    25 \  18          //
//   \ |     \  \ |         10 |        12|        10 |  21    12|           //
//    \|      w  \|           \|         \|          \|         \|           //
//     4----------5            4----16----5           4----16----5           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Prism6:                      Prism15:               Prism18:              //
//                                                                           //
//            w                                                              //
//            ^                                                              //
//            |                                                              //
//            3                       3                      3               //
//          ,/|`\                   ,/|`\                  ,/|`\             //
//        ,/  |  `\               12  |  13              12  |  13           //
//      ,/    |    `\           ,/    |    `\          ,/    |    `\         //
//     4------+------5         4------14-----5        4------14-----5        //
//     |      |      |         |      8      |        |      8      |        //
//     |    ,/|`\    |         |      |      |        |    ,/|`\    |        //
//     |  ,/  |  `\  |         |      |      |        |  15  |  16  |        //
//     |,/    |    `\|         |      |      |        |,/    |    `\|        //
//    ,|      |      |\        10     |      11       10-----17-----11       //
//  ,/ |      0      | `\      |      0      |        |      0      |        //
// u   |    ,/ `\    |    v    |    ,/ `\    |        |    ,/ `\    |        //
//     |  ,/     `\  |         |  ,6     `7  |        |  ,6     `7  |        //
//     |,/         `\|         |,/         `\|        |,/         `\|        //
//     1-------------2         1------9------2        1------9------2        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Pyr5:                            Pyr13/14:                                //
//                                                                           //
//                4                               4                          //
//              ,/|\                            ,/|\                         //
//            ,/ .'|\                         ,/ .'|\                        //
//          ,/   | | \                      ,/   | | \                       //
//        ,/    .' | `.                   ,/    .' | `.                      //
//      ,/      |  '.  \                ,7      |  12  \                     //
//    ,/       .' w |   \             ,/       .'   |   \                    //
//  ,/         |  ^ |    \          ,/         9    |    11                  //
// 0----------.'--|-3    `.        0--------6-.'----3    `.                  //
//  `\        |   |  `\    \         `\        |      `\    \                //
//    `\     .'   +----`\ - \ ->       `5     .'(13)    10   \               //
//      `\   |    `\     `\  \           `\   |           `\  \              //
//        `\.'      `\     `\`             `\.'             `\`              //
//           1----------------2               1--------8-------2             //
//                     `\                              `\                    //
//                        u                               u                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////  



  using io::Element;

  inline Element::Type element_type(int i){
    switch (i) {
      case  1: return Element::Type::Line2;
      case  8: return Element::Type::Line3;
      case  2: return Element::Type::Tri3;
      case  9: return Element::Type::Tri6;
      case  3: return Element::Type::Quad4;
      case 16: return Element::Type::Quad8;
      case 10: return Element::Type::Quad9;
      case  4: return Element::Type::Tet4;
      case 11: return Element::Type::Tet10;
      case  7: return Element::Type::Pyr5;
      case 19: return Element::Type::Pyr13;
      case 14: return Element::Type::Pyr14;
      case  6: return Element::Type::Prism6;
      case 18: return Element::Type::Prism15;
      case 13: return Element::Type::Prism18;
      case  5: return Element::Type::Hex8;
      case 17: return Element::Type::Hex20;
      case 12: return Element::Type::Hex27;
      default: return Element::Type::Unsupported;
    }
  }

  inline int element_type(Element::Type type){
    switch (type) {
      case Element::Type::Line2:       return 1;
      case Element::Type::Line3:       return 8;
      case Element::Type::Tri3:        return 2;
      case Element::Type::Tri6:        return 9;
      case Element::Type::Quad4:       return 3;
      case Element::Type::Quad8:       return 16;
      case Element::Type::Quad9:       return 10;
      case Element::Type::Tet4:        return 4;
      case Element::Type::Tet10:       return 11;
      case Element::Type::Pyr5:        return 7;
      case Element::Type::Pyr13:       return 19;
      case Element::Type::Pyr14:       return 14;
      case Element::Type::Prism6:      return 6;
      case Element::Type::Prism15:     return 18;
      case Element::Type::Prism18:     return 13;
      case Element::Type::Hex8:        return 5;
      case Element::Type::Hex20:       return 17;
      case Element::Type::Hex27:       return 12;
      case Element::Type::Unsupported: return -1;
    }
    return -1;
  }

  // no permutations necessary here, we use gmsh's numbering

}

namespace vtk {

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                     vtk node numbering conventions                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Line2:                 Line3:                                             //
//                                                                           //
//       v                                                                   //
//       ^                                                                   //
//       |                                                                   //
//       |                                                                   //
// 0-----+-----1 --> u    0----2----1                                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Tri3:                   Tri6:                                             //
//                                                                           //
// v                                                                         //
// ^                                                                         //
// |                                                                         //
// 2                       2                                                 //
// |`\                     |`\                                               //
// |  `\                   |  `\                                             //
// |    `\                 5    `4                                           //
// |      `\               |      `\                                         //
// |        `\             |        `\                                       //
// 0----------1--> u       0-----3----1                                      //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Quad4:                 Quad8:                  Quad9:                     //
//                                                                           //
//       v                                                                   //
//       ^                                                                   //
//       |                                                                   //
// 3-----------2          3-----6-----2           3-----6-----2              //
// |     |     |          |           |           |           |              //
// |     |     |          |           |           |           |              //
// |     +---- | --> u    7           5           7     8     5              //
// |           |          |           |           |           |              //
// |           |          |           |           |           |              //
// 0-----------1          0-----4-----1           0-----4-----1              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Tetrahedron:                          Tetrahedron10:                      //
//                                     (nodes 8, 9 swapped relative to gmsh) //
//                   v                                                       //
//                 .                                                         //
//               ,/                                                          //
//              /                                                            //
//            2                                     2                        //
//          ,/|`\                                 ,/|`\                      //
//        ,/  |  `\                             ,/  |  `\                    //
//      ,/    '.   `\                         ,6    '.   `5                  //
//    ,/       |     `\                     ,/       9     `\                //
//  ,/         |       `\                 ,/         |       `\              //
// 0-----------'.--------1 --> u         0--------4--'.--------1             //
//  `\.         |      ,/                 `\.         |      ,/              //
//     `\.      |    ,/                      `\.      |    ,8                //
//        `\.   '. ,/                           `7.   '. ,/                  //
//           `\. |/                                `\. |/                    //
//              `3                                    `3                     //
//                 `\.                                                       //
//                    ` w                                                    //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Hex8:                  Hex20:                  Hex27:                     //
//                      (diff edge numbers)    (diff edge/face numbers)      //
//        v                                                                  //
// 3----------2            3----10----2           3----10----2               //
// |\     ^   |\           |\         |\          |\         |\              //
// | \    |   | \          | 19       | 18        | 19   23  | 18            //
// |  \   |   |  \        11  \       9  \       11  \ 24    9  \            //
// |   7------+---6        |   7----14+---6       |   7----14+---6           //
// |   |  +-- |-- | -> u   |   |      |   |       |20 |  26  | 21|           //
// 0---+---\--1   |        0---+-8----1   |       0---+-8----1   |           //
//  \  |    \  \  |         \ 15       \  13       \ 15    25 \  13          //
//   \ |     \  \ |         16 |       17 |        16 |  22   17 |           //
//    \|      w  \|           \|         \|          \|         \|           //
//     4----------5            4----12----5           4----12----5           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Prism6:                      Prism15:               Prism18:              //
//                         (diff edge numbers)    (diff edge/face numbers)   //
//            w                                                              //
//            ^                                                              //
//            |                                                              //
//            3                       3                      3               //
//          ,/|`\                   ,/|`\                  ,/|`\             //
//        ,/  |  `\                9  |  11               9  |  11           //
//      ,/    |    `\           ,/    |    `\          ,/    |    `\         //
//     4------+------5         4------10-----5        4------10-----5        //
//     |      |      |         |      12     |        |      12     |        //
//     |    ,/|`\    |         |      |      |        |    ,/|`\    |        //
//     |  ,/  |  `\  |         |      |      |        |  15  |  17  |        //
//     |,/    |    `\|         |      |      |        |,/    |    `\|        //
//    ,|      |      |\        13     |      14       13-----16-----14       //
//  ,/ |      0      | `\      |      0      |        |      0      |        //
// u   |    ,/ `\    |    v    |    ,/ `\    |        |    ,/ `\    |        //
//     |  ,/     `\  |         |  ,6     `8  |        |  ,6     `8  |        //
//     |,/         `\|         |,/         `\|        |,/         `\|        //
//     1-------------2         1------7------2        1------7------2        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Pyr5:                            Pyr13/14:                                //
//                                      (diff edge numbers)                  //
//                4                               4                          //
//              ,/|\                            ,/|\                         //
//            ,/ .'|\                         ,/ .'|\                        //
//          ,/   | | \                      ,/   | | \                       //
//        ,/    .' | `.                   ,/    .' | `.                      //
//      ,/      |  '.  \                ,9      |  12  \                     //
//    ,/       .' w |   \             ,/       .'   |   \                    //
//  ,/         |  ^ |    \          ,/         10   |    11                  //
// 0----------.'--|-3    `.        0--------8-.'----3    `.                  //
//  `\        |   |  `\    \         `\        |      `\    \                //
//    `\     .'   +----`\ - \ -> v     `5     .'(13)     7   \               //
//      `\   |    `\     `\  \           `\   |           `\  \              //
//        `\.'      `\     `\`             `\.'             `\`              //
//           1----------------2               1--------6-------2             //
//                     `\                              `\                    //
//                        u                               u                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////



  using io::Element;

  inline Element::Type element_type(int i){
    switch (i) {
      case  3: return Element::Type::Line2;
      case 21: return Element::Type::Line3;
      case  5: return Element::Type::Tri3;
      case 22: return Element::Type::Tri6;
      case  9: return Element::Type::Quad4;
      case 23: return Element::Type::Quad8;
      case 28: return Element::Type::Quad9;
      case 10: return Element::Type::Tet4;
      case 24: return Element::Type::Tet10;
      case 14: return Element::Type::Pyr5;
      case 27: return Element::Type::Pyr13;
      //case 37: return Element::Type::Pyr14;
      case 13: return Element::Type::Prism6;
      case 26: return Element::Type::Prism15;
      case 32: return Element::Type::Prism18;
      case 12: return Element::Type::Hex8;
      case 25: return Element::Type::Hex20;
      case 29: return Element::Type::Hex27;
      default: return Element::Type::Unsupported;
    }
  }

  inline int element_type(Element::Type type){
    switch (type) {
      case Element::Type::Line2:       return  3;
      case Element::Type::Line3:       return 21;
      case Element::Type::Tri3:        return  5;
      case Element::Type::Tri6:        return 22;
      case Element::Type::Quad4:       return  9;
      case Element::Type::Quad8:       return 23;
      case Element::Type::Quad9:       return 28;
      case Element::Type::Tet4:        return 10;
      case Element::Type::Tet10:       return 24;
      case Element::Type::Pyr5:        return 14;
      case Element::Type::Pyr13:       return 27;
      case Element::Type::Pyr14:       return -1; // not supported by VTK (?)
      case Element::Type::Prism6:      return 13;
      case Element::Type::Prism15:     return 26;
      case Element::Type::Prism18:     return 32;
      case Element::Type::Hex8:        return 12;
      case Element::Type::Hex20:       return 25;
      case Element::Type::Hex27:       return 29;
      case Element::Type::Unsupported: return -1;
    }
    return -1;
  }

  inline std::vector<int> permutation(Element::Type type) {
    switch (type) {

      // many of the elements share the same node ordering
      case Element::Type::Unsupported:
      case Element::Type::Line2:
      case Element::Type::Line3:
      case Element::Type::Tri3:
      case Element::Type::Tri6:
      case Element::Type::Quad4:
      case Element::Type::Quad8:
      case Element::Type::Quad9:
      case Element::Type::Tet4:
      case Element::Type::Pyr5:
      case Element::Type::Prism6:
      case Element::Type::Hex8: {
        return identity_permutation(nodes_per_elem(type));
      }

      // but some of the quadratic elements assign numbers
      // to edge/face nodes in a different order
      case Element::Type::Tet10:
        return {0, 1, 2, 3, 4, 5, 6, 7, 9, 8};
      case Element::Type::Pyr13:
        return {0, 1, 2, 3, 4, 5, 8, 10, 6, 7, 9, 11, 12};
      case Element::Type::Pyr14:
        return {0, 1, 2, 3, 4, 5, 8, 10, 6, 7, 9, 11, 12, 13};
      case Element::Type::Prism15:
        return {0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11};
      case Element::Type::Prism18:
        return {0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11, 15, 17, 16};
      case Element::Type::Hex20:
        return {0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15};
      case Element::Type::Hex27:
        return {0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15, 22, 23, 21, 24, 20, 25, 26};
    }

    return {};
  }

}