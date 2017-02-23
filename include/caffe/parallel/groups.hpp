//
//  Created by Tom Warfel on 1/16/17.
//  Copyright © 2017 Pacific Northwest National Lab.  All rights reserved.
//

#ifndef GROUPS_H
#define GROUPS_H

#include <cstddef>
#include <iostream>
#include <cstdio>
#include <vector>

#ifdef __APPLE__
#define __POPCNT__
#include <popcntintrin.h>
#elif __linux__

#endif


class Groups {
 private:
  std::vector<std::vector<std::vector<int>>> _membership;
  std::vector<std::vector<int>> _group_assignment;

  void inittwiddle(unsigned int m, unsigned int n, std::vector<int> &p);
  int twiddle(int *x, int *y, int *z, std::vector<int> &p);


 public:
  // nodelist
  // peers[node][stage] sorted list of peer-node in each stage for each node.
  // 0th element will be "root" for that group
  std::vector<std::vector<std::vector<int>>> _peers;

  const size_t _num_nodes;


 private:
  const bool _po2;
  const unsigned int _highbit;
  const unsigned int _max_sort_bits;
  std::vector<unsigned int> _ordered_masks;
  int _num_stages;
  const size_t _group_bits;

 public:
  // calculate an estimated "overlap cost" for a specific
  // maskbit pattern
  unsigned int cost(std::vector<unsigned int> &masks);

  Groups(Groups const&);
  void operator = (Groups const &);

  Groups(size_t num_nodes, size_t group_bits);
  Groups(size_t num_nodes);

  int getroot(int node, int stage);
  std::vector<int> get_peers_in_stage(int node, int stage);
  std::vector<std::vector<int>> get_stagelist(int node);
  std::vector<int> get_assigned_group_per_stage(int node);

  inline size_t get_num_groups_per_stage(int stage) {
    if ((stage <0) || (stage >= _num_stages)) {
      std::cerr << "get_num_groups_per_stage() called with invalid stage number." << std::endl;
      exit(1);
    }
    return _membership[stage].size();
  }

  inline size_t get_num_nodes() {
    return _num_nodes;
  }

  inline int get_num_stages() {
    return _num_stages;
  }

 private:
  inline size_t Factorial(size_t x) {
    return ((x == 1)
            ? x
            : (x * Factorial(x - 1)));
  }


  inline size_t n_choose_r(size_t n, size_t r) {
    if (r>n) {
      std::cerr << "Error: n_choose_r - r must be smaller than n" << std::endl;

    }
    if ((r==0) || (n==r)) return 1;
    return(Factorial(n) / (Factorial(r) * Factorial(n-r)));
  }


  inline std::vector<unsigned int> bitlist(size_t foo) {
    std::vector<unsigned int> result;
    for (unsigned int i=0; i<sizeof(size_t); i++) {
      const size_t bit = (0x1UL << i);
      if (foo & bit) {
        result.push_back(i);
      }
    }
    return result;
  }


  inline std::vector<unsigned int> bitlist(unsigned int foo) {
    std::vector<unsigned int> result;
    for (unsigned int i=0; i<sizeof(int); i++) {
      unsigned int bit = (0x1U << i);
      if (foo & bit) {
        result.push_back(i);
      }
    }
    return result;
  }

};


#endif //ORTHOGONALRECURSIVEDIVISION_GROUPS_H
