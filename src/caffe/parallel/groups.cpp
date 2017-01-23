//
//  Created by Tom Warfel on 1/16/17.
//  Copyright © 2017 Pacific Northwest National Lab.  All rights reserved.
//

#include "caffe/parallel/groups.hpp"

unsigned int Groups::cost(std::vector<unsigned int> &masks) {
  unsigned int num_elements=masks.size();
  if (num_elements<2) return 0;
  unsigned int multiplier = (num_elements+1);
  unsigned int sum=0;
  for (int j=1; j<num_elements; j++) {
    const unsigned int common_bits = (masks[j-1] & masks[j]);
#ifdef __APPLE__
    const unsigned int common_count = _mm_popcnt_u32(common_bits);
#else
    const unsigned int common_count = __builtin_popcount(common_bits);
#endif
    sum += (multiplier * common_count);
    multiplier--; // weight early overlaps as more costly than later overlaps
  }
  return sum;
};


int Groups::getroot(int node, int stage) {
  return (_peers[node][stage][0]);

}


std::vector<std::vector<int>> Groups::get_stagelist(int node) {
  std::vector<std::vector<int>> my_stagelist;
  for (unsigned int i=0; i<_num_stages; i++) {
    std::vector<int> my_peerlist(_peers[node][i]);
    my_stagelist.push_back(my_peerlist);
  } 
  return my_stagelist;
}


Groups::Groups(size_t num_nodes, size_t group_bits):
_num_nodes(num_nodes),
_num_groups_per_stage(0x1UL << group_bits),
_po2(!(num_nodes & (num_nodes-1))),
_highbit( 63 - __builtin_clzll(num_nodes)),
_num_stages( n_choose_r(_highbit, group_bits))
{
  if (_po2) {
    std::clog << "number of nodes "
              << _num_nodes << " is a power of 2"
              << std::endl;
  } else {
    std::clog << "number of nodes "
              << _num_nodes << " is NOT a power of 2"
              << std::endl;
  }


  std::clog << "There are "
            << _num_groups_per_stage
            << " groups for each of "
            << _num_stages
            << " stages." << std::endl;


  // assume a "binary" number with "_highbits" digits
  // select all such binary numbers with exactly "group_bits" ones
  // i.e. a Gray code generator, using
  // Philip Chase's M-out-of-N algorithm
  // http://dl.acm.org/citation.cfm?id=362502
  //
  std::vector<unsigned int> masks(_num_stages);

  { std::vector<int> b(_highbit);
    std::vector<int> p(_highbit+2);

    unsigned int stage0=0;
    const unsigned int N_M = static_cast<unsigned int>(_highbit - group_bits);
    for (unsigned int i=0; i<N_M; i++) {
      b[i] = 0;
     // std::clog << "0";
    }

    for (unsigned int i=N_M; i<_highbit; i++) {
      stage0 = (stage0 << 1);
      b[i]=1;
//      std::clog << "1";
      stage0 |= 0x1;
    }
//    std::clog << std::endl;

    //std::clog << "stage0 is " << stage0 << std::endl;

    masks[0]=stage0;

    inittwiddle(static_cast<unsigned int>(group_bits), _highbit, p);

    int stage=1;
    int x, y, z;
    while(!twiddle(&x, &y, &z, p))
    {
      b[x] = 1;
      b[y] = 0;
      unsigned int stage_n=0;
      for(unsigned int i = 0; i != _highbit; i++) {
//        std::clog << (b[i] ? "1" : "0");
        stage_n = stage_n << 1;
        if (b[i]) {
          stage_n |= 0x1;
        }
      }
      masks[stage] = stage_n;
      stage++;
//      std::clog << std::endl;
    }
  }

  // Masks contains the list of all possible bit combinations with
  // "group_bits" ones.

  // Now, brute-force through all possible list orderings using
  // "Heap's algorithm" and calculate an "overlap cost" for each
  // ordering, penalizing overlapping bits more at the early stages
  // than at later stages.  Our goal is to find the order
  // with the lowest item-to-item bit overlap, rotated with
  // the least overlap at the earliest stages.

  unsigned int current_cost = cost(masks);
  _ordered_masks = std::vector<unsigned int> (masks);

  const unsigned int N = masks.size();
  if (current_cost >0) {
    std::vector<unsigned int> c(N, 0);
    unsigned int i = 0;
    while (i < N) {
      if (c[i] < i) {
        if ((~i) & 0x1) {
          unsigned int temp = masks[i];
          masks[i] = masks[0];
          masks[0] = temp;
        } else {
          unsigned int temp = masks[i];
          masks[i] = masks[c[i]];
          masks[c[i]] = temp;
        }

        // evaluate cost of new ordering.
        // if it is the lowest cost so far, then
        // swap with our prior ordering.
        unsigned int newcost = cost(masks);
        if (newcost < current_cost) {
          current_cost = newcost;
          for (int k = 0; k < N; k++) {
            _ordered_masks[k] = masks[k];
          }
        }

        c[i] += 1;
        i = 0;
      } else {
        c[i] = 0;
        i++;
      }
    }
  }
/*
  std::clog << std::endl << "----------" << std::endl << std::endl;
  // ordered_masks now has the minimal cost bit ordering
  // display the low-cost ordering
  for (int i=0; i<N; i++)  {
    const unsigned int t = _ordered_masks[i];
    for(int j = (_highbit-1); j >=0; j--) {
      std::clog << (((t>>j)&0x1) ? "1" : "0");
    }
    std::clog << std::endl;
  }
*/

  // generate list of "1" bits for each mask
  std::vector<std::vector<int>> bitlists;

  for (int i=0; i<N; i++) {
    std::vector<int> stage;
    int bitcounter=0;
    const unsigned int mcopy = _ordered_masks[i];
    for (int j=31; j>=0; j--) {
      if ((mcopy>>j) & 0x1) {
        stage.push_back(j);
        bitcounter++;
        if (bitcounter == group_bits) break;
      }
    }
    bitlists.push_back(stage);
  }

  // generate the communication patterns from the bit ordering
  for (int k=0; k<_num_stages; k++) {
    std::vector<std::vector<int>> stage;
    for (int j = 0; j < _num_groups_per_stage; j++) {
      std::vector<int> group;
      stage.push_back(group);
    }
    _membership.push_back(stage);
  }


  for (int stage = 0; stage < _num_stages; stage++) {
    //std::clog << "Beginning stage " << stage << std::endl;

    for (int node = 0; node < _num_nodes; node++) {
      unsigned int group = 0;
      for (int group_index = 0; group_index < group_bits; group_index++) {
        unsigned int bit = bitlists[stage][group_index];
        group = (group <<1);
        group |= ((((0x1 << bit) & node)) ? 1 : 0);
      }
    /*  std::clog << "At stage " << stage << ", node "  << node
                << " is in group " << group << std::endl;
     */
      _membership[stage][group].push_back(node);
    }
  }


  // peers[node][stage] list of node // 0th element will be "root" for that group
  //std::vector<std::vector<std::vector<int>>> peers[node][stage] list
  for (int node = 0; node < _num_nodes; node++) {
    std::vector<std::vector<int>> stages;
    for (int stage = 0; stage < _num_stages ; stage++) {
      std::vector<int> nodelist;
      stages.push_back(nodelist);
    }
    _peers.push_back(stages);

    std::vector<int> foo;
    _group_assignment.push_back(foo);
  }

  for (int stage = 0; stage < _num_stages; stage++) {
    //std::clog << " stage: " << stage << std::endl;
    for (int groupnum=0; groupnum<_num_groups_per_stage; groupnum++) {
      /*
      std::clog << "    group: " << groupnum << std::endl;
      std::clog << "       nodes: " << std::endl;
      std::clog << "         (there are "
                << _membership[stage][groupnum].size()
                << " nodes in this group in this stage "
                << std::endl;
      */
      for (int node_index_in_group=0; node_index_in_group<_membership[stage][groupnum].size(); node_index_in_group++) {
        //std::clog << _membership[stage][groupnum][node_index_in_group] << " ";
        const int assigning_node =  _membership[stage][groupnum][node_index_in_group];
        _group_assignment[assigning_node].push_back(groupnum);

      /*
        std::clog << "processing node " << assigning_node
                  << " stage " << stage
                  << std::endl;
        std::clog << "        peers: ";
       */
        for (int z=0; z<_membership[stage][groupnum].size(); z++) {
          int node = _membership[stage][groupnum][z];
          _peers[assigning_node][stage].push_back(node);
        //  std::clog << node << " ";
        }
        //std::clog << std::endl;
      }
      //std::clog << std::endl;
    }
  }

/*
  std::clog << std::endl << std::endl << std::endl;


  for (int i=0; i<_num_nodes; i++) {
    std::clog << "Node " << i << ":" << std::endl;
    for (int j=0; j<_num_stages; j++) {
      std::clog << "   Stage " << j << ": " << std::endl;
      std::clog << "      Node peers: ";
      for (unsigned int k=0; k<_peers[i][j].size(); k++) {
        std::clog << _peers[i][j][k] << " ";
      }
      std::clog << std::endl;
    }
    std::clog << std::endl;
  }
*/


}

std::vector<int> Groups::get_peers_in_stage(int node, int stage) {
  std::vector<int> peerlist(_peers[node][stage]);
  return peerlist;
}

std::vector<int> Groups::get_assigned_group_per_stage(int node) {
  std::vector<int> group_IDs(_group_assignment[node]);
  return group_IDs;
}

int Groups::twiddle(int *x, int *y, int *z, std::vector<int> &p) {
  int i, j, k;
  j = 1;
  while (p[j] <= 0) j++;
  if (p[j - 1] == 0) {
    for (i = j - 1;  i != 1; i--) {
      p[i] = -1;
    }
    p[j] = 0;
    *x=0;
    *z = 0;
    p[1] = 1;
    *y = (j - 1);
  } else {
    if (j > 1) {
      p[j - 1] = 0;
    }

    do {
      j++;
    } while (p[j] > 0);
    k = j - 1;
    i = j;

    while (p[i] == 0)  p[i++] = -1;

    if (p[i] == -1) {
      p[i] = p[k];
      *z = (p[k] - 1);
      *x = (i - 1);
      *y = (k - 1);
      p[k] = -1;
    } else {
      if (i == p[0]) {
        return (1);
      } else {
        p[j] = p[i];
        *z = (p[i] - 1);
        p[i] = 0;
        *x = (j - 1);
        *y = (i - 1);
      }
    }
  }
  return (0);
}

void Groups::inittwiddle(unsigned int m, unsigned int n, std::vector<int> &p) {
  int i;
  p[0] = n + 1;
  for (i = 1; i < (n - m + 1); i++) {
    p[i] = 0;
  }

  while (i != (n + 1)) {
    p[i] = (i + m - n);
    i++;
  }

  p[n + 1] = -2;
  if (m == 0) {
    p[1] = 1;
  }
}

