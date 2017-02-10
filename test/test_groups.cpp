//
//  Created by Tom Warfel on 1/16/17.
//  Copyright © 2017 Pacific Northwest National Lab.  All rights reserved.
//

#include <iostream>
#include <cstdio>
#include "caffe/parallel/groups.hpp"

int main(int argc, char *argv[]) {
    const int num_nodes=63;
    Groups group1(num_nodes);
    std::clog << std::endl;
    std::clog << std::endl;
    std::clog << std::endl;

    int num_stages = group1.get_num_stages();
   // int num_groups = group1.get_num_groups_per_stage();

  std::clog << std::endl << " - - - - - - - - - - - - - - - - " << std::endl <<std::endl;
  //  std::clog << "There are " << num_groups << " groups per stage." << std::endl;

    for (int node=0; node<num_nodes; node++) {
        std::clog << "Node " << node << ": " << std::endl;
        for (int stage=0; stage<num_stages; stage++) {
            std::clog << "    Stage " << stage << ": ";

            std::vector<int> peerlist = group1.get_peers_in_stage(node,stage);
            int num_peers = peerlist.size();
            for (int j=0; j<num_peers; j++) {
                std::clog << peerlist[j] << " ";
            }
            if (peerlist[0] == node) {
                std::clog << " (I am root)";
            }
            std::clog << std::endl;
        }
    }


/*
  Groups group2(num_nodes, 2);
  std::clog << std::endl;
  std::clog << std::endl;
  std::clog << std::endl;

  Groups group3(num_nodes, 3);
  std::clog << std::endl;
*/

/*
  Groups group4(num_nodes, 3);
  std::clog << std::endl;

  Groups group5(7, 1);
  std::clog << std::endl;
  Groups group6(15, 2);
  std::clog << std::endl;
  Groups group7(128, 4);
*/
    std::clog << std::endl;
}

