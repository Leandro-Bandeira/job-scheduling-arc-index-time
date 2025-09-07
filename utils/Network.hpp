#pragma once

#include <vector>
#include <unordered_map>

#include "Arcs.hpp"
#include "Vertex.hpp"


class Network{
  public:
    Network();

    /* Given a Vertex we get all arcs enter end out from this Vertex */
    std::unordered_map<Vertex, std::vector<int>> adjOut;
    std::unordered_Map<Vertex, std::vector<int>> adjIn;

    void addArc(const Arc& a){
      this->arcs1.push_back(a);
      adjOut[a.from].push_back(arcs1.size() - 1);
      adjIn[a.to].push_back(arcs1.size() - 1);
    }
    
  private:
    std::vector<Vertex>nodes;
    std::vector<Arc>arc1; /* All combinations between nodes */
        
};
