#pragma once

#include "Vertex.hpp"


class Job{
  public:
    int id;
    int p_time; /* Processing time */
    std::vector<Vertex> jobVertices;

    Job(int id, int p_time): id(id), p_time(p_time) {}
    
  
};
