#include <iostream>
#include <unordered_map>
#include <vector>

#include "../utils/Vertex.hpp"
#include "../utils/Job.hpp"
#include "../utils/Arcs.hpp"

int main(){

  
  std::vector<int>jobs_index = {0, 1, 2, 3, 4};
  std::vector<int>times_index = {0, 1, 2, 3, 4};
  
  std::vector<Job>jobs;
  std::vector<Arc> arcs;
  /* Set setup constraint */

  std::vector<std::vector<double>> matrix_setup;
  
  for(size_t i = 0; i < jobs_index.size(); i++){
    std::vector<double>current_setups;
    
    for(size_t j = 0; j < jobs_index.size(); j++){
      current_setups.push_back(0);  
    }
    matrix_setup.push_back(current_setups);
  }


  /* Define Jobs */
  for(size_t j = 0; j < jobs_index.size(); j++){
      std::vector<Vertex> currentJobVertices;
      for(size_t t = 0; t < times_index.size(); t++){
        currentJobVertices.push_back(Vertex(j, t));  
      }
      Job currentJob(j, 2);
      currentJob.jobVertices = currentJobVertices;
      jobs.push_back(currentJob);
      
  }

  /* Define Arcs */
  for(size_t i = 0; i < jobs.size(); i++){

    for(size_t j = 0; j < jobs.size(); j++){
      
    }
  }  
  
  return 0;
}
