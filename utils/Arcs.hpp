#pragma once
#include <vector>

class Arc {

  public:
    int from_job;
    int to_job;
    int setup_time;

    Arc(int from_job, int to_job, std::vector<std::vector<double>>& setup_matrix){
      this->from_job = from_job;
      this->to_job = to_job;
      this->setup_time = setup_matrix[from_job, to_job];
      
    }

  private:
};
