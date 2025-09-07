#pragma once


class Vertex {
  public:
    int job; /* job index (0 = dummy) */
    int time; /* Time index */
    
    /* Each Vertex is given by (job_index, time_index) */
    Vertex(int j, int t): job(j), time(t) {}

  private:
    
  
};
