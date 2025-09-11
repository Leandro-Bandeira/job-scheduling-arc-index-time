
import json



TIME_STEP = 5
HORIZON = 7


class Arc:
    def __init__(self, job_i, job_j, t):
        self.job_i = job_i
        self.job_j = job_j
        self.t = t
class Node:
    def __init__(self, id, t, job):
        self.id = id
        self.t = t
        self.job = job
        self.in_arcs = []
        self.out_arcs = []
    

class Job:
    def __init__(self, id, p_time):
        self.id = id
        self.p_time = p_time
        self.nodes = []

    def create_nodes(self, T):
        self.nodes = [Node(id, t, self) for t in range(0, T) if t + self.p_time < T]


class Network:
    def __init__(self, nodes: list, m_setup):
       

        self.arcs1 = []

        for node_i in nodes:
            for node_j in nodes:
                if node_i.id == node_j.id: continue

                p_i = node_i.job.p_time
                t = node_j.t
                s_ij = m_setup[node_i.id - 1][node_j.id - 1]

                if  t - p_i - s_ij >= 0:
                    self.arcs1.append(Arc(node_i.job, node_j.job, t))





def main():
    with open("instances/test.json", 'r') as file:
        data = json.load(file)

    jobs = [Job(job['id'], job['p_time']) for job in data['jobs']]
    m_setup = data['setup_times']
    max_setup = max(max(row) for row in m_setup)
    # Calculatin T = max upper bound for Completion time
    T = sum([job.p_time + max_setup for job in jobs])
    print(f"Upper Bound for completion time {T}")
    
    # Creating nodes for each job
    for job in jobs:
        job.create_nodes(T)
    
    job_dummy = Job(0, 0) # Job_dummy id 0 and p_time = 0  
    nodes =  [node for job in jobs for node in job.nodes]
    
    net = Network(nodes, m_setup)
    # Creating nodes
    



















if __name__ == "__main__":
    main()