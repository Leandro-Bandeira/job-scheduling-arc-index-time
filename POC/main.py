
import json
import pyomo.environ as pyo


TIME_STEP = 5
HORIZON = 7


class Arc:
    def __init__(self, src_node, dst_node, arc_type, t):
        self.src_node = src_node
        self.dst_node = dst_node
        self.t = t
        self.type = arc_type

class Node:
    def __init__(self, job, t):
        self.job = job
        self.t = t
        self.in_arcs = []
        self.out_arcs = []

class Job:
    def __init__(self, id, p_time):
        self.id = id
        self.p_time = p_time
        self.nodes = []

    def create_nodes(self, T):
        self.nodes = [Node(self, t) for t in range(0, T) if t + self.p_time < T]


class Network:
    def __init__(self, nodes: list, m_setup, job_dummy):
        
        node_dict = {(n.job.id, n.t): n for n in nodes} # Look up para ver se o node realmente existe
        for dn in job_dummy.nodes:
            node_dict[(dn.job.id, dn.t)] = dn

        """
        O conjunto A_1 representa todos os arcos dado por todas as combinacoes dos Nodes possiveis
        Todo arco, eh representado da forma (i, j, t) tal que ele conecte os nodes (i,t -pi-s_ij)->(j, t)
        Sendo assim, o arco esta entrando no node (j, t) e saindo do node (i, t - pi -s_ij)        
        """
        self.arcs_A1 = []
        for dst in nodes:
            for src in nodes:
                if src.job.id == dst.job.id: continue

                p_i = src.job.p_time
                s_ij = m_setup[src.job.id - 1][dst.job.id - 1]

                t_src = dst.t - p_i - s_ij
                if t_src < 0:
                    continue

                # Verifica se o node realmente existe
                src = node_dict.get((src.job.id, t_src))

                if src is not None:
                    a = Arc(src, dst, 1, dst.t)
                    self.arcs_A1.append(a); src.out_arcs.append(a)
                    dst.in_arcs.append(a)     

        # O conjunto A_2 representa a combinacao de todos os arcos saindo de (dummy, 0) -> (j, 0)
        self.arcs_A2 = []
        init_node_dummy = job_dummy.nodes[0]
        for dst in nodes:
            if dst.t != 0:
                continue
            a = Arc(init_node_dummy, dst, 2, dst.t)
            self.arcs_A2.append(a)
            dst.in_arcs.append(a)
            init_node_dummy.out_arcs.append(a)
        
        self.arcs_A3 = []
        # A3 conecta os jobs da forma (j, t -pj-sj0) -> (0, t)
        for src in nodes:  # src = (j, t_src)
            p_j = src.job.p_time
            t_dst = src.t + p_j
            dst = node_dict.get((0, t_dst))  # dummy no tempo do término
            if dst is not None:
                a = Arc(src, dst, 3, t_dst)
                self.arcs_A3.append(a)
                src.out_arcs.append(a)
                dst.in_arcs.append(a)




def build_model(network, jobs):
    model = pyo.ConcreteModel()

    # Conjunto de arcos (índice único)
    all_arcs = network.arcs_A1 + network.arcs_A2 + network.arcs_A3
    model.A = pyo.Set(initialize=range(len(all_arcs)))

    # Variáveis binárias: x[a] = 1 se arco a é usado
    model.x = pyo.Var(model.A, within=pyo.Binary)

    # Mapeia arcos por job de destino
    arcs_in_by_job = {}
    for idx, arc in enumerate(all_arcs):
        j_id = arc.dst_node.job.id
        if j_id == 0:  # ignora dummy
            continue
        if arc.type == 4:  # exclui A4
            continue
        arcs_in_by_job.setdefault(j_id, []).append(idx)

    # Restrição: cada job recebe exatamente 1 arco de entrada
    def in_flow_rule(m, j):
        return sum(m.x[a] for a in arcs_in_by_job[j]) == 1
    
    
    model.jobs = pyo.Set(initialize=list(arcs_in_by_job.keys()))
    model.in_flow = pyo.Constraint(model.jobs, rule=in_flow_rule)

    # Função objetivo dummy (só para resolver viabilidade)
    model.obj = pyo.Objective(expr=sum(model.x[a] for a in model.A), sense=pyo.maximize)

    return model


def main():
    with open("instances/test.json", 'r') as file:
        data = json.load(file)

    jobs = [Job(job['id'], job['p_time']) for job in data['jobs']]
    m_setup = data['setup_times']
    max_setup = max(max(row) for row in m_setup)
    # Calculatin T = max upper bound for Completion time
    # No pior caso, teremos todos os sendo rodado em uma maquina e entre eles o max setup
    T = sum([job.p_time + max_setup for job in jobs])
    print(f"Upper Bound for completion time {T}")
    
    # Creating nodes for each job
    for job in jobs:
        job.create_nodes(T)
    
    job_dummy = Job(0, 0) # Job_dummy id 0 and p_time = 0
    job_dummy.create_nodes(T)


    nodes =  [node for job in jobs for node in job.nodes]
    
    net = Network(nodes, m_setup, job_dummy)
    # Creating nodes
    
    # Construir modelo Pyomo
    model = build_model(net, jobs)

    # Resolver com HiGHS
    solver = pyo.SolverFactory("highs")
    result = solver.solve(model, tee=True)
   


















if __name__ == "__main__":
    main()