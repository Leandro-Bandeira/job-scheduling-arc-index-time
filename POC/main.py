
import json
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Param, Var, Binary, NonNegativeReals, Set, Constraint, ConstraintList, Objective, minimize, maximize,value, SolverFactory

with open("instances/job_scheduling_input.json", 'r') as file:
    data = json.load(file)

TIME_STEP = data['time_step']
HORIZON = data['n_days']

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
        self.idx = None
        self.p_time = p_time
        self.nodes = []

    def create_nodes(self, T):
        self.nodes = [Node(self, t) for t in range(0, T) if t + self.p_time < T]


class Network:
    def __init__(self, nodes: list, m_setup, job_dummy):
        
        self.node_dict = {(n.job.id, n.t): n for n in nodes} # Look up para ver se o node realmente existe

        for dn in job_dummy.nodes:
            self.node_dict[(dn.job.id, dn.t)] = dn

        """
        O conjunto A_1 representa todos os arcos dado por todas as combinacoes dos Nodes possiveis
        Todo arco eh representado da forma (i, j, t) tal que ele conecte os nodes (i,t -pi-s_ij)->(j, t)
        Sendo assim, o arco esta entrando no node (j, t) e saindo do node (i, t - pi -s_ij)        
        """
        self.arcs_A1 = []
        for dst in nodes:
            for src in nodes:
                if src.job.id == dst.job.id: continue

                p_i = src.job.p_time
                s_ij = m_setup[src.job.idx][dst.job.idx]

                t_src = dst.t - p_i - s_ij
                if t_src < 0:
                    continue

                # Verifica se o node realmente existe
                src = self.node_dict.get((src.job.id, t_src))

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
            dst = self.node_dict.get((0, t_dst))  # dummy no tempo do término
            if dst is not None:
                a = Arc(src, dst, 3, t_dst)
                self.arcs_A3.append(a)
                src.out_arcs.append(a)
                dst.in_arcs.append(a)




def build_model(network, jobs, count_machines):
    model = ConcreteModel()

    # Conjunto de arcos (índice único)
    all_arcs = network.arcs_A1 + network.arcs_A2 + network.arcs_A3
    all_nodes_keys = list(network.node_dict.keys())
    
    
    # Mapeia arcos por job de destino e saida
    arcs_in_by_job = {}
    arcs_out_by_job = {}

    # Mapeia arcos por node de destino e saida (para o Flow Conservation Constraint)
    arcs_in_by_node_key = {}
    arcs_out_by_node_key = {}

    for idx, arc in enumerate(all_arcs):
        j_id_in = arc.dst_node.job.id
        j_id_out = arc.src_node.job.id

        arcs_in_by_job.setdefault(j_id_in, []).append(idx)
        arcs_out_by_job.setdefault(j_id_out, []).append(idx)

        node_key_dst = (arc.dst_node.job.id, arc.dst_node.t)
        node_key_src = (arc.src_node.job.id, arc.src_node.t)
        arcs_in_by_node_key.setdefault(node_key_dst, []).append(idx)
        arcs_out_by_node_key.setdefault(node_key_src, []).append(idx)

    # Conjuntos
    model.jobs = Set(initialize=list(arcs_in_by_job.keys()))
    model.A = Set(initialize=range(len(all_arcs)))

    # Variáveis binárias: x[a] = 1 se arco a é usado
    model.x = Var(model.A, within=Binary)
    model.C_max = Var(within=NonNegativeReals)
    model.V = Set(dimen=2, initialize=all_nodes_keys)

    # Constraints
    model.only_one_in = ConstraintList()
    model.use_machines = ConstraintList()
    model.flow_conservation = ConstraintList()
    model.makespan = ConstraintList()

    print("Um inicio por job")
    # Each job must be processed exactly once
    for j in model.jobs:
        expr = sum(model.x[a] for a in arcs_in_by_job[j])
        model.only_one_in.add(expr == 1)
    
    print("Quantia de maquinas")
    # Cada arco que sai do job dummy indica uma maquina sendo utilizada
    model.use_machines.add(sum(model.x[a] for a in range(len(all_arcs)) if all_arcs[a].type == 1) == count_machines)

    print("Flow Conservation")
    # ------------------ C3: Flow Conservation (Node Balance) ------------------

    # Nodes that require flow balance (all actual job nodes)
    nodes_for_flow = []
    for j, t in model.V:
        is_actual_job_node = (j != 0)
        
        # Balance only applies to actual job nodes. 
        # Source (0, 0) and sinks (0, t > 0) are excluded.
        if is_actual_job_node:
             nodes_for_flow.append((j, t))
    
    # The actual job nodes (j > 0) are the internal nodes:
    # They receive an arc (A1 or A2) and must send one (A1 or A3).
    for j, t in nodes_for_flow:
        # Indices of arcs entering node (j, t)
        in_arcs = arcs_in_by_node_key.get((j, t), [])
        # Indices of arcs leaving node (j, t)
        out_arcs = arcs_out_by_node_key.get((j, t), [])

        expr_in = sum(model.x[a] for a in in_arcs)
        expr_out = sum(model.x[a] for a in out_arcs)
        
        # Balance constraint: Flow In = Flow Out
        model.flow_conservation.add(expr_in - expr_out == 0)

    print("Makespan")
    for j in model.jobs:
        expr = sum(model.x[a] * all_arcs[a].t for a in arcs_in_by_job[j])
        model.makespan.add(model.C_max >= expr)
    # Função objetivo dummy (só para resolver viabilidade)
    model.obj = Objective(expr=model.C_max, sense=minimize)

    return model


def main():
    
    machines = data['machines']
    jobs = data['jobs']
    setups = data['setups']

    for machine in machines:
        if machine['machine_name'] != 'coldboxgasado_coldbox4':
            continue

        machine_id = machine['machine_id']

        jobs_machine = [Job(job['job_id'], job['processing_slots']) for job in jobs if job['assigned_machine_id'] == machine_id]
        setups_machine = [setup for setup in setups if setup['machine_id'] == machine_id]
        
        """
        m_setup = data['setup_times']

        max_setup = max(max(row) for row in m_setup)
        """
        n = len(jobs_machine)
        for i, j in enumerate(jobs_machine):
            j.idx = i
        # matriz n x n zerada (setup de i -> j)
        m_setup = [[0 for _ in range(n)] for _ in range(n)]
        print()
        max_setup = 0
        # Calculatin T = max upper bound for Completion time
        # No pior caso, teremos todos os sendo rodado em uma maquina e entre eles o max setup
        T = sum([job.p_time + max_setup for job in jobs_machine])
        print(f"Upper Bound for completion time {T}")
        
        # Creating nodes for each job
        for job in jobs_machine:
            job.create_nodes(T)
        
        job_dummy = Job(0, 0) # Job_dummy id 0 and p_time = 0
        job_dummy.create_nodes(T)


        nodes =  [node for job in jobs_machine for node in job.nodes]
        
        net = Network(nodes, m_setup, job_dummy)
        # Creating nodes
        
        # Construir modelo Pyomo
        model = build_model(net, jobs_machine, 1)

        # Resolver com HiGHS
        solver = SolverFactory("gurobi")
        result = solver.solve(model, tee=True)
        # Check solution status
        if result.solver.status == pyo.SolverStatus.ok and result.solver.termination_condition == pyo.TerminationCondition.optimal:
            # **Call the new function to save the Gantt data**
            print("Resolvido")
        else:
            print(f"\nSolver failed to find an optimal solution. Status: {result.solver.status}, Condition: {result.solver.termination_condition}")
        

















if __name__ == "__main__":
    main()