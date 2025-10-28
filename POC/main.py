import gurobipy as gp
import pandas as pd

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from gurobipy import GRB
import json
import math 
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Param, Var, Binary, NonNegativeReals, Set, Constraint, ConstraintList, Objective, minimize, maximize,value, SolverFactory

with open("POC/instances/job_scheduling_input.json", 'r') as file:
    data = json.load(file)

TIME_STEP = data['time_step']
HORIZON = data['n_days']
INIT_DATE = data['init_date']

class Arc:
    def __init__(self, src_node, dst_node, arc_type, t, setup_time=0):
        self.src_node = src_node
        self.dst_node = dst_node
        self.t = t
        self.type = arc_type
        self.setup_time = setup_time

class Node:
    def __init__(self, job, t):
        self.job = job
        self.t = t
        self.in_arcs = []
        self.out_arcs = []

class Job:
    def __init__(self, id, p_time):
        self.id = id # Id from summary idetification
        self.idx = None # Id for matematical problems
        self.p_time = p_time
        self.nodes = []

    def create_nodes(self, T):
        t_max_start = T - self.p_time
        self.nodes = [Node(self, t) for t in range(0, t_max_start + 1)]


class Network:
    def __init__(self, nodes: list, m_setup, job_dummy, jobs: list):
        self.node_dict = {(n.job.idx, n.t): n for n in nodes}
        for dn in job_dummy.nodes:
            self.node_dict[(dn.job.idx, dn.t)] = dn

        # --- ARCOS A1: Job -> Job ---
        print("Criando arcos A1")
        self.arcs_A1 = []
        for dst_node in nodes:
            job_dst = dst_node.job
            for job_src in jobs:
                if job_src.idx == job_dst.idx or job_src.idx == 0:
                    continue

                p_i = job_src.p_time
                # CORREÇÃO: Acessa a matriz m_setup com o índice correto (0 a n-1).
                # Usamos o índice do mapa 'job_map' criado na função main.
                s_ij = m_setup[job_src.idx - 1][job_dst.idx - 1]
                t_src = dst_node.t - p_i - s_ij
                
                if t_src >= 0:
                    src_node = self.node_dict.get((job_src.idx, t_src))
                    if src_node is not None:
                        a = Arc(src_node, dst_node, 1, dst_node.t, s_ij)
                        self.arcs_A1.append(a)
                        src_node.out_arcs.append(a)
                        dst_node.in_arcs.append(a)

        # --- ARCOS A2: Dummy -> Primeiro Job (com setup zero) ---
        print("Criando arcos A2")
        self.arcs_A2 = []
        init_node_dummy = self.node_dict.get((0, 0))
        if init_node_dummy:
            # Itera sobre jobs reais para criar arcos que começam em t=0
            for job_dst in jobs:
                 if job_dst.idx == 0: continue
                 # O setup inicial é zero, então o nó de destino deve estar em t=0
                 dst_node = self.node_dict.get((job_dst.idx, 0))
                 if dst_node:
                     a = Arc(init_node_dummy, dst_node, 2, dst_node.t, setup_time=0)
                     self.arcs_A2.append(a)
                     dst_node.in_arcs.append(a)
                     init_node_dummy.out_arcs.append(a)
        
        # --- ARCOS A3: Último Job -> Dummy ---
        print("Criando arcos A3")
        self.arcs_A3 = []
        for src_node in nodes:
            p_j = src_node.job.p_time
            t_dst = src_node.t + p_j
            dst_node = self.node_dict.get((0, t_dst))
            if dst_node is not None:
                a = Arc(src_node, dst_node, 3, t_dst, setup_time=0)
                self.arcs_A3.append(a)
                src_node.out_arcs.append(a)
                dst_node.in_arcs.append(a)

        print(f"Tamanho de A_1: {len(self.arcs_A1)}")
        print(f"Tamanho de A_2: {len(self.arcs_A2)}")
        print(f"Tamanho de A_3: {len(self.arcs_A3)}")



def build_model(network, jobs, count_machines, time_capacity_data=None):
    """
    Constrói o modelo de otimização usando a biblioteca Gurobipy.
    """
    model = gp.Model("job_scheduling")

    # --- Preparação dos Dados ---
    all_arcs = network.arcs_A1 + network.arcs_A2 + network.arcs_A3
    network.all_arcs = all_arcs
    arc_indices = range(len(all_arcs))
    
    # Mapeia arcos por job (usando o índice matemático job.idx)
    arcs_in_by_job = {}
    for idx, arc in enumerate(all_arcs):
        # CORREÇÃO: Usar job.idx em vez de job.id
        j_idx_in = arc.dst_node.job.idx
        if j_idx_in != 0: # Ignora arcos que chegam no job dummy
            arcs_in_by_job.setdefault(j_idx_in, []).append(idx)

    # Mapeia arcos por nó (usando o índice matemático job.idx)
    arcs_in_by_node_key = {}
    arcs_out_by_node_key = {}
    for idx, arc in enumerate(all_arcs):
        # CORREÇÃO: Usar job.idx em vez de job.id
        node_key_dst = (arc.dst_node.job.idx, arc.dst_node.t)
        node_key_src = (arc.src_node.job.idx, arc.src_node.t)
        arcs_in_by_node_key.setdefault(node_key_dst, []).append(idx)
        arcs_out_by_node_key.setdefault(node_key_src, []).append(idx)

    # --- Definição das Variáveis ---
    x = model.addVars(arc_indices, vtype=GRB.BINARY, name="x")
    C_max = model.addVar(vtype=GRB.CONTINUOUS, name="C_max")

    # --- Definição das Restrições ---
    # CORREÇÃO: Usar job.idx na lista de jobs
    job_indices = [j.idx for j in jobs]
    

    print("Adicionando restrição: Um início por job")
    for j_idx in job_indices:
        # Garante que a chave existe antes de somar
        model.addConstr(gp.quicksum(x[a] for a in arcs_in_by_job.get(j_idx, [])) == 1, name=f"only_one_in_{j_idx}")
    
    print("Adicionando restrição: Quantidade de máquinas")
    model.addConstr(gp.quicksum(x[a] for a in arc_indices if all_arcs[a].type == 2) == count_machines, name="use_machines")

    print("Adicionando restrição: Conservação de fluxo")
    # Itera sobre os nós REAIS (onde o fluxo deve ser conservado)
    for node_key in network.node_dict:
        j_idx, t = node_key
        # A conservação não se aplica aos nós dummy (índice 0)
        if j_idx == 0: 
            continue

        in_arcs = arcs_in_by_node_key.get(node_key, [])
        out_arcs = arcs_out_by_node_key.get(node_key, [])
        
        # Fluxo de entrada deve ser igual ao fluxo de saída
        model.addConstr(gp.quicksum(x[a] for a in in_arcs) == gp.quicksum(x[a] for a in out_arcs), name=f"flow_cons_{j_idx}_{t}")
    
    print("Adicionando restrição: Cálculo do Makespan")
    for j_idx in job_indices:
        completion_time_expr = gp.quicksum(x[a] * all_arcs[a].t for a in arcs_in_by_job.get(j_idx, []))
        model.addConstr(C_max >= completion_time_expr, name=f"makespan_{j_idx}")
    
    if time_capacity_data:
        windows = time_capacity_data.get('slots', [])
        capacities = time_capacity_data.get('slots_capacity', [])

        for k, window in enumerate(windows):
            window_start, window_end = window
            min_cap, max_cap = capacities[k]
            total_usage_in_window = gp.LinExpr()

            for a_idx in arc_indices:
                arc = all_arcs[a_idx]
                
                # Ignora arcos que não representam um job real
                if arc.dst_node.job.idx == 0:
                    continue

                # 1. Encontra o intervalo de processamento do job [inicio, fim)
                job_start_time = arc.t
                job_p_time = arc.dst_node.job.p_time
                # O fim é exclusivo, ex: se começa em 10 e dura 2, usa os slots 10 e 11.
                # O intervalo é [10, 12).
                job_end_time = job_start_time + job_p_time 

                # 2. Calcula a sobreposição (overlap) com a janela [window_start, window_end)
                # A fórmula é: max(0, min(fim_A, fim_B) - max(inicio_A, inicio_B))
                
                overlap_start = max(job_start_time, window_start)
                overlap_end = min(job_end_time, window_end)
                
                # 3. Este é o coeficiente: o tempo real gasto DENTRO da janela
                proc_in_window = max(0, overlap_end - overlap_start)

                # 4. Adiciona à expressão APENAS se houver sobreposição
                if proc_in_window > 0:
                    total_usage_in_window.add(x[a_idx], proc_in_window)

            model.addConstr(
                total_usage_in_window <= max_cap, 
                name=f"max_cap_window_{k}_{window_start}"
            )

    # --- Definição da Função Objetivo ---
    model.setObjective(C_max, GRB.MINIMIZE)

    return model, x, C_max

def display_solution_and_gantt(solution_jobs, machine_name):
    """
    Gera um Gantt estático (PNG) com um único item no eixo Y (a máquina)
    e, no eixo X (tempo), os blocos de processamento dos jobs.
    Não há prints.
    """
    if not solution_jobs:
        return

    # Ordena por início e computa timestamps absolutos (sem imprimir)
    solution_jobs.sort(key=lambda j: j['start_slot'])
    init_date = datetime.strptime(INIT_DATE, '%Y-%m-%d %H:%M')
    for job in solution_jobs:
        start_delta = timedelta(minutes=job['start_slot'] * TIME_STEP)
        end_delta = timedelta(minutes=(job['start_slot'] + job['p_time']) * TIME_STEP)
        job['start_time'] = init_date + start_delta
        job['end_time'] = init_date + end_delta

    # Monta as janelas (start, duração) no eixo X (matplotlib usa número de dias)
    spans = []
    for job in solution_jobs:
        start_num = mdates.date2num(job['start_time'])
        end_num = mdates.date2num(job['end_time'])
        spans.append((start_num, end_num - start_num))

    # Figura
    fig, ax = plt.subplots(figsize=(12, 3))

    # Desenha todos os jobs na mesma "faixa" Y (ex.: y=10 com altura 9)
    y_base, y_height = 10, 9
    ax.broken_barh(spans, (y_base, y_height), edgecolor='black')

    # Deixa apenas 1 tick no eixo Y com o nome da máquina
    ax.set_yticks([y_base + y_height / 2])
    ax.set_yticklabels([machine_name])

    # Eixo X em datetime
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    # Grid sutil no X
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)

    # Rótulos / título (sem prints)
    ax.set_xlabel('Tempo')
    ax.set_ylabel('Máquina')
    ax.set_title(f'Gráfico de Gantt - {machine_name}')

    fig.autofmt_xdate()
    plt.tight_layout()

    # Salva PNG (sem imprimir nada)
    out_name = f"gantt_{machine_name}.png"
    plt.savefig(out_name, dpi=150)
    plt.close(fig)

def main():
    machines = data['machines']
    jobs_data = data['jobs']
    setups_data = data['setups']

    for machine in machines:
        machine_name = machine['machine_name']
        if machine_name != 'coldboxgasado_coldbox4':
            continue
        
        machine_time_capacity  = machine['time_capacity']
        machine_id = machine['machine_id']
        machine_name = machine['machine_name']
        jobs_machine = [Job(j['job_id'], j['processing_slots']) for j in jobs_data if j['assigned_machine_id'] == machine_id]
        print(f"Quantia de jobs na máquina {machine_name}: {len(jobs_machine)}")

        n = len(jobs_machine)
        # CORREÇÃO: Mapeia ID do job para um índice de 0 a n-1
        job_map_to_idx = {job.id: i for i, job in enumerate(jobs_machine)}

        # Matriz de setup n x n (apenas para jobs reais)
        m_setup = [[0 for _ in range(n)] for _ in range(n)]

        for setup in setups_data:
            if setup['machine_id'] == machine_id:
                # Pula setups envolvendo o dummy, pois a regra é setup inicial zero
                if setup['from_job_id'] == 0:
                    continue
                from_idx = job_map_to_idx.get(setup['from_job_id'])
                to_idx = job_map_to_idx.get(setup['to_job_id'])
                if from_idx is not None and to_idx is not None:
                    m_setup[from_idx][to_idx] = setup['setup_slots']
        
        max_setup = max(max(row) for row in m_setup) if m_setup else 0
        max_use_per_day = machine_time_capacity['slots_capacity'][0][1]
        """
        T = 0
        for j in jobs_machine:
            T += j.p_time + max_setup
            if T >= max_use_per_day:
                T = 24 * 60 / TIME_STEP
        """
        time_needed = sum(j.p_time for j in jobs_machine) + (n - 1) * max_setup if n > 0 else 0
        days_needed = math.ceil(time_needed / max_use_per_day)
        
        T  = math.ceil(days_needed * 24 * 60 / TIME_STEP)
        #T = int(T_slots)
        #T = int(290)
        print(f"Upper Bound for completion time {T}")
        #T = 800
        # Define idx (1 a n) e cria nós
        for i, job in enumerate(jobs_machine):
            job.idx = i + 1
            job.create_nodes(T)
        
        job_dummy = Job(0, 0)
        job_dummy.idx = 0
        job_dummy.create_nodes(T)
        
        nodes = [node for job in jobs_machine for node in job.nodes]
        
        # Passa a lista completa de jobs (reais + dummy)
        all_jobs_for_net = jobs_machine + [job_dummy]
        net = Network(nodes, m_setup, job_dummy, all_jobs_for_net)
        
        model, x, C_max = build_model(net, jobs_machine, 1, machine_time_capacity)
        model.write("modelo.lp")
        model.optimize()

        # ... (O restante do seu código para processar a solução permanece o mesmo) ...
        if model.Status == GRB.OPTIMAL:
            print("\n-------------------------------------------")
            print("Solução ótima encontrada!")
            print(f"Makespan (C_max): {C_max.X:.2f} slots = {C_max.X * TIME_STEP / 60:.2f} horas")
            print("-------------------------------------------")

            # --- Extração da Solução para o Gráfico ---
            solution_jobs = []
            for a in x:
                if x[a].X > 0.5:
                    arc = net.all_arcs[a]
                    # Arcos de entrada em jobs reais (tipos 1 e 2) definem o tempo de início
                    if arc.dst_node.job.idx != 0:
                        job = arc.dst_node.job
                        start_slot = arc.t - job.p_time
                        solution_jobs.append({
                            'id': job.id,
                            'p_time': job.p_time,
                            'start_slot': start_slot
                        })
            
            display_solution_and_gantt(solution_jobs, machine_name)

        elif model.Status == GRB.INFEASIBLE:
            print("\nO modelo é inviável.")
            model.computeIIS()
            model.write("model_iis.ilp")
            print("Arquivo 'model_iis.ilp' gerado para depuração.")
        else:
            print(f"\nStatus do solver: {model.Status}")

if __name__ == "__main__":
    main()

