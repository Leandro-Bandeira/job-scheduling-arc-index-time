import gurobipy as gp
import pandas as pd
import plotly.express as px
import os
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
SLOTS_PER_DAY = data['slots_per_day']


def create_json_output(solution_jobs):
    """
    Cria um arquivo JSON com o agendamento detalhando os atrasos (Tardiness).
    """
    init_date = datetime.strptime(INIT_DATE, '%Y-%m-%d %H:%M')
    output_data = []

    for job in solution_jobs:
        # Cálculos de tempo
        start_delta = timedelta(minutes=job['start_slot'] * TIME_STEP)
        end_delta = timedelta(minutes=(job['start_slot'] + job['p_time']) * TIME_STEP)
        
        # Cálculo de datas para Due Date
        # Se due_date_slot for muito grande (infinito), tratamos visualmente
        if job['due_date_slot'] >= SLOTS_PER_DAY * 10: 
            due_date_str = "Sem Prazo"
        else:
            due_delta = timedelta(minutes=job['due_date_slot'] * TIME_STEP)
            due_date_str = (init_date + due_delta).strftime('%Y-%m-%d %H:%M:%S')

        start_time = init_date + start_delta
        end_time = init_date + end_delta
        
        output_data.append({
            'job_id': job['id'],
            "start_slot": job['start_slot'],
            "end_slot": job['start_slot'] + job['p_time'],
            "due_date_slot": job['due_date_slot'],
            "tardiness_slots": job['tardiness'],
            'start_datetime': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_datetime': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'due_datetime': due_date_str,
            'duration_minutes': job['p_time'] * TIME_STEP,
            'tardiness_minutes': job['tardiness'] * TIME_STEP
        })

    output_filename = f"POC/results/machine_schedule_output.json"
    # Garante que o diretório existe
    if not os.path.exists("POC/results"):
        os.makedirs("POC/results")

    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\nResultado detalhado salvo em {output_filename}")


def create_interactive_gantt(solution_jobs, machine_name):
    """
    Gera um Gráfico de Gantt Interativo (.html) com informações de TARDINESS.
    """
    if not solution_jobs:
        print("Nenhum job para gerar gráfico.")
        return

    init_date = datetime.strptime(INIT_DATE, '%Y-%m-%d %H:%M')
    gantt_data = []
    
    for job in solution_jobs:
        if job['id'] == 0: continue

        start_delta = timedelta(minutes=job['start_slot'] * TIME_STEP)
        end_delta = timedelta(minutes=(job['start_slot'] + job['p_time']) * TIME_STEP)
        
        start_dt = init_date + start_delta
        end_dt = init_date + end_delta

        # Formatação do Due Date para o hover
        if job['due_date_slot'] >= 99999:
            dd_text = "N/A"
        else:
            dd_text = str(job['due_date_slot'])

        # Criação dos dados do hover
        gantt_data.append({
            "Job ID": str(job['id']),
            "Start": start_dt,
            "Finish": end_dt,
            "Machine": machine_name,
            "Duration (min)": job['p_time'] * TIME_STEP,
            "Start Slot": job['start_slot'],
            "Release Date": job.get('release_date_slot', 'N/A'),
            "Due Date Slot": dd_text,
            "Tardiness (min)": job['tardiness'] * TIME_STEP, # Atraso em minutos
            "Status": "ATRASADO" if job['tardiness'] > 0 else "No Prazo"
        })

    if not gantt_data:
        return

    df = pd.DataFrame(gantt_data)

    # Define cores baseadas no status (Atrasado = Vermelho, No Prazo = Azul/Verde)
    color_map = {"ATRASADO": "#EF553B", "No Prazo": "#636EFA"}

    fig = px.timeline(
        df, 
        x_start="Start", 
        x_end="Finish", 
        y="Machine",
        color="Status", # Agora a cor indica se atrasou ou não
        color_discrete_map=color_map,
        text="Job ID",
        hover_data={
            "Machine": False, "Status": False,
            "Start": True, "Finish": True,
            "Duration (min)": True,
            "Tardiness (min)": True, # Mostra o atraso
            "Due Date Slot": True,
            "Release Date": True,
        },
        title=f"Agendamento - {machine_name} (Foco em Tardiness)"
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_traces(textposition='inside', marker_line_color='black', marker_line_width=1)
    fig.update_layout(
        xaxis=dict(title='Linha do Tempo', tickformat='%Y-%m-%d %H:%M', gridcolor='lightgray'),
        bargap=0.2, height=350, showlegend=True
    )

    output_dir = "POC/results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_filename = f"{output_dir}/gantt_tardiness_{machine_name}.html"
    fig.write_html(output_filename)
    print(f"Gráfico Interativo de Tardiness salvo em: {output_filename}")

class Arc:
    def __init__(self, src_node, dst_node, arc_type, t, setup_time=0):
        self.src_node = src_node # Nó de origem
        self.dst_node = dst_node # Nó de destino
        self.t = t # Tempo de chegado no nó de destino
        self.type = arc_type # Tipo do arco (1, 2 ou 3)
        self.setup_time = setup_time # Tempo de setup associado ao arco

class Node:
    def __init__(self, job, t):
        self.job = job
        self.t = t
        self.in_arcs = [] # Arcos que chegam neste nó
        self.out_arcs = [] # Arcos que saem deste nó

class Job:
    def __init__(self, id, p_time, release_date_slot=0, due_date_slot=SLOTS_PER_DAY):
        self.id = id # Id from summary idetification
        self.idx = None # Id for matematical problems
        self.p_time = p_time
        self.release_date_slot = release_date_slot
        self.due_date_slot = due_date_slot
        self.nodes = []

    def create_nodes(self, T):
        t_max_start = T - self.p_time
        self.nodes = [Node(self, t) for t in range(self.release_date_slot, t_max_start + 1)]


class Network:
    def __init__(self, nodes: list, m_setup, job_dummy, jobs: list):
        
        # Cria dicionario para acesso rapido de todos os nos
        self.node_dict = {(n.job.idx, n.t): n for n in nodes}
        for dn in job_dummy.nodes:
            self.node_dict[(dn.job.idx, dn.t)] = dn

        # --- ARCOS A1: Job -> Job ---
        # O Arco A1 (i, j, t) do job i -> j chegando em t em j, j começa no tempo t
        print("Criando arcos A1")
        self.arcs_A1 = []
        for dst_node in nodes:
            job_dst = dst_node.job
            for job_src in jobs:
                if job_src.idx == job_dst.idx or job_src.idx == 0:
                    continue

                p_i = job_src.p_time
                # Acessa a matriz m_setup com o índice correto (0 a n-1).
                # Usamos o índice do mapa 'job_map' criado na função main.
                s_ij = m_setup[job_src.idx - 1][job_dst.idx - 1]

                # Encontramos o tempo de inicio do node source
                t_src = dst_node.t - p_i - s_ij
                
                # Caso exista esse nó criamos o arco
                if t_src >= 0:
                    src_node = self.node_dict.get((job_src.idx, t_src))
                    if src_node is not None:
                        a = Arc(src_node, dst_node, 1, dst_node.t, s_ij)
                        self.arcs_A1.append(a)
                        src_node.out_arcs.append(a)
                        dst_node.in_arcs.append(a)

        # --- ARCOS A2: Dummy -> Job (Apenas em t=0) ---
        print("Criando arcos A2 (Inicialização)")
        self.arcs_A2 = []
        for job_dst in jobs:
            if job_dst.idx == 0: continue
            
            # Pega APENAS o nó do job no tempo 0
            dst_node = self.node_dict.get((job_dst.idx, 0))
            
            # Pega o nó dummy no tempo 0
            src_dummy = self.node_dict.get((0, 0))
            
            if dst_node and src_dummy:
                a = Arc(src_dummy, dst_node, 2, dst_node.t, setup_time=0)
                self.arcs_A2.append(a)
                src_dummy.out_arcs.append(a)
                dst_node.in_arcs.append(a)
        
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

        # --- ARCOS A4: Ociosidade
        # Conecta Node(j, t) -> Node(j, t+1)
        print("Criando arcos A4 (Espera no Job)")
        self.arcs_A4 = []
        for job in jobs:
            # Ignora o Dummy na espera (Dummy só serve de fonte/sumidouro agora)
            if job.idx == 0: continue 
            
            # Ordena nós pelo tempo
            job_nodes = sorted(job.nodes, key=lambda n: n.t)
            for i in range(len(job_nodes) - 1):
                src_node = job_nodes[i]
                dst_node = job_nodes[i+1]
                
                # Arco de espera: custo 0 (ou 1 se quiser penalizar espera), setup 0
                # Ele apenas avança o tempo sem processar
                a = Arc(src_node, dst_node, 4, dst_node.t, setup_time=0)
                self.arcs_A4.append(a)
                src_node.out_arcs.append(a)
                dst_node.in_arcs.append(a)

        print(f"Tamanho de A_1: {len(self.arcs_A1)}")
        print(f"Tamanho de A_2: {len(self.arcs_A2)}")
        print(f"Tamanho de A_3: {len(self.arcs_A3)}")
        print(f"Tamanho de A_4: {len(self.arcs_A4)}")



def build_model(network, jobs, count_machines, time_capacity_data=None):

    model = gp.Model("machine_scheduling")

    # --- Preparação dos Dados ---
    all_arcs = network.arcs_A1 + network.arcs_A2 + network.arcs_A3 + network.arcs_A4
    network.all_arcs = all_arcs
    arc_indices = range(len(all_arcs))
    
    # Mapeia arcos por job
    arcs_in_by_job = {}
    for idx, arc in enumerate(all_arcs):
        j_idx_in = arc.dst_node.job.idx
        if j_idx_in != 0: # Ignora arcos que chegam no job dummy
            arcs_in_by_job.setdefault(j_idx_in, []).append(idx)

    # Mapeia arcos por nó
    arcs_in_by_node_key = {}
    arcs_out_by_node_key = {}
    for idx, arc in enumerate(all_arcs):
        node_key_dst = (arc.dst_node.job.idx, arc.dst_node.t)
        node_key_src = (arc.src_node.job.idx, arc.src_node.t)
        arcs_in_by_node_key.setdefault(node_key_dst, []).append(idx)
        arcs_out_by_node_key.setdefault(node_key_src, []).append(idx)

     # --- Definição das Restrições ---
    job_indices = [j.idx for j in jobs]

    # --- Definição das Variáveis ---
    x = model.addVars(arc_indices, vtype=GRB.BINARY, name="x")
    e = model.addVars(job_indices, vtype=GRB.BINARY, name="e")
    #C_max = model.addVar(vtype=GRB.CONTINUOUS, name="C_max")
    # Definição do tardiness dado por max(0, C_j - d_j)
    T = model.addVars(job_indices, vtype=GRB.CONTINUOUS, name="T")

    # Definindo valores de custo
    arc_costs = {}

    for arc_idx, arc in enumerate(all_arcs):
        if arc.type == 3 or arc.type == 4:
            arc_costs[arc_idx] = 0
        else:
            arc_costs[arc_idx] = arc.dst_node.job.p_time + arc.dst_node.t
   
    

    print("Adicionando restrição: Um início por job")
    for j_idx in job_indices:
        incoming_real_arcs = [a for a in arcs_in_by_job.get(j_idx, []) if all_arcs[a].type != 4]
        model.addConstr(gp.quicksum(x[a] for a in incoming_real_arcs) + e[j_idx] == 1, name=f"only_one_in_{j_idx}")
    
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
    
    # Definição do tardiness

    for j_indx in job_indices:
        completion_time_expr = gp.quicksum(x[a] * arc_costs[a] for a in arcs_in_by_job.get(j_indx, []))
        job_due_date = next(j.due_date_slot for j in jobs if j.idx == j_indx)

        model.addConstr(T[j_indx] >= completion_time_expr - job_due_date, name=f"tardiness_def_{j_indx}")
        model.addConstr(T[j_indx] >= 0, name=f"tardiness_nonneg_{j_indx}")



    """
    print("Adicionando restrição: Capacidade de tempo")
    if time_capacity_data:
        windows = time_capacity_data.get('slots', [])
        capacities = time_capacity_data.get('slots_capacity', [])
        
        for k, window in enumerate(windows):
            window_start, window_end = window
            max_cap = capacities[k]
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
    """
    e_penalty_weight = 10e9
    print(f"Adicionando Função Objetivo (Penalidade e = {e_penalty_weight})")
    
    # Soma de todas as variáveis 'e' (jobs não alocados)
    total_penalty = gp.quicksum(e[j] * e_penalty_weight for j in job_indices)
    total_tardiness = gp.quicksum(T[j] for j in job_indices)
    
    # --- Definição da Função Objetivo ---
    model.setObjective(total_tardiness + total_penalty, GRB.MINIMIZE)
    #model.setObjective(total_completion_time + total_penalty, GRB.MINIMIZE)
    return model, x, e, total_tardiness



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
        jobs_machine = [Job(j['job_id'], j['processing_slots'], j.get('release_date_slot', 0), j.get('deadline_date_slot', SLOTS_PER_DAY)) for j in jobs_data if j['assigned_machine_id'] == machine_id]
        print(f"Quantia de jobs na máquina {machine_name}: {len(jobs_machine)}")

        n = len(jobs_machine)
        # Mapeia ID do job para um índice de 0 a n-1
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
        max_use_per_day = machine_time_capacity['slots_capacity'][0]
        print("max_use_per_day ", max_use_per_day)
        time_needed = sum(j.p_time for j in jobs_machine) + (n - 1) * max_setup if n > 0 else 0
        days_needed = math.ceil(time_needed / max_use_per_day)
        # Horizonte T: Adicionamos uma margem de segurança para evitar infactibilidade com release dates tardios
        max_release = max([j.release_date_slot for j in jobs_machine]) if jobs_machine else 0
        print("max_release ", max_release)
        T = math.ceil(days_needed * 24 * 60 / TIME_STEP) + max_release + SLOTS_PER_DAY

        print(f"Upper Bound for completion time {T}")
        # Define idx (1 a n) e cria nós
        for i, job in enumerate(jobs_machine):
            job.idx = i + 1
            job.create_nodes(T)
        
        # Criacao do job dummy
        job_dummy = Job(0, 0)
        job_dummy.idx = 0
        job_dummy.create_nodes(T)
        
        nodes = [node for job in jobs_machine for node in job.nodes]
        
        # Passa a lista completa de jobs (reais + dummy)
        all_jobs_for_net = jobs_machine + [job_dummy]
        net = Network(nodes, m_setup, job_dummy, all_jobs_for_net)
        
        model, x, e, C_max = build_model(net, jobs_machine, 1, machine_time_capacity)
        model.write("modelo.lp")
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            print("\n-------------------------------------------")
            print("Solução ótima encontrada!")
            
            # Cálculo dos resultados reais para exibição
            # total_tardiness_val = sum(T_vars[j.idx].X for j in jobs_machine)
            obj_val = model.ObjVal
            unscheduled_count = sum(1 for job in jobs_machine if e[job.idx].X > 0.5)
            
            # Se houver penalidade, subtrai do obj para mostrar só o atraso real
            real_tardiness = obj_val - (unscheduled_count * 10e9)
            
            print(f"Total Tardiness (Soma): {real_tardiness:.2f} slots")
            print(f"Operações não atendidas: {unscheduled_count}")
            print("-------------------------------------------")

            solution_jobs = []
            for a_idx, arc in enumerate(net.all_arcs):
                # Extrai apenas jobs produtivos
                if x[a_idx].X > 0.5 and arc.dst_node.job.idx != 0 and arc.type in [1, 2]:
                    job = arc.dst_node.job
                    start_slot = arc.dst_node.t
                    end_slot = start_slot + job.p_time
                    
                    # Calcula o atraso individual para o relatório
                    tardiness_val = max(0, end_slot - job.due_date_slot)

                    solution_jobs.append({
                        'id': job.id,
                        'p_time': job.p_time,
                        'start_slot': start_slot,
                        'release_date_slot': job.release_date_slot,
                        'due_date_slot': job.due_date_slot,
                        'tardiness': tardiness_val
                    })
            
            # Ordena pelo slot de início
            solution_jobs.sort(key=lambda j: j['start_slot'])

            # 1. Cria a saída JSON com datetimes
            create_json_output(solution_jobs)

            # 2. Cria o gráfico de Gantt contínuo
            create_interactive_gantt(solution_jobs, machine_name)

        elif model.Status == GRB.INFEASIBLE:
            print("\nO modelo é inviável.")
            model.computeIIS()
            model.write("model_iis.ilp")
            print("Arquivo 'model_iis.ilp' gerado para depuração.")
        else:
            print(f"\nStatus do solver: {model.Status}")

if __name__ == "__main__":
    main()

