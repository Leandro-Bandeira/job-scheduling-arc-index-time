import argparse
import re
import pathlib


import pandas as pd
import numpy as np
from unidecode import unidecode

from datetime import datetime, timedelta
from typing import Dict, Tuple

weekday_idx = {
    "segunda": 0, "terca": 1, "terça": 1, "quarta": 2,
    "quinta": 3, "sexta": 4, "sabado": 5, "sábado": 5, "domingo": 6
}

def normalize_string(value) -> str:
    """Normaliza string: remove acentos, minúsculas, sem espaços."""
    if pd.isna(value):
        return ""
    return unidecode(str(value)).lower().replace(" ", "")

def parse_turnos_to_list(value):
    if pd.isna(value):
        return []
    if value == 'geral':
        return [0]
    
    return [int(x.strip()) for x in str(value).split(',') if x.strip().isdigit()]
def get_total_minutes(hora_str):
    hora = datetime.strptime(hora_str, "%H:%M")
    total_minutes = hora.hour * 60 + hora.minute
    return total_minutes

# Função responsavel por calcular o inicio e fim de um dataframe
# 1. Df é um dataframe ordenado em prioridade de deadline e ordenação
# 2. work_days é o conjunto de dias possiveis de trabalho dado o dia inicial ate a sexta-feira da mesma semana
# 3. Para cada linha do dataframe, caso seu not_before_date seja menor ou igual ao dia no sequenciamento, calcula-se o inicio e o fim
# 4. Se no fim existirem linhas que não foram calculadas inicio e fim, é porque não é possível produzir o job naquela semana

def calculate_init_end_singleMachine(
        df:pd.DataFrame,
        work_days:list,
        machine_shift,
        turnos_df,
        current_machine_time_capacity,
        kp_macho_data,
        setup_times_df,
        resource
    ) -> pd.DataFrame:    
    
    df['not_sequence'] = False # Define que nenhum job foi sequenciado
    
    # Vamos organizar do init_dt ate a sexta feira da mesma semana
    # Cria uma mascara dos jobs que estao sequenciados em df que possuem not before date >= init_date
    # Cria o sequenciamento para esses jobs e adiciona uma nova coluna chamada sequenciado
    # Mesma coisa para cada dia
    for day in work_days:
        day_date = pd.Timestamp(day).date()
        base_dt = day
        
        day_work_shifts = list()
        range_work_shifts = list()
        for _, row in turnos_df.iterrows():
            if row['turno'] == 'geral': continue
            if int(row['turno']) in machine_shift.iloc[0] and weekday_idx[row['dia']] == day_date.weekday():
                ini  = get_total_minutes(row["inicio"])
                brk1 = get_total_minutes(row["intervalo_inicio"])
                brk2 = get_total_minutes(row["intervalo_fim"])
                end  = get_total_minutes(row["fim"])

                day_work_shifts += [ini, brk2]
                range_work_shifts.append((ini, brk1))
                range_work_shifts.append((brk2, end))

        day_work_shifts.sort()
        
        
        mask = (df['not_before_date'].dt.date <= day_date) & (~df['not_sequence'])
        current_df = df.loc[mask]

        t = base_dt + timedelta(minutes=day_work_shifts.pop(0))
        total_use_day = 0
        for pos, (idx, row) in enumerate(current_df.iterrows()):
            if row['tempo_total'] == '-':
                row['tempo_total'] = 1

            setup_time = 0
            if pos > 0:
                prev_row = current_df.iloc[pos - 1] 
                
                p_macho  = int(prev_row['_kf_macho'])
                
                
                c_macho = int(row['_kf_macho'])
                p_key = (p_macho, resource)
                c_key = (c_macho, resource)
                p_info = kp_macho_data.get(p_key)
                c_info = kp_macho_data.get(c_key)
                # precisa ter info para ambos
                if p_info and c_info:
                    
                    # se trocar para um macho que NÃO está na lista de "sem setup"
                    if c_macho not in p_info.get('not_setup', []):
                        
                        mask_setup = (
                        (setup_times_df['de_config'] == p_info['config']) &
                        (setup_times_df['para_config'] == c_info['config']) &
                        (setup_times_df['recurso'] == resource)
                        )
                    
                        if mask_setup.any():
                            # se houver múltiplas linhas, pega a primeira (ou poderia usar .min())
                            setup_time = setup_times_df.loc[mask_setup, 'setup_time_min'].iloc[0]
            
            st = 0 if pd.isna(setup_time) else int(setup_time)
            t = t + timedelta(minutes=st)
            p = float(row["qtd_moldes"]) * float(row["tempo_ciclo"])
            can_work = False
            delta_min = (t - base_dt).total_seconds() // 60            

            for (start, end) in range_work_shifts:
                if delta_min >= start and delta_min <= end and  delta_min + p <= end:
                    can_work = True
                    break

            if can_work:
                start_dt = t
                end_dt   = t + timedelta(minutes=p)
            else:
                start_dt = base_dt + timedelta(minutes=day_work_shifts.pop(0))
                end_dt = start_dt + timedelta(minutes=p)
            
            if total_use_day <= current_machine_time_capacity:
                df.at[idx, 'inicio'] = start_dt.isoformat(timespec='minutes')
                df.at[idx, 'fim']   = end_dt.isoformat(timespec='minutes')
                df.at[idx, 'not_sequence'] = True
                total_use_day += p
            else:
                break
            t = end_dt

    df['inicio'] = pd.to_datetime(df['inicio'], errors='coerce')
    df['fim'] = pd.to_datetime(df['fim'], errors='coerce')
    return df


def build_kp_macho_data(setup_df):
    
    # (kp_macho, resource) -> {"not_setup": [...], "config": str}
    kp_macho_data: Dict[Tuple[str, str], Dict] = {}

    for _, row in setup_df.iterrows():
        
        macho_info = re.sub(r"[A-Za-z]+", "", str(row["_kp_macho"]).replace(" ", ""))

        if "/" in macho_info:
            parts = macho_info.split("/")
            for i, left in enumerate(parts):
                not_setup = [int(p) for j, p in enumerate(parts)] + [i]
                kp_macho_data[(int(left), row["recurso"])] = {"not_setup": not_setup, "config": row["configuracao"]}
        else:
            kp_macho_data[(int(macho_info), row["recurso"])] = {"not_setup": [int(macho_info)], "config": row["configuracao"]}
    return kp_macho_data

# Função responsável por calcular métricas como
# 1. Quantia de jobs atrasados
def calculate_metrics(df:pd.DataFrame, kp_macho_data, setup_times_df):

    metrics = []
    for machine, current_df in df.groupby("maquina", sort=False):
        process, resource = machine.split("_")
        if process in ['pepset']:
            continue

        # cria uma coluna de dia baseada no inicio (ajuste se preferir 'fim' ou 'deadline')
        current_df = current_df.copy()
        current_df["dia"] = current_df["inicio"].dt.date

        for dia, day_df in current_df.groupby("dia", sort=False):

            # atrasados no dia (fim > deadline)
            mask_lateness = (day_df['fim'].dt.date > day_df['deadline'].dt.date)
            atrasados = int(mask_lateness.sum())

            # setup do dia, ordenando por inicio
            sorted_df = day_df.sort_values(by='inicio').reset_index(drop=True)

            total_setup_time = 0
            for idx in range(1, len(sorted_df)):
                # macho anterior (p_) e atual (c_)
                p_macho = int(sorted_df.iloc[idx - 1]['_kf_macho'])
                c_macho = int(sorted_df.iloc[idx]['_kf_macho'])

                p_key = (p_macho, resource)
                c_key = (c_macho, resource)

                p_info = kp_macho_data.get(p_key)
                c_info = kp_macho_data.get(c_key)

                # precisa ter info para ambos
                if not p_info or not c_info:
                    continue

                # se trocar para um macho que NÃO está na lista de "sem setup"
                if c_macho not in p_info.get('not_setup', []):
                    mask_setup = (
                        (setup_times_df['de_config'] == p_info['config']) &
                        (setup_times_df['para_config'] == c_info['config']) &
                        (setup_times_df['recurso'] == resource)
                    )
                    
                    if mask_setup.any():
                        # se houver múltiplas linhas, pega a primeira (ou poderia usar .min())
                        setup_min = setup_times_df.loc[mask_setup, 'setup_time_min'].iloc[0]
                        total_setup_time += int(setup_min)

            total_uso = (day_df['fim'] - day_df['inicio']).dt.total_seconds().sum() // 60
            makespan = (day_df['fim'].max() - day_df['inicio'].min()) / pd.Timedelta(minutes=1)
            metrics.append({
                'machine_name': machine,
                'dia': dia,                
                'setup_time': total_setup_time,
                'atrasados': atrasados,
                'total_uso': total_uso,
                'makespan': makespan
            })

    metrics_df = pd.DataFrame(metrics).sort_values(['machine_name', 'dia']).reset_index(drop=True)
    return metrics_df


def compare_metrics(machines, metrics_real_df: pd.DataFrame, metrics_sched_df: pd.DataFrame, time_capacity_df: pd.DataFrame):
    """
    Compara métricas de setup, atraso e makespan entre 'real' e 'scheduling' por máquina e dia.
    Adiciona % de ganho de tempo em relação à capacidade máxima (max_dia) usando delta de setup.
    Retorna: (per_day_diff, avg_by_machine, overall_avg)
    """

    # 1) Filtra (se 'machines' foi passado)
    if machines:
        subset_real  = metrics_real_df[metrics_real_df['machine_name'].isin(machines)].copy()
        subset_sched = metrics_sched_df[metrics_sched_df['machine_name'].isin(machines)].copy()
    else:
        subset_real  = metrics_real_df.copy()
        subset_sched = metrics_sched_df.copy()

    # 2) Agrega por máquina e dia (setup_time, total_uso, atrasados, makespan)
    real_agg = (
        subset_real
        .groupby(['machine_name', 'dia'], as_index=False, sort=False)
        .agg({'setup_time': 'sum', 'total_uso': 'sum', 'atrasados': 'sum', 'makespan': 'max'})
        .rename(columns={
            'setup_time': 'setup_real',
            'total_uso':  'uso_real',
            'atrasados':  'atrasados_real',
            'makespan':   'makespan_real'
        })
    )

    sched_agg = (
        subset_sched
        .groupby(['machine_name', 'dia'], as_index=False, sort=False)
        .agg({'setup_time': 'sum', 'total_uso': 'sum', 'atrasados': 'sum', 'makespan': 'max'})
        .rename(columns={
            'setup_time': 'setup_sched',
            'total_uso':  'uso_sched',
            'atrasados':  'atrasados_sched',
            'makespan':   'makespan_sched'
        })
    )

    # 3) Outer join por máquina+dia
    per_day_diff = pd.merge(real_agg, sched_agg, on=['machine_name', 'dia'], how='outer')

    # 4) Preenche faltantes e tipa
    for c in [
        'setup_real','setup_sched','uso_real','uso_sched',
        'atrasados_real','atrasados_sched','makespan_real','makespan_sched'
    ]:
        per_day_diff[c] = pd.to_numeric(per_day_diff[c], errors='coerce').fillna(0)
        # atrasos são contagens inteiras; tempos em minutos podem ser inteiros também
        per_day_diff[c] = per_day_diff[c].astype(int)

    # 5) Deltas (positivo = melhor: redução no valor)
    per_day_diff['delta_setup']       = per_day_diff['setup_real']     - per_day_diff['setup_sched']
    per_day_diff['delta_atrasados']   = per_day_diff['atrasados_real'] - per_day_diff['atrasados_sched']
    per_day_diff['delta_makespan']    = per_day_diff['makespan_real']  - per_day_diff['makespan_sched']

    # ---------- MERGE COM CAPACIDADE USANDO O SUFIXO APÓS O ÚLTIMO "_" ----------
    def _norm(s: str) -> str:
        return str(s).lower().strip().replace(' ', '').replace('-', '').replace('/', '')

    per_day_diff['merge_key'] = (
        per_day_diff['machine_name'].astype(str).str.rsplit('_', n=1).str[-1].map(_norm)
    )

    cap_df = time_capacity_df.copy()
    if 'recurso' in cap_df.columns and 'machine_name' not in cap_df.columns:
        cap_df = cap_df.rename(columns={'recurso': 'machine_name'})

    if 'max_dia' in cap_df.columns:
        cap_df['max_dia'] = pd.to_numeric(cap_df['max_dia'], errors='coerce')

    cap_df['merge_key'] = cap_df['machine_name'].map(_norm)

    per_day_diff = per_day_diff.merge(
        cap_df[['merge_key','max_dia']],
        on='merge_key', how='left'
    )

    # 7) Percentual de ganho de tempo (vs capacidade máxima do dia) — baseado no delta de setup
    per_day_diff['time_gain_max'] = (per_day_diff['delta_setup'] / per_day_diff['max_dia']) * 100
    per_day_diff['time_gain_max'] = per_day_diff['time_gain_max'].replace([np.inf, -np.inf], np.nan)

    # 8) Médias por máquina (inclui delta de atrasos e de makespan)
    avg_by_machine = (
        per_day_diff
        .groupby('machine_name', as_index=False)
        .agg(
            avg_daily_delta_setup=('delta_setup','mean'),
            avg_daily_time_gain_max=('time_gain_max','mean'),
            avg_daily_delta_atrasados=('delta_atrasados','mean'),
            avg_daily_delta_makespan=('delta_makespan','mean'),
        )
        .sort_values('machine_name', kind='stable')
    )

    # 9) Média global (minutos de setup)
    overall_avg = float(per_day_diff['delta_setup'].mean()) if not per_day_diff.empty else 0.0

    # organiza
    per_day_diff = (
        per_day_diff
        .drop(columns=['merge_key'])
        .sort_values(['machine_name','dia'])
        .reset_index(drop=True)
    )

    return per_day_diff, avg_by_machine, overall_avg




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--base-data',
        type=pathlib.Path,
        required=True,
        help='O caminho do arquivo de dados base a ser processado.'
    )
    parser.add_argument(
        '--init-date',
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        required=True,
        help='Dia de inicio do processamento.'
    )

    parser.add_argument(
        '--scheduling-solution',
        type=pathlib.Path,
        required=True,
        help='Caminho do arquivo da solucao do scheduling.'
    )
    args = parser.parse_args()

    if not args.base_data or not args.init_date:
        print(f"Adicione os dados da simulação real em --base-data")
        exit()
    
    file_path = pathlib.Path(args.base_data)
    scheduling_solution_path = pathlib.Path(args.scheduling_solution)

    dir_path = file_path.parent 

    init_date = args.init_date
    
    scheduling_solution_df = pd.read_csv(
        scheduling_solution_path,
        parse_dates=['not_before_date', 'deadline', 'inicio', 'fim']
    )
    
    scheduling_solution_df = scheduling_solution_df.dropna(subset=['inicio']).copy()
    


    machine_information_path = dir_path / "brut_machine_information.csv"
    machine_shifts_df = pd.read_csv(machine_information_path, converters={
            'turnos': parse_turnos_to_list,
            'processo': normalize_string,
            'recurso': normalize_string
    })

    turnos_path = dir_path / "brut_shifts.csv"
    turnos_df = pd.read_csv(turnos_path, sep=r"\s*,\s*", engine="python")

    time_capacity_path = dir_path / "brut_recurso_time_capacity.csv"
    time_capacity_df = pd.read_csv(
        time_capacity_path,
        converters={
            'recurso': normalize_string
        }
    )

    demand_df = pd.read_csv(
    file_path,
    parse_dates=['not_before_date', 'deadline'],
    converters={'processo': normalize_string,
                'recurso': normalize_string,
                'tempo_total': lambda x: str(x).strip()}
    )
    # 2) Higienizar e converter a minutos (não numérico -> NaN)
    tempo_raw = demand_df['tempo_total'].str.replace('.', '', regex=False)   # remove milhar
    tempo_raw = tempo_raw.str.replace(',', '.', regex=False)                           # vírgula -> ponto
    demand_df['tempo_total'] = pd.to_numeric(tempo_raw, errors='coerce')

    # 3) Remover linhas inválidas (ex.: '  -   ') ou sem valor
    mask_invalid = demand_df['tempo_total'].isna()
    if mask_invalid.any():
        # opcional: logar quantas foram removidas
        print(f"Removendo {mask_invalid.sum()} linha(s) com 'tempo_total' inválido(s).")
    demand_df = demand_df.loc[~mask_invalid].copy()

    setup_df = pd.read_csv(dir_path / 'setup.csv', converters={'recurso': normalize_string})
    kp_macho_data = build_kp_macho_data(setup_df)
    setup_times_df = pd.read_csv(dir_path / 'setup_times.csv')
    
    demand_df['not_before_date'] = pd.to_datetime(demand_df['not_before_date'], dayfirst=True, errors='coerce')
    
    # Constroi o range de dias
    friday = init_date + timedelta(days=(4 - init_date.weekday()))
    count_days = (friday - init_date).days + 1
    work_days = [init_date + timedelta(days=i) for i in range(count_days)]


    # Verifica os jobs que possuem not before date maior do que a friday, ou seja nao podem ser planejados essa semana
    mask = (demand_df['not_before_date'].dt.date > pd.Timestamp(friday).date())
    demand_df = demand_df.loc[~mask].copy()
    print(f"Quantia de jobs em demand apos a remoacao por not_before_date {demand_df.shape[0]}")

    # Verifica os jobs que possuem deadline < init_date
    mask = (demand_df['deadline'] < pd.Timestamp(init_date))
    demand_df.loc[mask, 'deadline'] = init_date

    demand_df['maquina'] = (demand_df['processo'] + '_' + demand_df['recurso'])
    
    real_sequence_solution = []
    # Agrupando por processo e recurso
    for machine, current_df in demand_df.groupby("maquina", sort=False):

        process, resource = machine.split("_")
        
        if process in ['pepset']:
            continue
        if process != 'coldboxsoprado':
            continue

        current_machine_shift = machine_shifts_df[machine_shifts_df['recurso'] == resource]['turnos']
        current_machine_time_capacity = int(time_capacity_df[time_capacity_df['recurso'] == resource]['max_dia'].iloc[0])
        
        # Ordenação dos jobs em ordem crescente de deadline e caso haja empate por ordenacao
        real_solution_df = current_df.sort_values(by=['deadline', 'ordenacao'], ascending=[True, True])
        
        real_solution_df = calculate_init_end_singleMachine(real_solution_df, work_days, current_machine_shift, turnos_df, current_machine_time_capacity, kp_macho_data, setup_times_df, resource)

        real_sequence_solution.append(real_solution_df)

    path_output = dir_path / "sequence_solution.csv"
    real_sequence_solution_df = pd.concat(real_sequence_solution, ignore_index=True)
    real_sequence_solution_df.to_csv(path_output, index=False)
    #print(real_sequence_solution_df[real_sequence_solution_df['recurso'] == 'sopradora'].shape[0])
    #print(scheduling_solution_df[scheduling_solution_df['maquina'] == 'coldboxsoprado_sopradora'].shape[0])
    metrics_real_solution_df = pd.DataFrame(calculate_metrics(real_sequence_solution_df, kp_macho_data, setup_times_df))
    metrics_scheduling_solution_df = pd.DataFrame(calculate_metrics(scheduling_solution_df, kp_macho_data, setup_times_df))

    
    machines = demand_df['maquina'].unique().tolist()

    
    per_day_diff, avg_by_machine, overall_avg = compare_metrics(
        machines,
        metrics_real_solution_df,
        metrics_scheduling_solution_df,
        time_capacity_df
    )

    per_day_diff.to_csv(dir_path / "metrics.csv")
    print(per_day_diff.head())
    print(avg_by_machine)
    print("Média global do ganho diário de setup:", overall_avg)

    
if __name__ == '__main__':
    main()