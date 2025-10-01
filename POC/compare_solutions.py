import argparse
import re
import pathlib


import pandas as pd

from unidecode import unidecode

from datetime import datetime, timedelta
from typing import Dict, Tuple


def normalize_string(value) -> str:
    """Normaliza string: remove acentos, minúsculas, sem espaços."""
    if pd.isna(value):
        return ""
    return unidecode(str(value)).lower().replace(" ", "")

# Função responsavel por calcular o inicio e fim de um dataframe
# 1. Df é um dataframe ordenado em prioridade de deadline e ordenação
# 2. work_days é o conjunto de dias possiveis de trabalho dado o dia inicial ate a sexta-feira da mesma semana
# 3. Para cada linha do dataframe, caso seu not_before_date seja menor ou igual ao dia no sequenciamento, calcula-se o inicio e o fim
# 4. Se no fim existirem linhas que não foram calculadas inicio e fim, é porque não é possível produzir o job naquela semana

def calculate_init_end_singleMachine(df:pd.DataFrame, work_days:list) -> pd.DataFrame:    

    df['not_sequence'] = False # Define que nenhum job foi sequenciado
    
    # Vamos organizar do init_dt ate a sexta feira da mesma semana
    # Cria uma mascara dos jobs que estao sequenciados em df que possuem not before date >= init_date
    # Cria o sequenciamento para esses jobs e adiciona uma nova coluna chamada sequenciado
    # Mesma coisa para cada dia
    for day in work_days:
        current_dt = day
        day_date = pd.Timestamp(day).date()
        mask = (df['not_before_date'].dt.date <= day_date) & (~df['not_sequence'])
        current_df = df.loc[mask]

        for idx, row in current_df.iterrows():
            df.at[idx, 'inicio'] = current_dt
            if row['Tempo Total (minutos)'] == '-':
                row['Tempo Total (minutos)'] = 1
            fim  = current_dt + pd.to_timedelta(int(row['Tempo Total (minutos)']), unit='m')
            df.at[idx, 'fim'] = fim
            df.at[idx, 'not_sequence'] = True
            current_dt = fim
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
        if resource not in ['sopradora']:
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
                    #print(f"from: {p_key} to {c_key}")
                    #print(p_info.get('not_setup'))

                    if mask_setup.any():
                        # se houver múltiplas linhas, pega a primeira (ou poderia usar .min())
                        setup_min = setup_times_df.loc[mask_setup, 'setup_time_min'].iloc[0]
                        total_setup_time += int(setup_min)

            total_uso = (day_df['fim'] - day_df['inicio']).dt.total_seconds().sum() // 60
            metrics.append({
                'machine_name': machine,
                'dia': dia,                
                'setup_time': total_setup_time,
                'atrasados': atrasados,
                'total_uso': total_uso
            })

    metrics_df = pd.DataFrame(metrics).sort_values(['machine_name', 'dia']).reset_index(drop=True)
    return metrics_df


def compare_metrics(machines, metrics_real_df: pd.DataFrame, metrics_sched_df: pd.DataFrame):
    """
    Compara métricas de setup entre 'real' e 'scheduling' por máquina e dia.
    
    Delta de setup (ganho por dia) = setup_real - setup_sched.
    Valores negativos indicam que o scheduling gastou MENOS setup (ou seja, melhor).
    
    Parâmetros
    ----------
    machines : list[str]
        Lista de máquinas a considerar (filtra as duas tabelas).
    metrics_real_df : pd.DataFrame
        DataFrame com colunas ['machine_name', 'dia', 'setup_time', ...].
    metrics_sched_df : pd.DataFrame
        DataFrame com as mesmas colunas acima.
    
    Retorna
    -------
    per_day_diff : pd.DataFrame
        Linhas por (machine_name, dia) com colunas:
        ['machine_name', 'dia', 'setup_real', 'setup_sched', 'delta_setup'].
    avg_by_machine : pd.DataFrame
        Linhas por machine_name com a média do delta por dia:
        ['machine_name', 'avg_daily_delta_setup'].
    overall_avg : float
        Média global do delta por dia considerando todas as máquinas/dias.
    """

    # 1) Filtra pelas máquinas pedidas (se 'machines' estiver vazia, não filtra)
    subset_real = metrics_real_df.copy()
    subset_sched = metrics_sched_df.copy()

    subset_real = subset_real[subset_real['machine_name'].isin(machines)]
    subset_sched = subset_sched[subset_sched['machine_name'].isin(machines)]

    # 2) Agrega por máquina e dia (caso existam múltiplas linhas)
    real_agg = (
    subset_real
    .groupby(['machine_name', 'dia'], as_index=False, sort=False)
    .agg({'setup_time': 'sum', 'total_uso': 'sum'})
    .rename(columns={'setup_time': 'setup_real', 'total_uso': 'uso_real'})
    )

    sched_agg = (
        subset_sched
        .groupby(['machine_name', 'dia'], as_index=False, sort=False)
        .agg({'setup_time': 'sum', 'total_uso': 'sum'})
        .rename(columns={'setup_time': 'setup_sched', 'total_uso': 'uso_sched'})
    )

    # 3) Junta as duas visões por máquina+dia (outer para não perder dias exclusivos de um lado)
    per_day_diff = pd.merge(
        real_agg, sched_agg,
        on=['machine_name', 'dia'],
        how='outer'
    )

    # Preenche faltantes como 0 (se um lado não teve produção naquele dia)
    per_day_diff['setup_real'] = per_day_diff['setup_real'].fillna(0).astype(int)
    per_day_diff['setup_sched'] = per_day_diff['setup_sched'].fillna(0).astype(int)
    per_day_diff['uso_real'] = per_day_diff['uso_real'].fillna(0).astype(int)
    per_day_diff['uso_sched'] = per_day_diff['uso_sched'].fillna(0).astype(int)

    # 5) Delta diário (ganho): real - scheduling
    per_day_diff['delta_setup'] = per_day_diff['setup_real'] - per_day_diff['setup_sched']

    
    # 6) Média do delta por dia para cada máquina
    avg_by_machine = (
        per_day_diff
        .groupby('machine_name', as_index=False)['delta_setup']
        .mean()
        .rename(columns={'delta_setup': 'avg_daily_delta_setup'})
        .sort_values('machine_name', kind='stable')
    )

    # 7) Média global
    overall_avg = float(per_day_diff['delta_setup'].mean()) if not per_day_diff.empty else 0.0

    return per_day_diff.sort_values(['machine_name', 'dia']).reset_index(drop=True), avg_by_machine, overall_avg


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
    
    demand_df = pd.read_csv(
    file_path,
    parse_dates=['not_before_date', 'deadline'],
    converters={'processo': normalize_string,
                'recurso': normalize_string,
                'Tempo Total (minutos)': lambda x: str(x).strip()}
    )
    # 2) Higienizar e converter a minutos (não numérico -> NaN)
    tempo_raw = demand_df['Tempo Total (minutos)'].str.replace('.', '', regex=False)   # remove milhar
    tempo_raw = tempo_raw.str.replace(',', '.', regex=False)                           # vírgula -> ponto
    demand_df['Tempo Total (minutos)'] = pd.to_numeric(tempo_raw, errors='coerce')

    # 3) Remover linhas inválidas (ex.: '  -   ') ou sem valor
    mask_invalid = demand_df['Tempo Total (minutos)'].isna()
    if mask_invalid.any():
        # opcional: logar quantas foram removidas
        print(f"Removendo {mask_invalid.sum()} linha(s) com 'Tempo Total (minutos)' inválido(s).")
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
    
    # Verifica os jobs que possuem deadline < init_date
    mask = (demand_df['deadline'] < pd.Timestamp(init_date))
    demand_df.loc[mask, 'deadline'] = init_date

    demand_df['maquina'] = (demand_df['processo'] + '_' + demand_df['recurso'])

    real_sequence_solution = []
    # Agrupando por processo e recurso
    for machine, current_df in demand_df.groupby("maquina", sort=False):

        process, resource = machine.split("_")
        
        if resource not in ['sopradora']:
            continue
        # Ordenação dos jobs em ordem crescente de deadline e caso haja empate por ordenacao
        real_solution_df = current_df.sort_values(by=['deadline', 'ordenacao'], ascending=[True, True])
        
        real_solution_df = calculate_init_end_singleMachine(real_solution_df, work_days)

        real_sequence_solution.append(real_solution_df)

    path_output = dir_path / "sequence_solution.csv"
    real_sequence_solution_df = pd.concat(real_sequence_solution, ignore_index=True)
    real_sequence_solution_df.to_csv(path_output, index=False)

    
    metrics_real_solution_df = pd.DataFrame(calculate_metrics(real_sequence_solution_df, kp_macho_data, setup_times_df))
    metrics_scheduling_solution_df = pd.DataFrame(calculate_metrics(scheduling_solution_df, kp_macho_data, setup_times_df))
    
    
    machines = demand_df['maquina'].unique().tolist()

    
    per_day_diff, avg_by_machine, overall_avg = compare_metrics(
        machines,
        metrics_real_solution_df,
        metrics_scheduling_solution_df
    )

    print(per_day_diff.head())
    print(avg_by_machine)
    print("Média global do ganho diário de setup:", overall_avg)

    
if __name__ == '__main__':
    main()