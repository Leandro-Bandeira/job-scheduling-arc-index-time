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
                not_setup = [p for j, p in enumerate(parts) if j != i]
                kp_macho_data[(left, row["recurso"])] = {"not_setup": not_setup, "config": row["configuracao"]}
        else:
            kp_macho_data[(macho_info, row["recurso"])] = {"not_setup": [], "config": row["configuracao"]}

    return kp_macho_data

# Função responsável por calcular métricas como
# 1. Quantia de jobs atrasados
def calculate_metrics(df:pd.DataFrame, kp_macho_data, setup_times_df):

    metrics_df = {}
    # Calculando quantia de jobs em atraso
    mask_lateness = (df['fim'].dt.date > df['deadline'].dt.date)
    df_lateness = df.loc[mask_lateness].copy()


    total_setup_time = 0
    # Calculando tempo de setup total
    for (process, resource), current_df in df.groupby(["processo", "recurso"], sort=False):
        if resource not in ['vibrado', 'sopradora']:
            continue

        sorted_df = current_df.sort_values(by=['inicio']).reset_index(drop=True)
        
        for idx in range(1, len(sorted_df)):
            if not idx:
                continue
            p_macho = str(int(sorted_df.iloc[idx - 1]['_kf_macho']))
            c_macho = str(int(sorted_df.iloc[idx]['_kf_macho']))
            
            p_key = (p_macho, resource)
            c_key = (c_macho, resource)

            # Colentando informacao sobre o macho anterior
            p_info = kp_macho_data.get(p_key)
            c_info = kp_macho_data.get(c_key)
            
            if p_info:
                if c_macho not in p_info['not_setup']:

                    mask_setup = (
                        (setup_times_df['de_config'] == p_info['config']) &
                        (setup_times_df['para_config'] == c_info['config']) &
                        (setup_times_df['recurso'] == resource)
                    )

                    if mask_setup.any():
                        # se houver múltiplas linhas, pega a primeira
                        setup_min = setup_times_df.loc[mask_setup, 'setup_time_min'].iloc[0]
                        # garante inteiro
                        setup_min = int(setup_min)
                        total_setup_time += setup_min



    metrics_df['atrasados'] = df_lateness.shape[0]
    metrics_df['total_setup_time'] = total_setup_time

    return metrics_df

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

    args = parser.parse_args()
    if not args.base_data or not args.init_date:
        print(f"Adicione os dados da simulação real em --base-data")
        exit()
    
    file_path = pathlib.Path(args.base_data)
    dir_path = file_path.parent 

    init_date = args.init_date
    
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

    real_sequence_solution = []
    # Agrupando por processo e recurso
    for (process, resource), current_df in demand_df.groupby(["processo", "recurso"], sort=False):
        
        if process== 'pepset': continue
        
        # Ordenação dos jobs em ordem crescente de deadline e caso haja empate por ordenacao
        real_solution_df = current_df.sort_values(by=['deadline', 'ordenacao'], ascending=[True, True])
        
        real_solution_df = calculate_init_end_singleMachine(real_solution_df, work_days)

        real_sequence_solution.append(real_solution_df)

    path_output = dir_path / "sequence_solution.csv"
    real_sequence_solution_df = pd.concat(real_sequence_solution, ignore_index=True)
    real_sequence_solution_df.to_csv(path_output, index=False)


    metrics_df = calculate_metrics(real_sequence_solution_df, kp_macho_data, setup_times_df)
    print(metrics_df)

if __name__ == '__main__':
    main()