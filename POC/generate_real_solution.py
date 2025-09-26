import pandas as pd

import argparse
import pathlib

from datetime import datetime, timedelta


# Função responsavel por calcular o inicio e fim de um dataframe
# 1. Devemos agrupar por mesma deadline
# 2. Dentro do grupo que os jobs possuem a mesma deadline, agruparemos pela coluna ordenacao
# 3. Caso o not before date job nao seja maior que init_dt, devemos coloca-lo no proximo grupo

def calculate_init_end_singleMachine(df:pd.DataFrame, work_days:list) -> pd.DataFrame:    

    df['not_sequence'] = False # Define que nenhum job foi sequenciado
    

    # Agrupa por deadline
    
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
            fim  = current_dt + pd.to_timedelta(row['Tempo Total (minutos)'], unit='m')
            df.at[idx, 'fim'] = fim
            df.at[idx, 'not_sequence'] = True
            current_dt = fim

    """     
    for idx, row in df.iterrows():
        job_not_before_date = row['not_before_date']

        if pd.Timestamp(init_dt).date() >= pd.Timestamp(job_not_before_date).date():
            df.at[idx, 'inicio'] = init_dt
            fim  = init_dt + pd.to_timedelta(row['Tempo Total (minutos)'], unit='m')
            df.at[idx, 'fim'] = fim
            init_dt = fim
    """
    return df


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
    
    demand_df = pd.read_csv(file_path, parse_dates=['not_before_date', 'deadline'])
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

    violations_per_machine = []
    # Agrupando por processo e recurso
    for (process, resource), current_df in demand_df.groupby(["processo", "recurso"], sort=False):
        
        # Ordenação dos jobs em ordem crescente de deadline e caso haja empate por ordenacao
        current_df = current_df.sort_values(by=['deadline', 'ordenacao'], ascending=[True, True])
        
        current_df = calculate_init_end_singleMachine(current_df, work_days)
        

        current_df['deadline_violation'] = current_df['fim'].dt.date > current_df['deadline'].dt.date
        violations_per_machine.append(current_df)

    path_output = dir_path / "violations.csv"
    violations_df = pd.concat(violations_per_machine, ignore_index=True)
    violations_df.to_csv(path_output, index=False)



if __name__ == '__main__':
    main()