import pandas as pd

import argparse
import pathlib

from datetime import datetime


# Função responsavel por calcular o inicio e fim de um dataframe
def calculate_init_end_singleMachine(df:pd.DataFrame, init_dt:datetime) -> pd.DataFrame:    
    
    for idx, row in df.iterrows():
        df.at[idx, 'inicio'] = init_dt
        fim  = init_dt + pd.to_timedelta(row['Tempo Total (minutos)'], unit='m')
        df.at[idx, 'fim'] = fim
        init_dt = fim

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
    mask = demand_df['deadline'] < init_date
    demand_df.loc[mask, 'deadline'] = init_date
    
    violations_per_machine = []
    # Agrupando por processo e recurso
    for (process, resource), current_df in demand_df.groupby(["processo", "recurso"], sort=False):
        init_dt = init_date
        

        
        
        # Ordenação dos jobs em ordem crescente
        current_df = current_df.sort_values(by="ordenacao", ascending=True)
        current_df = calculate_init_end_singleMachine(current_df, init_dt)
        

        current_df['deadline_violation'] = current_df['fim'].dt.date > current_df['deadline'].dt.date
        violations_per_machine.append(current_df)

        print(f"Analisando a máquina: {process}_{resource} com dia de início {init_dt}")
        
    path_output = dir_path / "violations.csv"
    violations_df = pd.concat(violations_per_machine, ignore_index=True)
    violations_df.to_csv(path_output, index=False)



if __name__ == '__main__':
    main()