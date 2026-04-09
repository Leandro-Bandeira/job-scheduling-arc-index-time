# Copilot Instructions for job-scheduling-arc-index-time

## Visão Geral
Este projeto implementa e compara algoritmos de escalonamento de jobs em máquinas, utilizando modelos matemáticos (Gurobi, Pyomo) e análise de dados (Pandas, Plotly). O código está dividido entre scripts Python para análise/modelagem e código C++ para componentes de performance.

## Estrutura Principal
- `POC/`: Scripts Python para análise, modelagem, geração de gráficos e comparação de soluções.
  - `main.py`: Carrega dados de `POC/instances/job_scheduling_input.json`, executa modelos de otimização, gera gráficos Gantt e exporta resultados.
  - `compare_solutions.py`: Compara soluções de escalonamento, calcula métricas e gera arquivos de saída.
- `src/` e `utils/`: Implementação C++ de estruturas de dados e algoritmos de apoio (ex: `Arcs.hpp`, `Job.hpp`).
- `dados/`: Instâncias de dados reais e sintéticas, organizadas por subpastas.
- `build/`, `bin/`: Saídas de compilação C++.

## Fluxos de Trabalho
- **Execução Python**: Ative o ambiente virtual (`.venv`) e execute scripts em `POC/`.
  - Exemplo: `python3 POC/compare_solutions.py --base-data dados/29092025/parameters/brut_demand_29.csv --init-date "2025-09-29" --scheduling-solution dados/29092025/parameters/job_scheduling_output_2025-09-29.csv`
- **Compilação C++**: Use o `makefile` na raiz do projeto para compilar (`make all`) e limpar (`make clean`).
- **Dados de Entrada**: Os scripts Python esperam arquivos `.csv` em subpastas de `dados/` e arquivos `.json` em `POC/instances/`.
- **Saídas**: Resultados e gráficos são salvos em `POC/results/` ou na pasta de instância correspondente.

## Convenções e Padrões
- Funções utilitárias para normalização de strings e parsing de turnos estão em `compare_solutions.py`.
- Nomes de máquinas e recursos são normalizados para minúsculas, sem acentos e sem espaços.
- O modelo matemático utiliza Gurobi/Pyomo, com parâmetros lidos de JSON.
- Gráficos Gantt interativos são gerados via Plotly e salvos como HTML.
- O código C++ utiliza headers em `utils/` e implementações em `src/`.

## Integrações e Dependências
- Python: `gurobipy`, `pyomo`, `pandas`, `plotly`, `matplotlib`, `unidecode`.
- C++: Compilador g++ com suporte a C++17.
- Ambiente virtual Python deve ser ativado antes de rodar scripts.

## Exemplos de Uso
- Rodar análise de solução: ver README.md para exemplos de comandos.
- Adicionar nova instância: coloque arquivos `.csv` em uma nova subpasta de `dados/` e ajuste caminhos nos scripts.

## Dicas para Agentes AI
- Sempre normalize nomes de recursos/máquinas ao comparar ou manipular dados.
- Consulte `main.py` e `compare_solutions.py` para fluxos de dados e padrões de entrada/saída.
- Use o makefile para compilar C++ e scripts Python para análise/modelagem.
- Novos scripts devem seguir o padrão de leitura de dados e exportação de resultados já presente em `POC/`.
