# Projeto de Otimização de Produção

Este projeto realiza o processamento de dados, otimização de modelos e geração de relatórios e gráficos para análise de produção. Ele utiliza Python e ferramentas como `poetry` para gerenciamento de dependências e `Streamlit` para visualização interativa.

## Configuração do Ambiente Local

### Criar o Ambiente Virtual

Para configurar o ambiente virtual, execute os seguintes comandos:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -U pip setuptools
.venv/bin/pip install poetry
```

## Installing dependencies

Após configurar o ambiente virtual, instale as dependências do projeto utilizando o poetry:

```bash
poetry install
```

## Requisitos

1. Conta no AWS
2. Instalação do AWS CLI
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

unzip awscliv2.zip

sudo ./aws/install

aws --version
```
3. Para baixar o input para o dia corrente utilize o seguinte comando:
```bash
python src/main/athena_ingest.py --dt 2025-10-12
```

## Execução dos Scripts

1. Processamento de Dados de Entrada
   O script data_input_process.py processa os dados de entrada e gera os arquivos necessários para o modelo.

```bash
python3 src/main/data_input_process.py --dt 2025-10-12 --day-start "2025-10-13 00:00" --time-step 5
```

2. Otimização do Modelo
   O script optimize.py executa a otimização do modelo com base nos dados processados.

```bash
python3 src/main/optimize.py --dt 2025-10-12

```

3. Processamento dos Resultados
   O script data_output_process.py processa os resultados da otimização e gera gráficos e relatórios

```bash

python3 src/main/data_output_process.py --dt 2025-10-12
```

4. Visualização Interativa com Streamlit
   Para visualizar os gráficos e relatórios gerados, utilize o Streamlit. Execute o seguinte comando:

```bash
streamlit run src/main/dashboard.py
```

5. Você pode também rodar apenas

```bash
   chmod a+x run.sh
   ./run.sh --dt 2025-10-12 --day-start "2025-10-13 00:00"
```

## Estrutura do Projeto

- **data/raw**: Contém os dados brutos de entrada.
- **data/trusted**: Contém os dados processados e confiáveis.
- **src/main**: Contém os scripts principais do projeto:
  - **data_input_process.py**: Processa os dados de entrada.
  - **optimize.py**: Realiza a otimização do modelo.
  - **data_output_process.py**: Processa os resultados e gera relatórios.
  - **dashboard.py**: Interface interativa para visualização dos resultados.

## Requisitos

- **Python 3.11**
- **Poetry**: Para gerenciamento de dependências.
- **Streamlit**: Para visualização interativa.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## Integração com o S3

Este projeto está preparado para rodar com Amazon S3 para gerenciar os dados de entrada e saída do pipeline de produção. Abaixo está uma explicação detalhada das pastas utilizadas e como os dados fluem entre elas.

### Estrutura das Pastas no S3

1. **Pasta de Entrada**:

   - **Caminho**: `s3://harumi-production-data-pipeline/afm/input/demanda/dt={date}`
   - **Descrição**: Esta pasta é onde os arquivos de entrada são escritos pelo AWS Glue. O `{date}` no caminho é um marcador de data que indica o dia específico para o qual os dados foram gerados.
   - **Conteúdo**: Os arquivos nesta pasta contêm os dados necessários para o otimizador, como informações de demanda, máquinas, jobs e configurações.
2. **Pasta de Saída**:

   - **Caminho**: `s3://harumi-production-data-pipeline/afm/output/demanda/`
   - **Descrição**: Esta pasta é onde os arquivos de saída gerados pelo otimizador devem ser gravados. O AWS Glue lê esses arquivos e os salva na base de dados para consumo posterior.
   - **Conteúdo**: Os arquivos nesta pasta contêm os resultados do otimizador, como o planejamento de produção, horários de jobs e uso de máquinas.

### Fluxo de Dados

1. **Entrada**:

   - O AWS Glue escreve os arquivos de entrada na pasta `s3://harumi-production-data-pipeline/afm/input/demanda/dt={date}`.
   - O otimizador lê esses arquivos para processar os dados e gerar o planejamento.
2. **Processamento**:

   - O otimizador processa os dados de entrada e executa os cálculos necessários para gerar o planejamento de produção.
3. **Saída**:

   - Os resultados do otimizador são gravados na pasta `s3://harumi-production-data-pipeline/afm/output/demanda/`.
   - O AWS Glue lê esses arquivos de saída e os salva na base de dados para consumo por outras partes do sistema.

### Exemplo de Caminhos no S3

- **Entrada**:
  s3://harumi-production-data-pipeline/afm/input/demanda/dt=2023-10-01/

Arquivos esperados:

- `part-00000-2e40ab82-e87f-4631-b9b3-f750faa3cd0c-c000.snappy`
- **Saída**:
  s3://harumi-production-data-pipeline/afm/output/demanda/

Arquivos gerados:

- `job_scheduling_output.parquet`

### Observações Importantes

- Certifique-se de que os arquivos de entrada estejam no formato esperado pelo otimizador.
- Após o processamento, os arquivos de saída devem ser gravados no formato Parquet para garantir compatibilidade com o AWS Glue.
- O marcador `{date}` na pasta de entrada deve ser substituído pela data correspondente ao processamento.
