import os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import logging
from datetime import datetime, timedelta
import plotly.express as px
from pathlib import Path
from collections import defaultdict
import json
from harumi_api import request_otp, verify_otp, list_organizations
import argparse 

logging.basicConfig(level=logging.DEBUG)


AFM_ORG_ID = "e550054e-155d-4a6d-ba67-33ab9b510513"

# Formulário de login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.request_otp = False
    st.session_state.verify_otp = False

if not st.session_state.authenticated:
    st.title("🔒 Login")

    # Formulário para solicitar o OTP
    if not st.session_state.request_otp:
        with st.form("request_otp_form"):
            if st.session_state.get("invalid_org", False):
                st.error("Erro ao validar organização! Acesso negado.")

            email = st.text_input("E-mail")
            submit_request = st.form_submit_button("Solicitar OTP")

            if submit_request:
                if request_otp(email):
                    st.session_state.email = email  # Armazena o e-mail na sessão
                    st.session_state.request_otp = True
                    st.rerun()
                else:
                    st.error("Erro ao solicitar o OTP: Verifique o e-mail fornecido.")
                    
    # Formulário para validar o OTP
    if st.session_state.request_otp and not st.session_state.verify_otp:
        with st.form("verify_otp_form"):
            st.success("OTP enviado com sucesso! Verifique seu e-mail.")
            token = st.text_input("Digite o token recebido no e-mail")
            submit_verify = st.form_submit_button("Validar OTP")

            if submit_verify:
                verified, logged_user = verify_otp(st.session_state.email, token)
                if verified and logged_user:
                    st.session_state.verify_otp = True
                    st.session_state.logged_user = logged_user
                    st.success("Login bem-sucedido!")
                    st.rerun()
                else:
                    st.error("Erro ao validar o token: Verifique o token fornecido.")
    
    if st.session_state.verify_otp:
        success, organizations = list_organizations(st.session_state.logged_user)
        if success and organizations and any(org.get("id") == AFM_ORG_ID for org in organizations):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.session_state.request_otp = False
            st.session_state.verify_otp = False
            st.session_state.authenticated = False
            st.session_state.invalid_org = True
            
            st.rerun()

    st.stop()
else:
    # Adicionar botão de logout
    st.sidebar.title("Configurações")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.request_otp = False
        st.session_state.verify_otp = False
        st.success("Você foi desconectado.")
        st.rerun()



base_data = Path(__file__).parent.parent.parent / 'data/trusted/21102025/44'


# Configuração de tela cheia
st.set_page_config(
    # page_title="Painel – Gantt Trabalho",
    layout="wide",  # Define o layout como "wide" (tela cheia)
    initial_sidebar_state="collapsed",  # Oculta a barra lateral inicialmente
)


# Função para criar gráficos de Gantt por dia
def create_gantt_by_day(df):
    gantt_by_day = {}
    for day, day_df in df.groupby(df['inicio'].dt.date):
        # só plota se houver job no dia
        if not ((day_df['tipo'] == 'job')).any():
            continue

        # ===== CORES FIXAS POR MÁQUINA (estáveis ao longo do dataset) =====
        palette = (
            "#3B6BA5",  # steel blue
            "#2A7F7F",  # teal fechado
            "#6A7D2C",  # oliva
            "#5B4E77",  # roxo ardósia
            "#726C15",  # slate
            "#8C5A2B",  # castanho/rust
            "#2E6B3D",  # verde floresta
        )
        # usa o df completo para manter a cor estável entre dias
        machines = sorted(df['maquina'].dropna().unique().tolist())
        machine_color_map = {m: palette[i % len(palette)] for i, m in enumerate(machines)}
        # overrides de categorias especiais
        machine_color_map.update({
            'SETUP': 'black',
            'INDISPONIVEL': '#708090',
            'JOB_ATRASADO': '#d63031',
        })
        color_map = machine_color_map

        # ===== GANTT =====
        fig = px.timeline(
            day_df,
            x_start="inicio",
            x_end="fim",
            y="machine_y",
            color="color_tag",              # usa sua coluna
            custom_data=["hover_label"],
            title=f"Gantt – {day.strftime('%d/%m/%Y')}",
            color_discrete_map=color_map   # aplica as cores fixas
        )
        fig.update_traces(hovertemplate="%{customdata[0]}")

        # ===== ORDEM DO EIXO Y: agrupa por máquina e ordena pelo índice da banda =====
        yinfo = (day_df[['maquina', 'machine_y', 'machine_label']]
                 .drop_duplicates()
                 .copy())
        # extrai o sufixo inteiro de 'machine_y' (ex.: maquina_0, maquina_1, …)
        yinfo['band_idx'] = (
            yinfo['machine_y']
            .str.extract(r'_(\d+)$', expand=False)
            .fillna('0').astype(int)
        )
        yinfo = yinfo.sort_values(['maquina', 'band_idx'])

        y_order = yinfo['machine_y'].tolist()
        y_texts = yinfo['machine_label'].tolist()

        fig.update_yaxes(
            autorange="reversed",
            categoryorder='array',
            categoryarray=y_order,   # força a ordem
            tickmode='array',
            tickvals=y_order,
            ticktext=y_texts
        )

        # ===== EIXO X / LAYOUT =====
        start_day = datetime.combine(day, datetime.min.time())
        end_day = start_day + timedelta(days=1)
        fig.update_layout(
            height=max(600, 60 * len(y_order)),
            xaxis_title="Hora do Dia",
            yaxis_title="Máquina",
            hovermode="closest",
            showlegend=False,  # mude para True se quiser legenda
            xaxis=dict(
                range=[start_day, end_day],
                tickformat="%H:%M",
                dtick=3600000,
                ticklabelmode="period",
                tickangle=0
            )
        )

        # Adiciona separadores de horas no gráfico
        for hour in range(0, 24):
            hour_marker = start_day + timedelta(hours=hour)
            fig.add_vline(
                x=hour_marker,
                line_width=0.5,
                line_dash="dot",
                line_color="gray"
            )
        gantt_by_day[day] = fig
    return gantt_by_day


# ------------------------------
# Painel
# ------------------------------
st.title("Painel – Gantt trabalho")



# Código responsável por criar o gantt semanal
def create_week_gantt(df):
    # Paleta fixa p/ máquinas (sem vermelho); vermelho só para JOB_ATRASADO
    machines = sorted(df.loc[df['tipo'] == 'job', 'maquina'].dropna().unique().tolist())
    palette = [
        "#3B6BA5", "#2A7F7F", "#6A7D2C", "#5B4E77",
        "#726C15", "#8C5A2B", "#2E6B3D", "#1E88E5",
        "#00838F", "#6D4C41", "#7CB342", "#8E24AA",
        "#3949AB", "#00ACC1", "#5E548E"
    ]
    machine_color_map = {m: palette[i % len(palette)] for i, m in enumerate(machines)}
    machine_color_map.update({
        'SETUP': 'black',
        'INDISPONIVEL': '#708090',
        'JOB_ATRASADO': '#d63031'  # só atraso fica vermelho
    })

    fig = px.timeline(
        df,
        x_start="inicio",
        x_end="fim",
        y="machine_y",
        color="color_tag",
        custom_data=["hover_label"],
        title="Gantt Interativo – Visão por Semana",
        color_discrete_map=machine_color_map
    )
    fig.update_traces(hovertemplate="%{customdata[0]}")

    # ======= Força a ordem do eixo Y por (maquina, índice da banda) =======
    yinfo = (df[['maquina', 'machine_y', 'machine_label']]
             .drop_duplicates()
             .copy())
    # pega o sufixo numérico de machine_y; se não houver, usa 0
    yinfo['band_idx'] = (
        yinfo['machine_y']
        .str.extract(r'_(\d+)$', expand=False)
        .fillna('0').astype(int)
    )
    yinfo = yinfo.sort_values(['maquina', 'band_idx'])

    y_order = yinfo['machine_y'].tolist()
    y_texts = yinfo['machine_label'].tolist()

    fig.update_yaxes(
        autorange="reversed",
        categoryorder='array',
        categoryarray=y_order,    # ordem fixa
        tickmode='array',
        tickvals=y_order,
        ticktext=y_texts
    )

    return fig


# Função para criar gráficos de Gantt por dia
def create_gantt_by_day(df):
    gantt_by_day = {}
    for day, day_df in df.groupby(df['inicio'].dt.date):
        # só plota se houver job no dia
        if not ((day_df['tipo'] == 'job')).any():
            continue

        # ===== CORES FIXAS POR MÁQUINA (estáveis ao longo do dataset) =====
        palette = (
            "#3B6BA5",  # steel blue
            "#2A7F7F",  # teal fechado
            "#6A7D2C",  # oliva
            "#5B4E77",  # roxo ardósia
            "#726C15",  # slate
            "#8C5A2B",  # castanho/rust
            "#2E6B3D",  # verde floresta
        )
        # usa o df completo para manter a cor estável entre dias
        machines = sorted(df['maquina'].dropna().unique().tolist())
        machine_color_map = {m: palette[i % len(palette)] for i, m in enumerate(machines)}
        # overrides de categorias especiais
        machine_color_map.update({
            'SETUP': 'black',
            'INDISPONIVEL': '#708090',
            'JOB_ATRASADO': '#d63031',
        })
        color_map = machine_color_map

        # ===== GANTT =====
        fig = px.timeline(
            day_df,
            x_start="inicio",
            x_end="fim",
            y="machine_y",
            color="color_tag",              # usa sua coluna
            custom_data=["hover_label"],
            title=f"Gantt – {day.strftime('%d/%m/%Y')}",
            color_discrete_map=color_map   # aplica as cores fixas
        )
        fig.update_traces(hovertemplate="%{customdata[0]}")

        # ===== ORDEM DO EIXO Y: agrupa por máquina e ordena pelo índice da banda =====
        yinfo = (day_df[['maquina', 'machine_y', 'machine_label']]
                 .drop_duplicates()
                 .copy())
        # extrai o sufixo inteiro de 'machine_y' (ex.: maquina_0, maquina_1, …)
        yinfo['band_idx'] = (
            yinfo['machine_y']
            .str.extract(r'_(\d+)$', expand=False)
            .fillna('0').astype(int)
        )
        yinfo = yinfo.sort_values(['maquina', 'band_idx'])

        y_order = yinfo['machine_y'].tolist()
        y_texts = yinfo['machine_label'].tolist()

        fig.update_yaxes(
            autorange="reversed",
            categoryorder='array',
            categoryarray=y_order,   # força a ordem
            tickmode='array',
            tickvals=y_order,
            ticktext=y_texts
        )

        # ===== EIXO X / LAYOUT =====
        start_day = datetime.combine(day, datetime.min.time())
        end_day = start_day + timedelta(days=1)
        fig.update_layout(
            height=max(600, 60 * len(y_order)),
            xaxis_title="Hora do Dia",
            yaxis_title="Máquina",
            hovermode="closest",
            showlegend=False,  # mude para True se quiser legenda
            xaxis=dict(
                range=[start_day, end_day],
                tickformat="%H:%M",
                dtick=3600000,
                ticklabelmode="period",
                tickangle=0
            )
        )

        # Adiciona separadores de horas no gráfico
        for hour in range(0, 24):
            hour_marker = start_day + timedelta(hours=hour)
            fig.add_vline(
                x=hour_marker,
                line_width=0.5,
                line_dash="dot",
                line_color="gray"
            )
        gantt_by_day[day] = fig
    return gantt_by_day

###### Função responsável por criar o hover label de cada tipo ########
def create_hover_label(row):
        if row['tipo'] == 'setup':
            return (
                f"SETUP<br>"
                f"Início: {row['inicio'].strftime('%H:%M')}<br>"
                f"Fim: {row['fim'].strftime('%H:%M')}<br>"
                f"Tempo: {(row['fim'] - row['inicio']).total_seconds() / 60:.0f} min"
            )
        elif row['tipo'] == 'indisponivel':
            return (
                f"Fora de turno<br>"
                f"Início: {row['inicio'].strftime('%H:%M')}<br>"
                f"Fim: {row['fim'].strftime('%H:%M')}<br>")
        else:
            return (
                f"Máquina: {row['maquina']}<br>"
                f"Op: {int(row.get('op', ''))}<br>"
                f"Caixa: {row.get('caixa', '')}<br>"
                f"Início: {row['inicio'].strftime('%H:%M')}<br>"
                f"Fim: {row['fim'].strftime('%H:%M')}<br>"
                f"Config: {row.get('config', '')}<br>"
                f"Kp_fichaTenica: {row.get('kp', '')}<br>"
                f"_kf_macho: {row.get('_kf_macho', '')}<br>"
            )

def process_setups(
    gantt_df_with_band,
    setups,
    time_step
):
    """
    Cria barras de SETUP apenas quando A e B estão 'próximos':
      gap_minutos entre fim(A) e início(B) ∈ [setup_minutos, setup_minutos + slack_slots*time_step]
    e (opcionalmente) no mesmo dia.
    Se absolute_max_gap_min for definido, também exige gap_minutos <= absolute_max_gap_min.
    Agrupa por 'machine_y' (banda). Cai para 'maquina' se não existir.
    """
    # índice rápido (from_job_id, to_job_id) -> setup_time (em slots)
    setup_idx = {
        (int(s['from_job_id']), int(s['to_job_id'])): int(s['setup_time'])
        for s in setups
    }

    rows = []
    jobs = gantt_df_with_band[gantt_df_with_band['tipo'] == 'job'].copy()
    key_band = 'machine_y' if 'machine_y' in jobs.columns else 'maquina'

    for band_id, group in jobs.groupby(key_band):
        g = group.sort_values('inicio').reset_index(drop=True)

        for i in range(len(g) - 1):
            a = g.loc[i]
            b = g.loc[i + 1]

            from_job = int(a['job_id'])
            to_job   = int(b['job_id'])

            setup_slots = setup_idx.get((from_job, to_job))
            if not setup_slots or setup_slots <= 0:
                continue

            setup_min = setup_slots * time_step
            gap_min = (b['inicio'] - a['fim']).total_seconds() / 60.0

            # proximidade: gap >= setup e gap <= setup + folga
            if gap_min < setup_min:
                continue

            start = a['fim']
            end   = start + timedelta(minutes=setup_min)

            rows.append({
                'maquina': a.get('maquina'),
                'inicio': start,
                'fim': end,
                'machine_y': a.get('machine_y', a.get('maquina')),
                'machine_label': a.get('machine_label', a.get('maquina')),
                'color_tag': 'SETUP',
                'Label': (
                    f"SETUP<br>"
                    f"Início: {start.strftime('%H:%M')}<br>"
                    f"Fim: {end.strftime('%H:%M')}<br>"
                    f"Tempo: {setup_min:.0f} min"
                ),
                'tipo': 'setup'
            })

    return pd.DataFrame(rows)

########### Função responsável por agrupar slots consecutivos ###############
def grouped_slots(slots):
    if not slots:
        return []
    slots = sorted(slots)
    grupos = [[slots[0]]]
    for s in slots[1:]:
        if s == grupos[-1][-1] + 1:
            grupos[-1].append(s)
        else:
            grupos.append([s])
    return grupos


# Função principal
def main():
    st.title("Visualização de Gantt – Planejamento de Produção")
    
    input_file = base_data / 'job_scheduling_input.json'
    output_file = base_data / 'job_scheduling_output.json'

    assingmnet_jobs_path = base_data / 'assignment_jobs.csv'
    assingmnet_jobs = pd.read_csv(assingmnet_jobs_path, parse_dates=["inicio", "fim", "not_before_date", "deadline"])

    with open(input_file, 'r') as f:
        input_data = json.load(f)
    with open(output_file, 'r') as f:
        output_data = json.load(f)

    #### Criação de estruturas para fácil acesso ####
    count_time_slots = input_data['count_time_slots']
    init = datetime.strptime(input_data['init_date'], '%Y-%m-%d %H:%M')
    init_dt = pd.Timestamp(datetime.strptime(input_data['init_date'], '%Y-%m-%d %H:%M'))

    init_date = pd.Timestamp(init).normalize()
    
    time_step = input_data['time_step']

    machine_id_to_name = {m['machine_id']: m['machine_name'] for m in input_data['machines']}
    machine_id_to_max_use = {m['machine_id']: m.get('max_use_day') for m in input_data['machines']}
    jobs_dict = {j['job_id']: j for j in input_data['jobs']}
    setup_dict = defaultdict(list)
    for setup in input_data['setups']:
        setup_dict[(setup['from_job_id'], setup['to_job_id'], setup['machine_id'])] = setup['setup_time']
    machines_dict = defaultdict(list)
    for machine in input_data['machines']:
        machines_dict[machine['machine_name']] = machine
    
    ###################################################
     # Estrutura para capturar facilmente o id
    job_index = {}
    for j in input_data['jobs']:
        key = (
            j.get ('op'),j.get('job_register_id'), (j.get('_kf_macho')),
        )
        
        job_index[key] = j


    ###Criação do dataframe que corresponde ao planejamento dado pelo otimizador ###
    schedule = []
    for mach in output_data['machines_scheduling']:
        m_id = mach['machine_id']
        m_name = machine_id_to_name.get(m_id, f"Machine_{m_id}")

        for job in mach['jobs']:
            j_id = job['job_id']
            j_reg_id = job['job_register_id']

            job_info = jobs_dict.get(j_id, {})
            config = job_info.get('config', None)
            release_slot = job_info.get('release_date_slot')
            deadline_slot = job_info.get('deadline_slot')

            start_slot, end_slot = job['start'], job['end']
            
            
            inicio = init_date + timedelta(minutes=start_slot * time_step)
            fim = init_date + timedelta(minutes=(end_slot + 1) * time_step)
            release_time = init_date + timedelta(minutes=release_slot * time_step)
            deadline_time = init_date + timedelta(minutes=deadline_slot * time_step)

            job_entry = {
                'op': job['op'],
                'caixa': f"{j_reg_id}",
                'kp': job_info['kp_fichaTecnica'],
                '_kf_macho':job_info['_kf_macho'],
                'job_id': j_id,
                'maquina': m_name,
                'inicio': inicio,
                'fim': fim,
                'not_before_date': release_time,
                'deadline': deadline_time,
                'processing_slots': job.get('processing_slots', []),
                'config': config,
                'tipo': 'job',
                'sub_machine': job.get('sub_machine'),
                'has_lateness': int(job_info.get('has_lateness', 0))
            }
            schedule.append(job_entry)

    # Adicionando a solução do modelo de assignment 
    for _, row in  assingmnet_jobs.iterrows():
        match_key = (row.get('op'), row.get('caixa'), row.get('_kf_macho'))
        matched_job = job_index.get(match_key)

        if matched_job is None: 
            continue

        m_id = matched_job['assigned_machine_id']
        m_name = machine_id_to_name.get(m_id, f"Machine_{m_id}")
        schedule.append({
            'op': row['op'],
            'caixa': row['caixa'],
            'kp': row['kp'],
            '_kf_macho':row['_kf_macho'],
            'job_id': matched_job['job_id'],
            'maquina': m_name,
            'inicio': row['inicio'],
            'fim': row['fim'],
            'not_before_date': row['not_before_date'],
            'deadline': row['deadline'],
            'processing_slots': [],
            'config': row['config'],
            'tipo': 'job',
            'sub_machine': row['sub_machine'],
            'has_lateness': row['has_lateness'],
            'fixed_time': True
        })

    df = pd.DataFrame(schedule)
    df['delay'] = (df['fim'] - df['deadline']).dt.days.clip(lower=0)

    

    # Cria a coluna tempo total apenas para referência geral
    #df['tempo_processamento_minutos_total'] = df['processing_slots'].apply(lambda s: len(s) * time_step)
    df['tempo_processamento_minutos_total'] = df.apply(
    lambda r: (len(r['processing_slots']) * time_step) 
              if (isinstance(r['processing_slots'], list) and len(r['processing_slots']) > 0 and not r.get('fixed_time', False))
              else int((r['fim'] - r['inicio']).total_seconds() // 60),
    axis=1
    )
    expanded_df = None

    expanded_rows = []

    for _, row in df.iterrows():
        if row.get('fixed_time', False) or not row.get('processing_slots'):
            # 1 bloco direto (opcional: quebrar por dia se quiser)
            new_row = row.copy()
            new_row['day'] = new_row['inicio'].date()
            new_row['tempo_processamento_minutos'] = int((new_row['fim'] - new_row['inicio']).total_seconds() // 60)
            expanded_rows.append(new_row)
            continue
    
        grupos_slots = grouped_slots(row['processing_slots'])

        for grupo in grupos_slots:
            slot_times = [
                {
                    'slot': slot,
                    'timestamp': init_date + timedelta(minutes=slot * time_step)
                }
                for slot in grupo
            ]
            slot_df = pd.DataFrame(slot_times)
            slot_df['day'] = slot_df['timestamp'].dt.date

            for dia, slots_do_dia in slot_df.groupby('day'):
                slot_start = slots_do_dia['slot'].min()
                slot_end = slots_do_dia['slot'].max() + 1

                inicio = init_date + timedelta(minutes=int(slot_start * time_step))
                fim = init_date + timedelta(minutes=int(slot_end * time_step))

                new_row = row.copy()
                new_row['inicio'] = inicio
                new_row['fim'] = fim
                new_row['day'] = dia
                new_row['tempo_processamento_minutos'] = len(slots_do_dia) * time_step

                expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)
    
    expanded_df['date'] = expanded_df['inicio'].dt.date
    expanded_df['sub_machine'] = expanded_df['sub_machine'].astype(int)

    expanded_df_with_band_list = []
    for (machine, date, sm), group in expanded_df.groupby(['maquina', 'date', 'sub_machine']):
        group = group.copy()
        group['machine_y'] = group['maquina'] + "_" + group['sub_machine'].astype(str)
        expanded_df_with_band_list.append(group)

    expanded_df_with_band = pd.concat(expanded_df_with_band_list, ignore_index=True)
    expanded_df_with_band['machine_label'] = expanded_df_with_band['maquina']
    expanded_df_with_band['color_tag'] = expanded_df_with_band.apply(
        lambda row: (
            'SETUP' if row['tipo'] == 'setup' else 
            ('JOB_ATRASADO' if int(row.get('has_lateness', 0)) == 1 else row['maquina'])
        ),
        axis=1
    )

    # Adição do setups #
    setup_df = process_setups(expanded_df_with_band, input_data['setups'], time_step)
    if not setup_df.empty:
        expanded_df_with_band = pd.concat([expanded_df_with_band, setup_df])

    
   # ------------------ Adiciona indisponibilidade fora do turno (agrupado por blocos e lida com dias seguintes) ------------------
    indisponiveis = []

    for machine in input_data['machines']:
        machine_name = machine['machine_name']
        start_slots = set(machine['start_slots'])  # slots válidos
        job_capacity = int(machine.get('job_capacity', 1))

        all_slots = list(range(count_time_slots))
        unavailable_slots = sorted([s for s in all_slots if s not in start_slots])
        grupos_slots = []

        # Agrupar slots consecutivos
        for s in unavailable_slots:
            if not grupos_slots or s != grupos_slots[-1][-1] + 1:
                grupos_slots.append([s])
            else:
                grupos_slots[-1].append(s)

        for grupo in grupos_slots:
            slot_start = grupo[0]
            slot_end = grupo[-1] + 1

            timestamp_start = pd.Timestamp(init_date).normalize() + timedelta(minutes=slot_start * time_step)
            timestamp_end = pd.Timestamp(init_date) + timedelta(minutes=slot_end * time_step)

            # Dividir o bloco se ele ultrapassar a meia-noite
            current_start = timestamp_start
            while current_start < timestamp_end:
                current_day = current_start.date()
                end_of_day = datetime.combine(current_day, datetime.max.time()).replace(hour=23, minute=59, second=59)
                current_end = min(timestamp_end, end_of_day + timedelta(seconds=1))

                for band_index in range(job_capacity):
                    machine_y = f"{machine_name}_{band_index}"
                    indisponiveis.append({
                        'inicio': current_start,
                        'fim': current_end,
                        'maquina': machine_name,
                        'machine_y': machine_y,
                        'machine_label': machine_name,
                        'color_tag': 'INDISPONIVEL',
                        'tipo': 'indisponivel',
                        'hover_label': 'Fora do turno'
                    })

                # Avança para o próximo dia
                current_start = datetime.combine(current_day + timedelta(days=1), datetime.min.time())

    # Adiciona os blocos de indisponibilidade ao DataFrame principal
    if indisponiveis:
        indisponiveis_df = pd.DataFrame(indisponiveis)
        expanded_df_with_band = pd.concat([expanded_df_with_band, indisponiveis_df])


    expanded_df_with_band['hover_label'] = expanded_df_with_band.apply(create_hover_label, axis=1)

    # Exibir gráficos no Streamlit
    st.subheader("Gráfico de Gantt Interativo")
    fig = create_week_gantt(expanded_df_with_band)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gráficos de Gantt por Dia")
    gantt_by_day = create_gantt_by_day(expanded_df_with_band)
    for day, fig in gantt_by_day.items():
        st.write(f"**Gantt para o dia {day.strftime('%d/%m/%Y')}**")
        st.plotly_chart(fig, use_container_width=True)
    

if __name__ == "__main__":
    main()