import os
import json
import shutil
import pandas as pd
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import pyarrow as pa
import pyarrow.parquet as pq
from unidecode import unidecode
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

weekday_idx = {
    "segunda": 0,
    "terca": 1,
    "quarta": 2,
    "quinta": 3,
    "sexta": 4,
    "sabado": 5,
    "domingo": 6,
}


def ensure_exists(path: Path, kind: str = "file") -> None:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {path}")


def ensure_model_config_directory(
    model_config_dir: Path, possible_sources: Iterable[Path]
) -> Path:
    model_config_dir.mkdir(parents=True, exist_ok=True)

    existing_files = (
        {item.name for item in model_config_dir.iterdir()}
        if model_config_dir.exists()
        else set()
    )

    for source in possible_sources:
        if not source or not source.exists():
            continue
        for item in source.iterdir():
            destination = model_config_dir / item.name
            if destination.name in existing_files:
                continue
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)

    return model_config_dir


@dataclass
class IOPaths:
    base_data: Path
    output_dir: Path
    input_model_dir: Path
    output_model_dir: Path
    model_config_dir: Optional[Path] = None

    @classmethod
    def build(
        cls,
        *,
        base_data: Path,
        output_dir: Path,
        model_config_dir: Optional[Path] = None,
        input_model_dir: Optional[Path] = None,
        output_model_dir: Optional[Path] = None,
    ) -> "IOPaths":
        input_dir = input_model_dir or output_dir
        output_dir_model = output_model_dir or output_dir
        if model_config_dir is None:
            candidate = base_data / "parameters"
            model_config_dir = candidate if candidate.exists() else None

        return cls(
            base_data=base_data,
            output_dir=output_dir,
            input_model_dir=input_dir,
            output_model_dir=output_dir_model,
            model_config_dir=model_config_dir,
        )


def parse_turnos_to_list(value):
    if pd.isna(value):
        return []
    if value == "geral":
        return [0]

    return [int(x.strip()) for x in str(value).split(",") if x.strip().isdigit()]


def normalize_string(value) -> str:
    """Normaliza string: remove acentos, minúsculas, sem espaços."""
    if pd.isna(value):
        return ""
    return unidecode(str(value)).lower().replace(" ", "")


def get_total_minutes(hora_str):
    hora = datetime.strptime(hora_str, "%H:%M")
    total_minutes = hora.hour * 60 + hora.minute
    return total_minutes

class JobSchedulingOutput:
    # --------------------------
    # Helpers de tempo (coerentes com o pipeline)
    # --------------------------
    @staticmethod
    def _slot_to_dt(init_date: pd.Timestamp, time_step: int, slot: Optional[int]) -> Optional[datetime]:
        if slot is None:
            return None
        return (init_date + timedelta(minutes=int(slot) * time_step)).to_pydatetime()

    @staticmethod
    def _dayidx_to_dt(init_date: pd.Timestamp, day_idx: Optional[int]) -> Optional[datetime]:
        if day_idx is None:
            return None
        return (init_date + timedelta(days=int(day_idx))).to_pydatetime()

    def _get_release_time(self, job_out: dict, job_in: dict, init_date: pd.Timestamp, time_step: int) -> Optional[datetime]:
        # prioridade: slots (input) → índices de dia (output/input)
        if job_in.get("release_date_slot") is not None:
            return self._slot_to_dt(init_date, time_step, job_in["release_date_slot"])
        if job_out.get("release_date_index") is not None:
            return self._dayidx_to_dt(init_date, job_out["release_date_index"])
        if job_in.get("release_date_index") is not None:
            return self._dayidx_to_dt(init_date, job_in["release_date_index"])
        return None

    def _get_deadline_time(self, job_out: dict, job_in: dict, init_date: pd.Timestamp, time_step: int) -> Optional[datetime]:
        # prioridade: slots (input ou output) → índices de dia (output/input)
        if job_in.get("deadline_slot") is not None:
            return self._slot_to_dt(init_date, time_step, job_in["deadline_slot"])
        if job_out.get("deadline_slot") is not None:
            return self._slot_to_dt(init_date, time_step, job_out["deadline_slot"])
        if job_out.get("deadline_date_index") is not None:
            return self._dayidx_to_dt(init_date, job_out["deadline_date_index"])
        if job_in.get("deadline_date_index") is not None:
            return self._dayidx_to_dt(init_date, job_in["deadline_date_index"])
        return None

    def run(self, base_data: Path, output_dir: Path, assignment_jobs: pd.DataFrame) -> pd.DataFrame:
        # --- Lê os arquivos do pipeline "macharia_*" ---
        input_file = output_dir / "macharia_input.json"
        output_file = output_dir / "macharia_output.json"

        with open(input_file, "r") as f:
            input_data = json.load(f)
        with open(output_file, "r") as f:
            output_data = json.load(f)

        # Seções
        gap_section = output_data.get("gap_optimization", {}) or {}
        js_section  = output_data.get("job_scheduling_optimization", {}) or {}

        gap_sched = gap_section.get("machines_scheduling", []) or []
        js_sched  = js_section.get("machines_scheduling", []) or []

        # Datas/tempo base
        init = datetime.strptime(input_data["init_date"], "%Y-%m-%d %H:%M")
        init_date = pd.Timestamp(init).normalize()
        time_step = int(input_data["time_step"])

        # Dicionários de apoio
        machine_id_to_name = {m["machine_id"]: m["machine_name"] for m in input_data["machines"]}
        jobs_dict = {j["job_id"]: j for j in input_data["jobs"]}

        # Setups (opcional nesta etapa — usado na apuração de tempo real quando necessário)
        setup_dict = {}
        for s in input_data.get("setups", []):
            setup_dict[(s["from_job_id"], s["to_job_id"], s["machine_id"])] = s["setup_time"]

        # Penalizados juntando os dois modelos + erros do input
        penal_jobs = []
        penal_jobs += list(js_section.get("not_satisfied_jobs", []))
        penal_jobs += list(gap_section.get("not_satisfied_jobs", []))
        jobs_error = [j for j in input_data.get("jobs", []) if j.get("Status_Processed")]
        for j in jobs_error:
            penal_jobs.append({
                "op": j.get("op"),
                "job_register_id": j.get("job_register_id"),
                "job_id": j.get("job_id"),
                "start": None,
                "end": None,
                "processing_slots": None,
                "sub_machine": None,
                "status_integration_id": j.get("status_integration_id"),
                "Status_Processed": j.get("Status_Processed"),
            })

        # --------------------------
        # Monta os agendamentos do Job Scheduling (já têm start/end em slots)
        # --------------------------
        schedule_rows = []

        for mach in js_sched:
            m_id = mach["machine_id"]
            m_name = machine_id_to_name.get(m_id, f"Machine_{m_id}")

            for job in mach.get("jobs", []):
                j_id = job["job_id"]
                job_in = jobs_dict.get(j_id, {})

                start_slot = job.get("start")
                end_slot   = job.get("end")

                inicio = init_date + timedelta(minutes=int(start_slot) * time_step) if start_slot is not None else None
                # end é inclusivo em slot → somar 1 slot para fechar o intervalo
                fim    = init_date + timedelta(minutes=(int(end_slot) + 1) * time_step) if end_slot is not None else None

                release_time  = self._get_release_time(job, job_in, init_date, time_step)
                deadline_time = self._get_deadline_time(job, job_in, init_date, time_step)

                schedule_rows.append({
                    "op": job.get("op", job_in.get("op")),
                    "caixa": f"{job.get('job_register_id')}",
                    "kp": job_in.get("kp_fichaTecnica"),
                    "_kf_macho": job_in.get("_kf_macho"),
                    "job_id": j_id,
                    "maquina": m_name,
                    "inicio": inicio,
                    "fim": fim,
                    "not_before_date": release_time,
                    "deadline": deadline_time,
                    "processing_slots": job.get("processing_slots", []),
                    "config": job.get("config") or job_in.get("config") or "",
                    "tipo": "job",
                    "sub_machine": job.get("sub_machine"),
                    "has_lateness": int(job_in.get("has_lateness", 0)),
                    "status_integration_id": job.get("status_integration_id", job_in.get("status_integration_id")),
                    "Status_Processed": job.get("Status_Processed"),
                    "work_minutes": None,             # reservado para GAP
                    "breaks_minutes": job.get("breaks_minutes"),
                })

        # --------------------------
        # Anexa solução vinda do GAP (já calculada e recebida como DataFrame)
        # --------------------------
        if assignment_jobs is not None and not assignment_jobs.empty:
            for _, row in assignment_jobs.iterrows():
                j_id = row["job_id"]
                job_in = jobs_dict.get(j_id, {})
                m_id = job_in.get("assigned_machine_id")
                m_name = machine_id_to_name.get(m_id, f"Machine_{m_id}")

                schedule_rows.append({
                    "op": row.get("op"),
                    "caixa": row.get("caixa"),
                    "kp": row.get("kp"),
                    "_kf_macho": row.get("_kf_macho"),
                    "job_id": j_id,
                    "maquina": m_name,
                    "inicio": row.get("inicio"),
                    "fim": row.get("fim"),
                    "not_before_date": row.get("not_before_date"),
                    "deadline": row.get("deadline"),
                    "processing_slots": [],  # já vem com tempo real no GAP
                    "config": row.get("config") or job_in.get("config") or "",
                    "tipo": "job",
                    "sub_machine": row.get("sub_machine"),
                    "has_lateness": int(row.get("has_lateness", 0)),
                    "fixed_time": True,
                    "status_integration_id": job_in.get("status_integration_id"),
                    "Status_Processed": "",
                    "work_minutes": row.get("tempo_processamento_minutos_total"),
                    "breaks_minutes": row.get("breaks_minutes", 0),
                })

        # Constrói DataFrame e métricas de atraso/tempo
        os.makedirs(output_dir, exist_ok=True)

        if not schedule_rows:
            logger.warning("Nenhum agendamento encontrado (JS + GAP).")
            return pd.DataFrame()

        df = pd.DataFrame(schedule_rows)

        # Coerção de tipos antes de calcular delay
        for col in ["inicio", "fim", "not_before_date", "deadline"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Delay em dias (>=0)
        if "fim" in df.columns and "deadline" in df.columns:
            df["delay"] = (df["fim"] - df["deadline"]).dt.days.fillna(0).clip(lower=0)
        else:
            df["delay"] = 0

        # Tempo de processamento (min)
        def _processing_minutes_row(r) -> Optional[float]:
            # 1) Se veio do GAP, work_minutes já inclui pausas/ociosos (ou calc do GAP)
            if pd.notna(r.get("work_minutes")):
                return float(r["work_minutes"])
            # 2) Se JS tem processing_slots → slots * time_step
            if isinstance(r.get("processing_slots"), list) and len(r["processing_slots"]) > 0 and not r.get("fixed_time", False):
                return float(len(r["processing_slots"]) * time_step)
            # 3) Fallback: diferença entre fim e inicio
            if pd.notna(r.get("inicio")) and pd.notna(r.get("fim")):
                return float((r["fim"] - r["inicio"]).total_seconds() / 60.0)
            return None

        df["tempo_processamento_minutos_total"] = df.apply(_processing_minutes_row, axis=1)


        # --------------------------
        # Anexa penalizados (sem início/fim)
        # --------------------------
        penal_rows = []
        for pj in penal_jobs:
            j_id = pj.get("job_id")
            job_in = jobs_dict.get(j_id, {})
            if not job_in:
                continue
            m_id = job_in.get("assigned_machine_id")
            m_name = machine_id_to_name.get(m_id, f"Machine_{m_id}")

            release_time  = self._get_release_time(pj, job_in, init_date, time_step)
            deadline_time = self._get_deadline_time(pj, job_in, init_date, time_step)

            penal_rows.append({
                "op": pj.get("op", job_in.get("op")),
                "caixa": pj.get("job_register_id"),
                "_kf_macho": job_in.get("_kf_macho"),
                "maquina": m_name,
                "inicio": pd.NaT,
                "fim": pd.NaT,
                "not_before_date": release_time,
                "deadline": deadline_time,
                "config": pj.get("config") or job_in.get("config") or "",
                "sub_machine": pj.get("sub_machine"),
                "delay": 0,
                "tempo_processamento_minutos_total": 0.0,
                "has_lateness": int(job_in.get("has_lateness", 0)),
                "status_integration_id": job_in.get("status_integration_id"),
                "Status_Processed": pj.get("Status_Processed"),
            })

        if penal_rows:
            df = pd.concat([df, pd.DataFrame.from_records(penal_rows)], ignore_index=True, sort=False)

        # Tipagem final e seleção de colunas
        df = df.astype({
            
            "op": "string",
            "caixa": "float",
            "_kf_macho": "float",
            "maquina": "string",
            "config": "string",
            "tempo_processamento_minutos_total": "Int64",
            "sub_machine": "Float64",
            "has_lateness": "Int64",
            "status_integration_id": "string",
            "Status_Processed": "string",
        })

        df_transformed = df[[
            "job_id",
            "op",
            "caixa",
            "_kf_macho",
            "maquina",
            "fim",
            "not_before_date",
            "deadline",
            "config",
            "tempo_processamento_minutos_total",
            "inicio",
            "status_integration_id",
            "Status_Processed",
        ]].rename(columns={
            "op": "OP",
            "deadline": "Deadline",
            "tempo_processamento_minutos_total": "tempo_processamento",
        })

        # Saída (nome alinhado ao pipeline)
        current_date = datetime.now().strftime("%Y-%m-%d")
        demanda_dir = output_dir / "demanda"
        demanda_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = demanda_dir / f"macharia{current_date}.parquet"
        csv_path     = demanda_dir / f"macharia{current_date}.csv"

        pq.write_table(pa.Table.from_pandas(df_transformed), parquet_path, coerce_timestamps="us", allow_truncated_timestamps=True)
        df_transformed.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info(" JobSchedulingOutput finalizado. Arquivos salvos:\n  - %s\n  - %s", parquet_path, csv_path)
        return df_transformed





class GapOutput:

    def calculate_times(self, t, base_dt, jobs, range_work_shifts):
        """
        Agenda trabalhos consumindo tempo sequencialmente através dos turnos.
        Se um trabalho atravessa um intervalo, sua duração total é estendida
        para incluir o tempo ocioso, sem criar um novo job.

        Args:
            t (datetime): O tempo atual a partir do qual o agendamento pode começar.
            base_dt (datetime): O datetime de referência para o início do dia (minuto 0).
            jobs (list): Uma lista de dicionários de trabalhos.
            range_work_shifts (list): Uma lista de tuplas de turnos (inicio, fim) em minutos.

        Returns:
            tuple: Uma tupla contendo:
                - O tempo final após o último trabalho agendado (datetime).
                - Uma nova lista de trabalhos agendados com um único 'start' e 'end'.
        """
        job_queue = list(jobs)
        scheduled_jobs = []
        sorted_shifts = sorted(range_work_shifts, key=lambda x: x[0])
        num_shifts = len(sorted_shifts)

        while job_queue:
            job = job_queue.pop(0)
            p = job["processing_minutes"]

            # Se um job tiver 0 minutos, pula para o próximo para evitar loops infinitos.
            if p <= 0:
                continue

            # --- 1. Encontrar o ponto de partida inicial ---
            current_time_min = (t - base_dt).total_seconds() / 60
            job_start_min = -1
            start_shift_idx = -1

            for i in range(num_shifts):
                shift_start, shift_end = sorted_shifts[i]
                # O início potencial é o máximo entre o tempo disponível e o início do turno.
                potential_start = max(current_time_min, shift_start)
                # Se o início potencial for antes do fim do turno, encontramos um slot.
                if potential_start < shift_end:
                    job_start_min = potential_start
                    start_shift_idx = i
                    break

            if job_start_min == -1:
                print(
                    f"Aviso: Não foi possível encontrar um turno com tempo disponível para o job: {job}"
                )
                continue

            # --- 2. Calcular o tempo de término, pulando os intervalos ---
            time_to_process = p
            job_end_min = job_start_min

            for i in range(start_shift_idx, num_shifts):
                shift_start, shift_end = sorted_shifts[i]

                # O ponto de partida real neste segmento de turno
                effective_start_in_segment = max(job_end_min, shift_start)

                time_available_in_segment = shift_end - effective_start_in_segment

                # Se o trabalho termina neste turno
                if time_to_process <= time_available_in_segment:
                    job_end_min = effective_start_in_segment + time_to_process
                    time_to_process = 0
                    break
                # Se o trabalho consome este turno e continua no próximo
                else:
                    time_to_process -= time_available_in_segment
                    job_end_min = shift_end  # O tempo avança até o final deste turno

            if time_to_process > 0:
                print(
                    f"Aviso: O job não pôde ser concluído. Faltou tempo nos turnos. {job}"
                )

            # --- 3. Criar o registro do job final ---
            final_job = job.copy()
            start_dt = base_dt + timedelta(minutes=job_start_min)
            end_dt = base_dt + timedelta(minutes=job_end_min)

            final_job["start"] = start_dt.isoformat(timespec="minutes")
            final_job["end"] = end_dt.isoformat(timespec="minutes")

            # ATUALIZA o tempo de processamento para refletir a duração total (trabalho + ociosidade)
            total_duration = job_end_min - job_start_min
            final_job["processing_minutes"] = total_duration

            scheduled_jobs.append(final_job)

            # Atualiza o relógio para o próximo job
            t = end_dt

        return t, scheduled_jobs

    def job_to_row(self, j, init_day, machine_info):
        release_date = init_day + timedelta(days=int(j["release_date_index"]))
        deadline_date = init_day + timedelta(days=int(j["deadline_date_index"]))
        assigned_day = init_day + timedelta(days=int(j["assigned_day"]))

        # Início e fim só de exemplo (coloque sua lógica real se já tiver)
        inicio = datetime.fromisoformat(j["start"])
        fim = datetime.fromisoformat(j["end"])
        delay_days = (fim.date() - deadline_date.date()).days
        delay = max(delay_days, 0)
        m_name = machine_info[j["machine_id"]]["machine_name"]
        return {
            "job_id": j['job_id'],
            "op": j["op"],
            "caixa": j["job_register_id"],
            "kp": j["kp_fichaTecnica"],
            "_kf_macho": j["_kf_macho"],
            "maquina": m_name,
            "inicio": inicio,
            "fim": fim,
            "not_before_date": release_date,
            "deadline": deadline_date,
            "config": j["config"],
            "sub_machine": 0,
            "has_lateness": j["has_lateness"],
            "dia": assigned_day,
            "delay": delay,
            "tempo_processamento_minutos_total": j["processing_minutes"],
        }

    """
    1. Inicialmente ordenamos os jobs em ordem crescente de processing time
    2. Colocamos o primeiro job ordenado em jobs_sequence, a partir do segundo job seguimos o seguinte fluxo:
    3. Vamos percorrer todos os jobs e pegar aquele que nao possui setup, se existir
    Removemos o tempo de setup que estava atrelado ao p_time do job anterior, se nao existir,
    colocamos o job na sequencia e mantem o tempo de setup
    """

    def order_jobs(self, jobs, pairs_not_setup, setup_info):
        pool = sorted(jobs, key=lambda j: j["processing_minutes"])
        if not pool:
            return []

        seq = [pool.pop(0)]  # pega o menor e remove do pool

        while pool:
            prev = seq[-1]
            compat = pairs_not_setup.get(prev["job_id"], [])

            # tente achar no pool alguém compatível (sem setup) com 'prev'
            pick_idx = None
            if compat:
                for idx, job in enumerate(pool):
                    if job["job_id"] in compat:
                        pick_idx = idx
                        break

            # Vamos sempre remover o tempo de setup do job anterior, caso nao exista, devo adicionar um job
            # Com id -1 e processing_minutes igual ao setup_time
            config = prev["config"]
            machine_id = prev["machine_id"]
            setup_time = setup_info.get((config, config, machine_id), 0)
            prev["processing_minutes"] = max(0, prev["processing_minutes"] - setup_time)

            if pick_idx is None:
                pick_idx = 0
                # No ultimo nao iremos adicionar
                if len(pool) >= 1:
                    job_setup = {"job_id": -1, "processing_minutes": setup_time}
                    seq.append(job_setup)

            seq.append(pool.pop(pick_idx))  # REMOVE do pool ao adicionar

        return seq

    def run(self, base_data, output_dir, model_config_dir):
        input_file = output_dir / "macharia_input.json"
        output_file = output_dir / "macharia_output.json"

        with open(input_file, "r") as f:
            data_input = json.load(f)
        with open(output_file, "r") as f:
            data_output = json.load(f)

        turnos_path = model_config_dir / "brut_shifts.csv"
        machine_information_path = model_config_dir / "brut_machine_information.csv"
        turnos_df = pd.read_csv(turnos_path, sep=r"\s*,\s*", engine="python")
        machine_shifts_df = pd.read_csv(
            machine_information_path,
            converters={
                "turnos": parse_turnos_to_list,
                "processo": normalize_string,
                "recurso": normalize_string,
            },
        )

        machines_scheduling = data_output["gap_optimization"]["machines_scheduling"]


        init_date = datetime.fromisoformat(data_input["init_date"])

        pairs_not_setup = data_input["pairs_not_setup"]
        setup_data = data_input["setup_data"]

        setup_info = defaultdict(list)
        for setup in setup_data:
            from_config = setup["from_config"]
            to_config = setup["to_config"]
            machine_id = setup["machine_id"]

            setup_info[(from_config, to_config, machine_id)] = int(setup["setup_time"])

        machine_info = defaultdict()
        for machine in data_input["machines"]:
            machine_info[machine["machine_id"]] = machine

        rows_data = []
        total_jobs = []
        for machine in machines_scheduling:
            jobs = machine["jobs"]
            machine_id = machine["machine_id"]
            machine_name = machine["machine_name"]
            
            jobs_by_day = defaultdict(list)
            for job in jobs:
                jobs_by_day[job["assigned_day"]].append(job)

            machine_shifts = machine_shifts_df[
                machine_shifts_df["recurso"] == machine_name.split("_")[-1]
            ][
                "turnos"
            ]  # Turnos permitidos para o recurso

            current_pairs_not_setup = defaultdict(list)
            for p in pairs_not_setup:
                if p["machine_id"] == machine_id:
                    current_pairs_not_setup[p["job_i"]].append(p["job_j"])
                    current_pairs_not_setup[p["job_j"]].append(p["job_i"])

            # A ideia eh percorrer os jobs do primeiro dia ate o ultimo
            # Segundamente, captura os jobs que possuem configuracao A e configuracao B
            for day in sorted(jobs_by_day.keys()):

                base_dt = init_date + timedelta(days=day)
                day_work_shifts = list()
                range_work_shifts = list()
                
                for _, row in turnos_df.iterrows():
                    current_turno = row["turno"]
                    if row["turno"] == "geral":
                      current_turno = 0
                    if (
                        int(current_turno) in machine_shifts.iloc[0]
                        and weekday_idx[row["dia"]] == day
                    ):
                        ini = get_total_minutes(row["inicio"])
                        brk1 = get_total_minutes(row["intervalo_inicio"])
                        brk2 = get_total_minutes(row["intervalo_fim"])
                        end = get_total_minutes(row["fim"])

                        day_work_shifts += [ini, brk2]
                        range_work_shifts.append((ini, brk1))
                        range_work_shifts.append((brk2, end))

                day_work_shifts.sort()
                day_jobs = jobs_by_day[day]

                # Primeiro devemos verificar se nos jobs lateness tem duas configuracoes diferentes
                has_config_A = False
                has_config_B = False
                jobs_lateness = [job for job in day_jobs if job["has_lateness"]]

                for job in jobs_lateness:
                    if job["config"] == "A":
                        has_config_A = True
                    if job["config"] == "B":
                        has_config_B = True
                jobs_B_lateness = []
                jobs_A_lateness = []
                if has_config_B:
                    jobs_B_lateness = self.order_jobs(
                        [
                            job
                            for job in day_jobs
                            if job["config"] == "B" and job["has_lateness"]
                        ],
                        current_pairs_not_setup,
                        setup_info,
                    )
                    c_setup = setup_info.get(("B", "A", machine_id), 0)
                    if c_setup:
                        jobs_B_lateness.append(
                            {"job_id": -1, "processing_minutes": c_setup}
                        )

                if has_config_A:
                    jobs_A_lateness = self.order_jobs(
                        [
                            job
                            for job in day_jobs
                            if job["config"] == "A" and job["has_lateness"]
                        ],
                        current_pairs_not_setup,
                        setup_info,
                    )
                
                t = base_dt + timedelta(minutes=day_work_shifts.pop(0))

                # agenda de jobs_lateness_no_config
                jobs_lateness_no_config = self.order_jobs(
                    [
                        job
                        for job in day_jobs
                        if not job["config"] and job["has_lateness"]
                    ],
                    current_pairs_not_setup,
                    setup_info,
                )

                t, jobs_lateness_no_config = self.calculate_times(
                    t, base_dt, jobs_lateness_no_config, range_work_shifts
                )

                # agenda de Jobs_B_lateness
                t, jobs_B_lateness = self.calculate_times(
                    t, base_dt, jobs_B_lateness, range_work_shifts
                )
                # agenda de jobs_A_lateness
                t, jobs_A_lateness = self.calculate_times(
                    t, base_dt, jobs_A_lateness, range_work_shifts
                )

                # Construindo estrutura para recuperar os jobs que nao possuem setup com determinado job_id
                jobs_A = self.order_jobs(
                    [
                        job
                        for job in day_jobs
                        if job["config"] == "A" and not job["has_lateness"]
                    ],
                    current_pairs_not_setup,
                    setup_info,
                )
                # Devemos adicionar um job que indica setup de A para B
                c_setup = setup_info.get(("A", "B", machine_id), 0)
                if c_setup:
                    jobs_A.append({"job_id": -1, "processing_minutes": c_setup})
                jobs_B = self.order_jobs(
                    [
                        job
                        for job in day_jobs
                        if job["config"] == "B" and not job["has_lateness"]
                    ],
                    current_pairs_not_setup,
                    setup_info,
                )
                # Agenda de A
                t, jobs_A = self.calculate_times(t, base_dt, jobs_A, range_work_shifts)
                # Agenda de B
                t, jobs_B = self.calculate_times(t, base_dt, jobs_B, range_work_shifts)

                jobs_no_config = self.order_jobs(
                    [
                        job
                        for job in day_jobs
                        if not job["config"] and not job["has_lateness"]
                    ],
                    current_pairs_not_setup,
                    setup_info,
                )
                # agenda jobs sem configuracao
                t, jobs_no_config = self.calculate_times(
                    t, base_dt, jobs_no_config, range_work_shifts
                )

                day_seq = (
                    jobs_B_lateness
                    + jobs_A_lateness
                    + jobs_A
                    + jobs_B
                    + jobs_no_config
                    + jobs_lateness_no_config
                )
                
                day_seq = sorted(day_seq, key=lambda job: job["start"])
                total_jobs += day_seq

        df_jobs = pd.DataFrame(
            [
                self.job_to_row(j, init_date, machine_info)
                for j in total_jobs
                if j["job_id"] != -1
            ]
        )
        # So estamos salvando por conta do dashboard
        #assignment_jobs_path = output_dir / "assignment_jobs.csv"
        #df_jobs.to_csv(assignment_jobs_path, index=False)
        return df_jobs


# Argument parsing
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Processa arquivos locais de demanda já exportados para data/raw e gera as "
            "instâncias do job scheduling."
        )
    )
    default_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument(
        "--dt",
        required=True,
        help="Partição dt a ser processada (formato YYYY-MM-DD).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=default_root.joinpath("data", "raw"),
        help="Diretório raiz com os lotes importados (default: data/raw).",
    )
    parser.add_argument(
        "--trusted-root",
        type=Path,
        default=default_root.joinpath("data", "trusted"),
        help="Diretório raiz das saídas processadas (default: data/trusted).",
    )
    parser.add_argument(
        "--model-config-dir",
        type=Path,
        default=default_root.joinpath("data", "raw", "model_config"),
        help="Diretório compartilhado para os parâmetros do modelo (default: data/raw/model_config).",
    )
    parser.add_argument(
        "--only-status",
        nargs="+",
        default=None,
        help="Lista de status (nomes das pastas) a processar. Se omitido, processa todos os disponíveis.",
    )
    return parser.parse_args()


def main() -> None:

    args = parse_args()
    try:
        dt_obj = datetime.strptime(args.dt, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Data inválida para --dt: {args.dt}") from exc

    date_slug = dt_obj.strftime("%d%m%Y")
    raw_root: Path = args.raw_root
    trusted_root: Path = args.trusted_root
    model_config_dir: Path = args.model_config_dir
    date_dir = raw_root / date_slug

    base_data: Path = args.raw_root
    output_dir: Path = args.trusted_root

    logger.info("Base data dir: %s", base_data)
    logger.info("Output dir: %s", output_dir)

    ensure_exists(date_dir, kind="dir")

    available_status_dirs = [
        (item.name, item)
        for item in sorted(date_dir.iterdir())
        if item.is_dir() and (item / "demanda").is_dir()
    ]

    if not available_status_dirs:
        raise SystemExit(
            f"Nenhum lote de demanda encontrado em {date_dir}. Certifique-se de executar a ingestão antes."
        )

    if args.only_status:
        requested = {
            status.strip().lower() for status in args.only_status if status.strip()
        }
        found_map = {name.lower(): path for name, path in available_status_dirs}
        missing = sorted(requested - set(found_map.keys()))
        if missing:
            raise SystemExit(
                "Os seguintes status não foram encontrados na pasta local: "
                + ", ".join(missing)
            )
        status_to_process = [
            (name, found_map[name.lower()]) for name in args.only_status
        ]
    else:
        status_to_process = available_status_dirs

    possible_parameter_sources = [date_dir / "parameters", raw_root / "parameters"]
    model_config_dir = ensure_model_config_directory(
        model_config_dir, possible_parameter_sources
    )

    for status_label, dataset_dir in status_to_process:
        output_dir = trusted_root / date_slug / status_label
        paths = IOPaths.build(
            base_data=dataset_dir,
            output_dir=output_dir,
            model_config_dir=model_config_dir,
        )
        base_data = paths.base_data
        output_dir = paths.output_dir
        # Process data
        logger.info("Processing input data...")
        assingment_jobs = GapOutput().run(base_data, output_dir, model_config_dir)
        JobSchedulingOutput().run(base_data, output_dir, assingment_jobs)

        logger.info("Processing completed successfully.")


if __name__ == "__main__":
    main()