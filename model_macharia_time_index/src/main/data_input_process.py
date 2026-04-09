from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from unidecode import unidecode

# -----------------------------------------------------------------------------
# Config & Logging
# -----------------------------------------------------------------------------

LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("job-scheduling-builder")


# -----------------------------------------------------------------------------
# Constantes padrão
# -----------------------------------------------------------------------------

WORK_DAYS = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
DEFAULT_DAY_START = "2025-09-15 00:00"
DEFAULT_TIME_STEP = 5  # minutos
SLOTS_PER_DAY_DEFAULT = (24 * 60) // DEFAULT_TIME_STEP

# -----------------------------------------------------------------------------
# Funções auxiliares
# -----------------------------------------------------------------------------


def normalize_string(value) -> str:
    """Normaliza string: remove acentos, minúsculas, sem espaços."""
    if pd.isna(value):
        return ""
    return unidecode(str(value)).lower().replace(" ", "")


def append_df_to_csv(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()  # True na 1ª vez
    df.to_csv(path, mode="a", header=write_header, index=False)


def parse_turnos_to_list(value) -> List[int]:
    """Converte string de turnos para lista de ints, ou 'geral' -> [0]."""
    if pd.isna(value):
        return []
    if value == "geral":
        return [0]
    return [int(x.strip()) for x in str(value).split(",") if x.strip().isdigit()]


def parse_turno_to_int(val) -> int:
    """Mapeia 'geral' -> 0; caso contrário int(val)."""
    if val == "geral":
        return 0
    return int(val)


def get_hour_slot(hora_str: str, time_step: int) -> int:
    hora = datetime.strptime(hora_str, "%H:%M")
    total_min = hora.hour * 60 + hora.minute
    return total_min // time_step


def get_interval_slots(
    init_time: str, end_time: str, time_step: int, slots_per_day: int
) -> Tuple[int, int]:
    start_slot = get_hour_slot(init_time, time_step)
    end_slot = get_hour_slot(end_time, time_step)
    if start_slot <= end_slot:
        return start_slot, end_slot
    # cruza meia-noite
    return start_slot, slots_per_day + end_slot


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


# -----------------------------------------------------------------------------
# Dataclasses de Config
# -----------------------------------------------------------------------------
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


@dataclass
class RunConfig:
    day_start: pd.Timestamp
    time_step: int
    slots_per_day: int


def safe_int(value):
    """Converte valor em int, tratando vazios, NaN e strings inválidas."""
    try:
        if pd.isna(value) or value == "":
            return 0
        return int(float(value))
    except Exception:
        return 0




class SchedulingInputBuilder:
    """
    Classe unificada para gerar instâncias de input dos modelos GAP e Time-Index.
    Usa o parâmetro `mode` para alternar entre comportamentos específicos.
    """

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    # -------------------------------------------------------------------------
    # 1. MÁQUINAS
    # -------------------------------------------------------------------------

    def build_machines_information(
        self,
        demand_df: pd.DataFrame,
        machines_df: pd.DataFrame,
        caps_df: pd.DataFrame,
        shifts_df: pd.DataFrame,
        business_days: pd.DatetimeIndex,
        mode: str = "timeindex",
    ) -> List[Dict]:
        time_step = self.cfg.time_step
        slots_per_day = self.cfg.slots_per_day
        count_time_slots = len(business_days) * slots_per_day

        machines_information: List[Dict] = []
        index_machine = 0

        for process in demand_df["processo"].unique():
            recursos = (
                demand_df.loc[demand_df["processo"] == process, "recurso"]
                .unique()
                .tolist()
            )

            for recurso_type in recursos:
                machine_name = f"{process}_{recurso_type}"

                # capacidade de jobs da máquina (recurso)
                mi_rows = machines_df.query("recurso == @recurso_type")
                if mi_rows.empty:
                    continue
                job_capacity = int(mi_rows.iloc[0]["maximo_caixas"])

                # turnos válidos (idêntico ao original)
                current_shifts_df = machines_df.query("processo == @process")
                if current_shifts_df.shape[0] > 1:
                    current_shifts = machines_df.loc[
                        machines_df["recurso"] == recurso_type, "turnos"
                    ].to_list()[0]
                else:
                    current_shifts = current_shifts_df["turnos"].to_list()[0]

                # slots de trabalho válidos (a partir dos turnos da semana)
                slots_valid_machine: set[int] = set()
                for count_day, day in enumerate(business_days):
                    day_name = WORK_DAYS[day.weekday()]
                    start_slot_day = count_day * slots_per_day
                    day_shifts = shifts_df.loc[shifts_df["dia"] == day_name]
                    machine_shifts = day_shifts[day_shifts["turno"].isin(current_shifts)]

                    for _, row in machine_shifts.iterrows():
                        s1, e1 = get_interval_slots(
                            row["inicio"], row["intervalo_inicio"], time_step, slots_per_day
                        )
                        s2, e2 = get_interval_slots(
                            row["intervalo_fim"], row["fim"], time_step, slots_per_day
                        )
                        # Manhã
                        for slot in range(start_slot_day + s1, start_slot_day + e1):
                            if slot < count_time_slots:
                                slots_valid_machine.add(slot)
                        # Tarde
                        for slot in range(start_slot_day + s2, start_slot_day + e2):
                            if slot < count_time_slots:
                                slots_valid_machine.add(slot)

                start_slots_sorted = sorted(slots_valid_machine)

                # capacidade min/max por dia (em minutos e em slots), limitada pelos turnos do dia
                caps_rows = caps_df.query("recurso == @recurso_type")
                if caps_rows.empty:
                    continue
                process_max_minutes = float(caps_rows["max_dia"].iloc[0])
                process_min_minutes = float(caps_rows["min_dia"].iloc[0])
                process_max_slots = int(process_max_minutes // time_step)
                process_min_slots = int(process_min_minutes // time_step)

                if mode == "gap":
                    # ===== GAP: work_time_per_day em minutos, respeitando disponibilidade do dia =====
                    work_time_per_day: Dict[str, Dict[str, float]] = {}
                    for count_day, day in enumerate(business_days):
                        start_slot_day = count_day * slots_per_day
                        end_slot_day = start_slot_day + slots_per_day
                        todays_slots = {
                            s for s in start_slots_sorted if start_slot_day <= s < end_slot_day
                        }
                        avail_today_min = len(todays_slots) * time_step

                        max_work_time = min(process_max_minutes, avail_today_min)
                        min_work_time = min(process_min_minutes, avail_today_min)

                        work_time_per_day[str(count_day)] = {
                            "min": float(min_work_time),
                            "max": float(max_work_time),
                        }

                    machines_information.append(
                        {
                            "machine_id": index_machine,
                            "machine_name": machine_name,
                            "job_capacity": job_capacity,
                            "work_time_per_day": work_time_per_day,
                        }
                    )

                else:
                    # ===== TIME-INDEX: time_capacity por faixas de slots, respeitando disponibilidade =====
                    slots_ranges: List[Tuple[int, int]] = []
                    slots_capacity: List[Tuple[int, int]] = []
                    for count_day, day in enumerate(business_days):
                        start_slot_day = count_day * slots_per_day
                        end_slot_day = start_slot_day + slots_per_day
                        todays_slots = {
                            s for s in start_slots_sorted if start_slot_day <= s < end_slot_day
                        }
                        avail_today_slots = len(todays_slots)

                        min_cap_today = min(process_min_slots, avail_today_slots)
                        max_cap_today = min(process_max_slots, avail_today_slots)

                        slots_ranges.append((start_slot_day, end_slot_day))
                        slots_capacity.append((min_cap_today, max_cap_today))

                    time_capacity = {"slots": slots_ranges, "slots_capacity": slots_capacity}

                    machines_information.append(
                        {
                            "machine_id": index_machine,
                            "machine_name": machine_name,
                            "job_capacity": job_capacity,
                            "time_capacity": time_capacity,
                            "start_slots": start_slots_sorted,  # <<< correto
                        }
                    )

                index_machine += 1

        return machines_information

    # -------------------------------------------------------------------------
    # 2. JOBS
    # -------------------------------------------------------------------------
    def build_jobs_information(
        self,
        demand_df: pd.DataFrame,
        machines_information: List[Dict],
        business_days: pd.DatetimeIndex,
        mode: str = "timeindex",
    ) -> List[Dict]:
        """
        Cria lista de jobs.
        - mode="gap": usa índices de dias (release_date_index, deadline_date_index)
        - mode="timeindex": usa slots (release_date_slot, deadline_slot, processing_slots)
        """
        jobs_info = []
        time_step = self.cfg.time_step
        day_start = self.cfg.day_start

        for m in machines_information:
            machine_id = m["machine_id"]
            machine_name = m["machine_name"]

            df = demand_df.loc[demand_df["machine_name"] == machine_name]
            if df.empty:
                continue

            for _, row in df.iterrows():
                # Ignorar registros incompletos
                if pd.isna(row.get("not_before_date")) or pd.isna(row.get("deadline")):
                    continue

                qm = pd.to_numeric(row.get("qtd_moldes"), errors="coerce")
                tc = pd.to_numeric(row.get("tempo_ciclo"), errors="coerce")
                if pd.isna(qm) or pd.isna(tc):
                    continue

                # Tempo total de processamento
                processing_minutes = float(qm) * float(tc)
                if machine_name == "pepset_carrossel":
                    processing_minutes *= m["job_capacity"]
                    enhance_processing_time = True
                else:
                     enhance_processing_time = False



                # --- GARANTA ID VÁLIDO ---
                jid = row.get("job_id")
                if pd.isna(jid):
                    # se não houver job_id, não crie o job (evita virar 0)
                    continue
                jid = int(jid)  # NÃO usar safe_int aqui, 0 pode ser válido (primeiro job)

                job_base = {
                    "job_id": jid,
                    "op": safe_int(row.get("_kf_producao")),
                    "kp_fichaTecnica": safe_int(row.get("_kf_FichaTecnica")),
                    "_kf_macho": safe_int(row.get("_kf_macho")),
                    "job_register_id": safe_int(row.get("caixa")),
                    "assigned_machine_id": machine_id,
                    "has_lateness": safe_int(row.get("lateness")),
                    "status_integration_id": row.get("status_integration_id"),
                    "Status_Processed": row.get("Status_Processed", ""),
                    "enhance_processing_time": enhance_processing_time,
                }

                if mode == "gap":
                    release_index = (abs(business_days - row["not_before_date"])).argmin()
                    deadline_index = (abs(business_days - row["deadline"])).argmin()

                    # se deadline cair dentro da semana, soma +1 (representa "até o fim do dia")
                    if row["deadline"].normalize() <= business_days[-1]:
                        deadline_index = min(deadline_index + 1, len(business_days))


                    job = {
                        **job_base,
                        "processing_minutes": processing_minutes,
                        "release_date_index": int(release_index),
                        "deadline_date_index": int(deadline_index),
                        
                    }

                else:  # mode == "timeindex"
                    release_slot = int(max(0, (row["not_before_date"] - day_start).total_seconds() / 60 // time_step))
                    deadline_slot = int(max(0, (row["deadline"] - day_start).total_seconds() / 60 // time_step))
                    processing_slots = math.ceil(processing_minutes / time_step)

                    job = {
                        **job_base,
                        "processing_slots": processing_slots,
                        "release_date_slot": release_slot,
                        "deadline_slot": deadline_slot,
                    }

                jobs_info.append(job)

        return jobs_info


    # -------------------------------------------------------------------------
    # 3. SETUPS (comum)
    # -------------------------------------------------------------------------
    def build_setups(
        self,
        machines_information: List[Dict],
        setup_df: pd.DataFrame,
        setup_df_time: pd.DataFrame,
        jobs_df: pd.DataFrame,
    ) -> List[Dict]:
        """
        Retorna lista de setups entre jobs (from_job_id -> to_job_id) com tempos em slots.
        """
        time_step = self.cfg.time_step

        # Mapa rápido: recurso presente no nome da máquina -> machine_id
        recurso_to_machine_ids: Dict[str, List[int]] = defaultdict(list)
        for m in machines_information:
            _, recurso = m["machine_name"].split("_", 1)
            recurso_to_machine_ids[recurso].append(m["machine_id"])

        # (kp_macho, machine_id) -> {"not_setup": [...], "config": str}
        kp_macho_data: Dict[Tuple[str, int], Dict] = {}
        for _, row in setup_df.iterrows():
            recurso = normalize_string(row["recurso"])
            machine_ids = recurso_to_machine_ids.get(recurso, [])
            if not machine_ids:
                continue

            macho_info = re.sub(r"[A-Za-z]+", "", str(row["_kp_macho"]).replace(" ", ""))
            if "/" in macho_info:
                parts = macho_info.split("/")
                for m_id in machine_ids:
                    for i, left in enumerate(parts):
                        not_setup = [p for j, p in enumerate(parts) if j != i]
                        kp_macho_data[(left, m_id)] = {
                            "not_setup": not_setup,
                            "config": row["configuracao"],
                        }
            else:
                for m_id in machine_ids:
                    kp_macho_data[(macho_info, m_id)] = {
                        "not_setup": [],
                        "config": row["configuracao"],
                    }

        # tempos de setup: (from_config, to_config, machine_id) -> slots
        setup_time_data: Dict[Tuple[str, str, int], int] = {}
        for _, row in setup_df_time.iterrows():
            recurso = normalize_string(row["recurso"])
            machine_ids = recurso_to_machine_ids.get(recurso, [])
            if not machine_ids:
                continue
            for m_id in machine_ids:
                key = (row["de_config"], row["para_config"], m_id)
                setup_time_data[key] = math.ceil(float(row["setup_time_min"]) / time_step)

        # adiciona coluna 'config' em jobs (por (_kf_macho, machine_id))
        jobs_df = jobs_df.copy()
        jobs_df["config"] = None
        for (kp, m_id), info in kp_macho_data.items():
            mask = jobs_df["_kf_macho"].eq(int(kp)) & jobs_df["assigned_machine_id"].eq(m_id)
            jobs_df.loc[mask, "config"] = info["config"]
        # gera combinações dois-a-dois por máquina, respeitando not_setup
        setups_information: List[Dict] = []
        for (c_kp, c_m_id), c_info in kp_macho_data.items():
            not_setup = set(c_info["not_setup"])

            mask_src = (
                jobs_df["_kf_macho"].eq(int(c_kp))
                & jobs_df["assigned_machine_id"].eq(c_m_id)
            )
            src_jobs = jobs_df.loc[mask_src]
            if src_jobs.empty:
                continue

            # todos os outros KP na mesma máquina, exceto os em not_setup
            kps_same_machine = {
                kp
                for (kp, mid), _ in kp_macho_data.items()
                if mid == c_m_id and kp != c_kp and kp not in not_setup
            }

            for _, src in src_jobs.iterrows():
                from_job_id = int(src["job_id"])
                from_config = src["config"]
                from_macho = int(src["_kf_macho"])

                for kp in kps_same_machine:
                    mask_dst = (
                        jobs_df["_kf_macho"].eq(int(kp))
                        & jobs_df["assigned_machine_id"].eq(c_m_id)
                    )
                    dst_jobs = jobs_df.loc[mask_dst]
                    if dst_jobs.empty:
                        continue

                    for _, dst in dst_jobs.iterrows():
                        to_job_id = int(dst["job_id"])
                        to_config = dst["config"]
                        to_macho = int(dst["_kf_macho"])

                        #  evita arco para o mesmo job
                        if to_job_id == from_job_id:
                            continue

                        setup_slots = setup_time_data.get(
                            (from_config, to_config, c_m_id), 0
                        )

                        setups_information.append(
                            {
                                "from_macho": from_macho,
                                "from_job_id": from_job_id,
                                "to_job_id": to_job_id,
                                "to_macho": to_macho,
                                "setup_time": int(setup_slots),
                                "machine_id": int(c_m_id),
                            }
                        )

        return setups_information
    def _build_setups_and_pairs_exact(
        self,
        machines_information: List[Dict],
        setup_df: pd.DataFrame,
        setup_times_df: pd.DataFrame,
        jobs_information: List[Dict],
    ) -> Tuple[pd.DataFrame, List[Dict], List[Dict], List[Dict]]:
        """
        Constrói dados de setups (setup_data) e pares sem setup (pairs_not_setup),
        garantindo correspondência correta entre recurso e machine_id.
        Idêntico à lógica original do GapInput.
        """
        from collections import defaultdict

        jobs_information_df = pd.DataFrame(jobs_information).copy()
        if "config" not in jobs_information_df.columns:
            jobs_information_df["config"] = None

        # ===========================================================
        # Mapeamento robusto: recurso -> machine_id
        # ===========================================================
        recurso_to_machine_ids: Dict[str, List[int]] = defaultdict(list)
        for m in machines_information:
            # extrai apenas o trecho depois do "_" (ex: "Cold Box Soprado" -> "soprado")
            recurso_part = m["machine_name"].split("_", 1)[-1].lower()
            recurso_to_machine_ids[normalize_string(recurso_part)].append(m["machine_id"])

        # ===========================================================
        # (kp_macho, machine_id) -> {not_setup: [...], config: str}
        # ===========================================================
        kp_macho_data: Dict[Tuple[int, int], Dict] = {}

        for _, row in setup_df.iterrows():
            recurso = normalize_string(str(row.get("recurso", "")))
            macho_info = re.sub(r"[A-Za-z]+", "", str(row.get("_kp_macho", "")).replace(" ", ""))
            config = str(row.get("configuracao", ""))

            machine_ids = recurso_to_machine_ids.get(recurso, [])
            if not machine_ids:
                continue

            if "/" in macho_info:
                parts = [int(x) for x in macho_info.split("/") if x.strip()]
                for m_id in machine_ids:
                    for i, left in enumerate(parts):
                        not_setup = [p for j, p in enumerate(parts) if j != i]
                        kp_macho_data[(left, m_id)] = {"not_setup": not_setup, "config": config}
            else:
                for m_id in machine_ids:
                    if macho_info.strip() == "":
                        continue
                    kp_macho_data[(int(macho_info), m_id)] = {"not_setup": [], "config": config}

        # ===========================================================
        # Tempos (from_config, to_config, machine_id) -> minutos
        # ===========================================================
        setup_time_data: Dict[Tuple[str, str, int], int] = {}
        setup_time_rows: List[Dict] = []
        for _, row in setup_times_df.iterrows():
            recurso = normalize_string(str(row.get("recurso", "")))
            machine_ids = recurso_to_machine_ids.get(recurso, [])
            if not machine_ids:
                continue
            for m_id in machine_ids:
                key = (row["de_config"], row["para_config"], int(m_id))
                setup_time_data[key] = int(row["setup_time_min"])
                setup_time_rows.append(
                    {
                        "from_config": row["de_config"],
                        "to_config": row["para_config"],
                        "machine_id": int(m_id),
                        "setup_time": int(row["setup_time_min"]),
                    }
                )

        # ===========================================================
        # Atribui config e soma setup(A->A) nos jobs (modo GAP)
        # ===========================================================
        for (kp, m_id), info in kp_macho_data.items():
            mask = jobs_information_df["_kf_macho"].eq(int(kp)) & jobs_information_df["assigned_machine_id"].eq(int(m_id))
            jobs_information_df.loc[mask, "config"] = info["config"]
            self_setup = setup_time_data.get((info["config"], info["config"], int(m_id)), 0)
            if "processing_minutes" in jobs_information_df.columns and self_setup:
                jobs_information_df.loc[mask, "processing_minutes"] += int(self_setup)

        # ===========================================================
        # pairs_not_setup — gerado apenas entre machos/máquinas com setup permitido
        # ===========================================================
        pairs_no_setup: List[Dict] = []
        triplice_no_setup: List[Dict] = []

        for _, row in jobs_information_df.iterrows():
            c_kp = int(row["_kf_macho"])
            c_machine_id = int(row["assigned_machine_id"])
            c_id = int(row["job_id"])

            # candidatos na mesma máquina
            jobs_same_m = jobs_information_df.loc[
                jobs_information_df["assigned_machine_id"].eq(c_machine_id)
            ]

            # machos compatíveis (sem setup entre si)
            kps_not_setup = [c_kp]
            if (c_kp, c_machine_id) in kp_macho_data:
                kps_not_setup += kp_macho_data[(c_kp, c_machine_id)]["not_setup"]

            jobs_sem_setup = jobs_same_m.loc[
                jobs_same_m["_kf_macho"].isin(kps_not_setup)
                & jobs_same_m["job_id"].ne(c_id)
            ]

            for j in jobs_sem_setup["job_id"].unique().tolist():
                if j == c_id:
                    continue
                pairs_no_setup.append(
                    {"machine_id": c_machine_id, "job_i": c_id, "job_j": int(j)}
                )

        # ===========================================================
        # Debug opcional
        # ===========================================================
        # print(f"[DEBUG] kp_macho_data: {len(kp_macho_data)} combinações")
        # print(f"[DEBUG] setup_data: {len(setup_time_rows)} registros")
        # print(f"[DEBUG] pairs_not_setup: {len(pairs_no_setup)} pares")

        return jobs_information_df, setup_time_rows, pairs_no_setup, triplice_no_setup


    # -------------------------------------------------------------------------
    # 4. MÉTODO PRINCIPAL
    # -------------------------------------------------------------------------
    def create_input(
        self,
        demand_df,
        machines_df,
        caps_df,
        shifts_df,
        business_days,
        setup_df,
        setup_times_df,
        mode: str,
    ) -> Dict:
        machines = self.build_machines_information(
            demand_df, machines_df, caps_df, shifts_df, business_days, mode
        )
        # jobs base
        jobs = self.build_jobs_information(demand_df, machines, business_days, mode)

        setup_data: List[Dict] = []
        pairs_not_setup: List[Dict] = []

        # === versão idêntica ao original ===
        if mode == "gap":
            jobs_df, setup_time_rows, pairs_no_setup, _ = self._build_setups_and_pairs_exact(
                machines_information=machines,
                setup_df=setup_df,
                setup_times_df=setup_times_df,
                jobs_information=jobs,
                
            )
            jobs = jobs_df.to_dict(orient="records")
            setup_data = setup_time_rows
            pairs_not_setup = pairs_no_setup
            setups = []  # no GAP, lista de setups pairwise não entra; usamos setup_data/pairs_not_setup
        else:
            # timeindex mantém matriz de tempos de setup como arcos (opcional ao seu solver)
            # Se você quiser exatamente como o JobSchedulingInput original,
            # mantenha "setups" criando slots a partir de setup_times_df (já estava no seu código).
            setups = self.build_setups(machines, setup_df, setup_times_df, pd.DataFrame(jobs))

        # Cabeçalho identico ao original
        count_time_slots = len(business_days) * self.cfg.slots_per_day
        payload = {
            "count_time_slots": count_time_slots,
            "count_jobs": len(jobs),
            "time_step": self.cfg.time_step,
            "slots_per_day": self.cfg.slots_per_day,
            "init_date": business_days[0].strftime("%Y-%m-%d %H:%M"),
            "final_date": business_days[-1].date().isoformat(),
            "n_days": len(business_days),
            "machines": machines,
            "jobs": jobs,
            "setups": setups,
            "setup_data": setup_data,
            "pairs_not_setup": pairs_not_setup,
        }
        return payload


# -----------------------------------------------------------------------------
# Leitura e preparo de dados
# -----------------------------------------------------------------------------

def load_demand(base_data: Path) -> pd.DataFrame:
    """
    Carrega demandas de todos os arquivos Parquet na subpasta `demanda` dentro de `base_data`.
    """
    demand_path = base_data / "demanda"

    # Garantir que o diretório existe
    ensure_exists(demand_path, kind="dir")

    # Buscar todos os arquivos Parquet na pasta
    parquet_files = list(demand_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Nenhum arquivo Parquet encontrado em: {demand_path}")

    # Carregar e concatenar todos os arquivos Parquet
    df_list = []
    for file in parquet_files:
        logger.info("Carregando arquivo: %s", file)
        df_list.append(pd.read_parquet(file, engine="pyarrow"))

    # Concatenar todos os DataFrames
    df = pd.concat(df_list, ignore_index=True)

    # Renomear e ajustar colunas para formatos canônicos
    def _canonical_key(name: str) -> str:
        return re.sub(r"[^0-9a-z]+", "", name.lower())

    canonical_columns = {
        _canonical_key("Not_Before_Date"): "not_before_date",
        _canonical_key("DeadLine"): "deadline",
        _canonical_key("tempo_total"): "Tempo Total (minutos)",  # compat.
        _canonical_key("StatusIntegrationId"): "status_integration_id",
        _canonical_key("statusintegrationid"): "status_integration_id",
        _canonical_key("statusIntegrationId"): "status_integration_id",
        _canonical_key("dataHoraProcess"): "dataHoraProcess",
        _canonical_key("datahoraprocess"): "dataHoraProcess",
        _canonical_key("_kf_FichaTecnica"): "_kf_FichaTecnica",
        _canonical_key("_kf_fichatecnica"): "_kf_FichaTecnica",
        #  todas as variações conhecidas de _kf_macho
        _canonical_key("_kf_Macho"): "_kf_macho",
        _canonical_key("_Kf_Macho"): "_kf_macho",
        _canonical_key("_KfMacho"): "_kf_macho",
        _canonical_key("_kf_macho"): "_kf_macho",
        _canonical_key("kf_macho"): "_kf_macho",
        _canonical_key("macho"): "_kf_macho",
    }

    # Criar mapa de renomeação
    rename_map = {}
    for column in df.columns:
        key = _canonical_key(column)
        if key in canonical_columns:
            rename_map[column] = canonical_columns[key]

    # Aplicar renomeação
    df = df.rename(columns=rename_map)

    # 🔧 Garantir que a coluna _kf_macho exista
    if "_KfMacho" in df.columns:
        df.rename(columns={"_KfMacho": "_kf_macho"}, inplace=True)
    elif "_Kf_Macho" in df.columns:
        df.rename(columns={"_Kf_Macho": "_kf_macho"}, inplace=True)
    elif "KfMacho" in df.columns:
        df.rename(columns={"KfMacho": "_kf_macho"}, inplace=True)
    elif "Kf_Macho" in df.columns:
        df.rename(columns={"Kf_Macho": "_kf_macho"}, inplace=True)

    print(" Colunas após rename:", list(df.columns))

    if "status_integration_id" not in df.columns:
        df["status_integration_id"] = pd.Series(pd.NA, dtype="string", index=df.index)

    df = df.assign(
        not_before_date=lambda d: pd.to_datetime(d["not_before_date"], errors="coerce"),
        deadline=lambda d: pd.to_datetime(d["deadline"], errors="coerce"),
        processo=lambda d: d["processo"].astype("string").map(normalize_string),
        recurso=lambda d: d["recurso"].astype("string").map(normalize_string),
        status_integration_id=lambda d: d["status_integration_id"].astype("string"),
    )
    df["machine_name"] = df["processo"] + "_" + df["recurso"]
    df["lateness"] = 0
    df["Status_Processed"] = ""

    logger.info("Demandas (operações) carregadas: %d", df.shape[0])
    return df



def validate_and_fix_demand(
    df: pd.DataFrame, out_dir: Path, day_start: pd.Timestamp
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) not_before_date > deadline -> ajusta not_before_date para (deadline - 2 dias)
    mask_invalida = df["not_before_date"] > df["deadline"]
    if mask_invalida.any():
        logger.warning("%d caixas com not_before_date > deadline.", mask_invalida.sum())
        df.loc[mask_invalida, "Status_Processed"] = "Erro Not before date > deadline"
    # 2) faltar colunas essenciais -> remove
    required_cols = [
        "not_before_date",
        "_kf_FichaTecnica",
        "caixa",
        "processo",
        "recurso",
        "Tempo Total (minutos)",
        "deadline",
        "_kf_macho",
        "_kf_producao",
        "qtd_moldes",
    ]
    mask_missing = df[required_cols].isnull().any(axis=1)
    if mask_missing.any():
        logger.warning("Jobs com dados faltantes removidos: %d", mask_missing.sum())
        df.loc[mask_missing, "Status_Processed"] = "ERRO dado faltando"

    # 3) deadline < DAY_START -> ajusta e marca lateness
    mask_late = df["deadline"] < day_start
    if mask_late.any():
        df.loc[mask_late, "deadline"] = day_start
        df.loc[mask_late, "lateness"] = 1
        logger.info("Deadlines anteriores ao início ajustados: %d", mask_late.sum())

    return df


def load_aux_tables(
    base_data: Path,
    model_config_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega os arquivos de parâmetros a partir de ``model_config_dir`` ou ``base_data/parameters``."""
    if model_config_dir is not None:
        parameters_path = model_config_dir
    else:
        parameters_path = base_data / "parameters"

    ensure_exists(parameters_path, kind="dir")

    # Definir os caminhos dos arquivos
    machines_csv = parameters_path / "brut_machine_information.csv"
    cap_csv = parameters_path / "brut_recurso_time_capacity.csv"
    shifts_csv = parameters_path / "brut_shifts.csv"
    setup_csv = parameters_path / "setup.csv"
    setup_times_csv = parameters_path / "setup_times.csv"

    # Garantir que os arquivos existem
    for f in (machines_csv, cap_csv, shifts_csv, setup_csv, setup_times_csv):
        ensure_exists(f)

    # Carregar os arquivos CSV
    machines = pd.read_csv(
        machines_csv,
        converters={
            "turnos": parse_turnos_to_list,
            "processo": normalize_string,
            "recurso": normalize_string,
        },
    )
    caps = pd.read_csv(cap_csv, converters={"recurso": normalize_string})
    shifts = pd.read_csv(shifts_csv, converters={"turno": parse_turno_to_int})
    setup = pd.read_csv(setup_csv)
    setup_times = pd.read_csv(setup_times_csv)

    return machines, caps, shifts, setup, setup_times


def compute_business_days(day_start: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Gera dias úteis (seg-sex) a partir de day_start até a sexta-feira da mesma semana.
    - Se day_start for sábado/domingo, começa na 2ª feira seguinte.
    - Inclui o próprio dia_start se ele for dia útil.
    """
    day_start = pd.to_datetime(day_start)
    wd = day_start.weekday()  # 0=Mon ... 6=Sun

    start = day_start
    # Se já for sexta (4), a sexta "da mesma semana" é o próprio dia
    if wd == 4:
        end = start
    else:
        end = start + pd.offsets.Week(weekday=4)
    bday = CustomBusinessDay(weekmask="Mon Tue Wed Thu Fri")
    return pd.date_range(start=start, end=end, freq=bday, inclusive="both")


# -----------------------------------------------------------------------------
# Jobs
# -----------------------------------------------------------------------------


def filter_demand_by_horizon(
    demand_df: pd.DataFrame, business_days: pd.DatetimeIndex
) -> pd.DataFrame:
    last_day = business_days[-1].normalize()
    mask = demand_df["not_before_date"].dt.normalize() > last_day
    if mask.any():
        logger.info(
            "Removendo %d jobs com not_before_date > %s",
            mask.sum(),
            last_day.date().isoformat(),
        )
        demand_df.loc[mask, "Status_Processed"] = "Erro not before posterior a sexta"

    return demand_df


# -----------------------------------------------------------------------------
# Orquestração
# -----------------------------------------------------------------------------
def run(paths: IOPaths, cfg: RunConfig) -> None:
    # --- Carregar dados principais ---
    
    demand_df = load_demand(paths.base_data)
    demand_df = validate_and_fix_demand(demand_df, paths.output_dir, cfg.day_start)
    business_days = compute_business_days(cfg.day_start)
    demand_df = filter_demand_by_horizon(demand_df, business_days)
    demand_df = demand_df.reset_index(drop=True)
    demand_df["job_id"] = np.arange(len(demand_df), dtype=int)

    # Limitar a demanda aos 10 primeiros jobs
    #demand_df = demand_df.head(10).reset_index(drop=True)
    demand_df["job_id"] = np.arange(len(demand_df), dtype=int)
    logger.info("Demanda reduzida para %d jobs (teste)", len(demand_df))


    machines_df, caps_df, shifts_df, setup_df, setup_times_df = load_aux_tables(
        paths.base_data, paths.model_config_dir
    )
    


    # --- Builder unificado ---
    builder = SchedulingInputBuilder(cfg)

    gap_data = builder.create_input(
        demand_df, machines_df, caps_df, shifts_df,
        business_days, setup_df, setup_times_df, mode="gap"
    )
    time_data = builder.create_input(
        demand_df, machines_df, caps_df, shifts_df,
        business_days, setup_df, setup_times_df, mode="timeindex"
    )

    # Caminho de saída
    unified_path = paths.output_dir / "macharia_input.json"

    unify_jsons(gap_data, time_data, unified_path)

    



# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


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
    parser.add_argument(
        "--day-start",
        type=str,
        default=None,
        help="Data/hora de início do horizonte. Se omitido, utiliza dt às 00:00.",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=DEFAULT_TIME_STEP,
        help=f"Tamanho do slot em minutos (default: {DEFAULT_TIME_STEP}).",
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

    day_start_value = args.day_start or f"{args.dt} 00:00"
    day_start_ts = pd.to_datetime(day_start_value)
    time_step = int(args.time_step)
    slots_per_day = (24 * 60) // time_step
    cfg = RunConfig(
        day_start=day_start_ts, time_step=time_step, slots_per_day=slots_per_day
    )

    logger.info(
        "Processando dt=%s | lotes=%d | time_step=%d | slots/dia=%d",
        args.dt,
        len(status_to_process),
        cfg.time_step,
        cfg.slots_per_day,
    )

    for status_label, dataset_dir in status_to_process:
        output_dir = trusted_root / date_slug / status_label
        paths = IOPaths.build(
            base_data=dataset_dir,
            output_dir=output_dir,
            model_config_dir=model_config_dir,
        )

        logger.info(
            "Processando lote status=%s | base=%s",
            status_label,
            dataset_dir,
        )

        run(paths, cfg)


def unify_jsons(gap_data: dict, time_data: dict, out_path: Path) -> Path:
    """Une dados de GAP e Time-Index (em memória) e salva o JSON final unificado."""
    # ========================
    #  UNIFICAÇÃO DE JOBS
    # ========================
    gap_jobs = {j["job_id"]: j for j in gap_data.get("jobs", [])}
    unified_jobs = []
    for job in time_data.get("jobs", []):
        job_id = job.get("job_id")
        gap_job = gap_jobs.get(job_id, {})

        job["gap"] = {
            "release_date_index": gap_job.get("release_date_index"),
            "deadline_date_index": gap_job.get("deadline_date_index"),
            "processing_minutes": gap_job.get("processing_minutes"),
            "config": gap_job.get("config"),
        }
        job["time-index"] = {
            "release_date_slot": job.get("release_date_slot"),
            "deadline_slot": job.get("deadline_slot"),
            "processing_slots": job.get("processing_slots"),
        }
        unified_jobs.append(job)

    # =========================
    #  UNIFICAÇÃO DE MÁQUINAS
    # =========================
    gap_machines = {m["machine_id"]: m for m in gap_data.get("machines", [])}
    unified_machines = []
    for machine in time_data.get("machines", []):
        machine_id = machine.get("machine_id")
        gap_machine = gap_machines.get(machine_id, {})
        machine["gap"] = {"work_time_per_day": gap_machine.get("work_time_per_day")}
        machine["time-index"] = {"time_capacity": machine.get("time_capacity")}
        unified_machines.append(machine)

    setup_data = gap_data.get("setup_data", [])
    pairs_not_setup = gap_data.get("pairs_not_setup", [])

    machine_fields_to_keep = [
        "machine_id", "machine_name", "start_slots", "job_capacity", "gap", "time-index"
    ]
    job_fields_to_keep = [
        "job_id", "op", "_kf_macho", "kp_fichaTecnica", "job_register_id",
        "assigned_machine_id", "gap", "time-index", "status_integration_id",
        "Status_Processed", "has_lateness","enhance_processing_time"
    ]

    filtered_machines = [
        {k: v for k, v in m.items() if k in machine_fields_to_keep}
        for m in unified_machines
    ]
    filtered_jobs = [
        {k: v for k, v in j.items() if k in job_fields_to_keep}
        for j in unified_jobs
    ]

    unified_payload = time_data.copy()
    unified_payload["jobs"] = filtered_jobs
    unified_payload["machines"] = filtered_machines
    unified_payload["setup_data"] = setup_data
    unified_payload["pairs_not_setup"] = pairs_not_setup

    class PrettyEncoder(json.JSONEncoder):
        def iterencode(self, o, _one_shot=False):
            for s in super().iterencode(o, _one_shot=_one_shot):
                s = s.replace("], [", "],\n          [")
                yield s

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(unified_payload, f, cls=PrettyEncoder, indent=2, ensure_ascii=False)

    logger.info("Gerado arquivo unificado com sucesso: %s", out_path)
    return out_path



if __name__ == "__main__":
    main()
