import json
import os
import time
import math
import pandas as pd
import numpy as np
from pathlib import Path
from pyomo.environ import (
    ConcreteModel,
    Param,
    Var,
    Binary,
    NonNegativeReals,
    Set,
    Constraint,
    ConstraintList,
    Objective,
    minimize,
    value,
)
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
from collections import deque
from pyomo.opt import SolverFactory
from collections import defaultdict
import logging
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# optimize.py
# ---------------------------------------------------------------
# Script único, lendo APENAS macharia.json e rodando GAP + Scheduling
# com formulações mais enxutas (menos variáveis e restrições).
# ---------------------------------------------------------------

import json
import os
import time
import math
import logging
import argparse
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

from pyomo.environ import (
    ConcreteModel, Param, Var, Binary, NonNegativeReals, Set,
    Constraint, ConstraintList, Objective, minimize, value
)
from pyomo.opt import SolverFactory
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent


# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("optimize")

TEE = True
CONFIG_PHASE1 = {"TimeLimit": 1200, "MIPGap": 0.00, "MIPFocus": 1}
CONFIG_PHASE2 = {"TimeLimit": 2400, "MIPGap": 0.01, "MIPFocus": 1}


# -------------------- Utils --------------------
def ensure_exists(path: Path, kind: str = "file") -> None:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {path}")


def ensure_model_config_directory(
    model_config_dir: Path, possible_sources: Iterable[Path]
) -> Path:
    model_config_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in model_config_dir.iterdir()} if model_config_dir.exists() else set()
    for src in possible_sources:
        if not src or not src.exists():
            continue
        for item in src.iterdir():
            dest = model_config_dir / item.name
            if dest.name in existing:
                continue
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
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


# ==========================================================
#                     GAP (capacidade = 1)
# ==========================================================
class GapOptimizer:
    
    """Modelo diário (lot-sizing) com setups por configuração (A/B).
       Menos variáveis: sem TRIPLES/q quando não existirem; z restrito a pares elegíveis."""

    def _build_setup_index(self, setup_data):
        info = {}
        for s in setup_data:
            info[(s["from_config"], s["to_config"], s["machine_id"])] = int(s["setup_time"])
        return info

    def solve_model(
        self,
        machine: dict,
        machine_jobs: List[dict],
        pairs_not_setup_data: List[dict],
        setup_info: Dict[Tuple[str, str, int], int],
        days: List[int],
        output_dir: Path,
    ):
        bigM = 10**9
        machine_id = machine["machine_id"]
        machine_name = machine["machine_name"]

        if not machine_jobs:
            return ({}, [], None)

        # Normalização leve
        jobs_info = {j["job_id"]: j for j in machine_jobs}
        machine_job_ids = set(jobs_info.keys())

        # Pares sem setup (filtrados para esta máquina e existindo nos jobs)
        current_pairs_not_setup = {
            (p["job_i"], p["job_j"])
            for p in pairs_not_setup_data
            if p["machine_id"] == machine_id
            and p["job_i"] in machine_job_ids
            and p["job_j"] in machine_job_ids
        }

        # Parâmetro s: diferença de setup A->B menos A->A (só quando aplicável)
        s = 0
        if ("A", "B", machine_id) in setup_info and ("A", "A", machine_id) in setup_info:
            s = setup_info[("A", "B", machine_id)] - setup_info[("A", "A", machine_id)]

        # Capacidade por dia
        work_time_per_day = (
            machine.get("work_time_per_day")
            or machine.get("gap", {}).get("work_time_per_day")
        )
        if not work_time_per_day:
            raise KeyError(f"Máquina {machine_name} sem 'work_time_per_day'.")

        # Horizonte real + dia fictício
        days_real = sorted(map(int, work_time_per_day.keys()))
        fict_day = (days_real[-1] + 1) if days_real else 1
        days = list(days_real) + [fict_day]

        max_work = {int(d): work_time_per_day[str(d)]["max"] for d in days_real}
        max_work[fict_day] = bigM

        # ----------------- Modelo -----------------
        m = ConcreteModel()
        m.JOBS = Set(initialize=list(jobs_info.keys()))
        m.DAYS = Set(initialize=days)
        m.JOB_DAY = Set(dimen=2, initialize=[(i, d) for i in m.JOBS for d in m.DAYS])
        m.PHI = Set(dimen=2, initialize=list(current_pairs_not_setup))

        # Params
        def _deadline_init(i):
            return int(
                jobs_info[i].get("deadline_date_index")
                or jobs_info[i].get("time-index", {}).get("deadline_slot")
                or 0
            )

        def _release_init(i):
            return int(
                jobs_info[i].get("release_date_index")
                or jobs_info[i].get("time-index", {}).get("release_date_slot")
                or 0
            )

        def _ptime_init(i):
            # Prioriza minutos (GAP), senão slots (fallback)
            v = (
                jobs_info[i].get("processing_minutes")
                or jobs_info[i].get("gap", {}).get("processing_minutes")
                or jobs_info[i].get("time-index", {}).get("processing_slots")
                or 0
            )
            return int(v)

        m.DEADLINE = Param(m.JOBS, initialize={i: _deadline_init(i) for i in m.JOBS})
        m.RELEASE = Param(m.JOBS, initialize={i: _release_init(i) for i in m.JOBS})
        m.PTIME = Param(m.JOBS, initialize={i: _ptime_init(i) for i in m.JOBS})
        m.MAXW = Param(m.DAYS, initialize=max_work, within=NonNegativeReals)

        # Penalidade por atraso (exponencial após deadline)
        eps, gamma, base = 1e-3, 2.0, 2.0
        c_init = {}
        for i in m.JOBS:
            dline = m.DEADLINE[i]
            for d in m.DAYS:
                if d == fict_day:
                    c_init[(i, d)] = bigM
                elif d <= dline:
                    c_init[(i, d)] = eps * d
                else:
                    c_init[(i, d)] = base * (gamma ** (d - dline))
        m.C = Param(m.JOB_DAY, initialize=c_init, within=NonNegativeReals)

        # Vars (enxutas)
        m.x = Var(m.JOB_DAY, domain=Binary)   # alocação de job i no dia d
        m.a = Var(m.DAYS, domain=Binary)      # dia com config A
        m.b = Var(m.DAYS, domain=Binary)      # dia com config B
        m.g = Var(m.DAYS, domain=Binary)      # 1 se produz A e B no mesmo dia
        m.z = Var(m.PHI, m.DAYS, domain=Binary)  # ativa par (i,j) no dia d (sem setup entre si)

        # Cons
        m.one_start = ConstraintList()
        for i in m.JOBS:
            expr = sum(m.x[i, d] for d in m.DAYS if d >= m.RELEASE[i])
            m.one_start.add(expr == 1)

        # ativa a/b/g
        m.activate_a = ConstraintList()
        m.activate_b = ConstraintList()
        m.activate_g = ConstraintList()
        for d in m.DAYS:
            for i in m.JOBS:
                cfg = jobs_info[i].get("config") or jobs_info[i].get("gap", {}).get("config")
                if cfg == "A" and d >= m.RELEASE[i]:
                    m.activate_a.add(m.a[d] >= m.x[i, d])
                if cfg == "B" and d >= m.RELEASE[i]:
                    m.activate_b.add(m.b[d] >= m.x[i, d])
            m.activate_g.add(m.g[d] >= m.a[d] + m.b[d] - 1)

        # Capacidade diária com benefício dos pares sem setup
        m.max_use_day = ConstraintList()
        for d in m.DAYS:
            ptime = []
            add_pairs = []
            for i in m.JOBS:
                if d >= m.RELEASE[i]:
                    ptime.append(m.PTIME[i] * m.x[i, d])

            # só adiciona z se ambos elegíveis no dia
            for (i, j) in m.PHI:
                cfg_i = jobs_info[i].get("config") or jobs_info[i].get("gap", {}).get("config")
                cfg_j = jobs_info[j].get("config") or jobs_info[j].get("gap", {}).get("config")
                setup_ij = setup_info.get((cfg_i, cfg_j, machine_id), 0)
                if setup_ij > 0 and d >= max(m.RELEASE[i], m.RELEASE[j]):
                    add_pairs.append(setup_ij * m.z[i, j, d])

                # limita quantidade de pares por job no dia (no máximo 1)
                # (aplica uma vez por i)
            # limitador leve por job no dia
            # (se não houver PHI, nada é adicionado)
            for i in set([ii for (ii, _) in m.PHI] + [jj for (_, jj) in m.PHI]):
                z_terms = [m.z[ii, jj, d] for (ii, jj) in m.PHI if i in (ii, jj)]
                if z_terms:
                    m.max_use_day.add(sum(z_terms) <= 1)

            # capacidade
            m.max_use_day.add(
                sum(ptime) <= m.MAXW[d] - s * m.g[d] + sum(add_pairs)
            )

        # FO
        m.total_penalty = Objective(
            expr=sum(m.C[i, d] * m.x[i, d] for (i, d) in m.JOB_DAY),
            sense=minimize,
        )

        solver = SolverFactory("highs")
        solver.options["TimeLimit"] = 1200
        result = solver.solve(m, tee=TEE)

        # Métricas HiGHS
        try:
            machine_capacity = {m["machine_name"]: m.get("job_capacity", 1) for m in data["machines"]}
            
            metrics = {
                "model_label": f"gap_{machine_name}",
                "solver": "highs",
                "objective_value": float(value(m.total_penalty)),
                "num_variables": sum(1 for _ in m.component_objects(Var, active=True)),
                "num_constraints": sum(1 for _ in m.component_objects(Constraint, active=True)),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            metrics_all.append(metrics)
        except Exception as e:
            logger.warning(f"⚠️ Falha ao salvar métricas HiGHS: {e}")



        # Coleta
        jobs_on_machine, not_sat = [], []
        for job in machine_jobs:
            j = job["job_id"]
            alloc = None
            for d in days:
                if value(m.x[j, d]) > 0.5:
                    alloc = d
                    break

            if alloc is not None and alloc != fict_day:
                jobs_on_machine.append(
                    {
                        "job_id": j,
                        "job_register_id": job.get("job_register_id"),
                        "kp_fichaTecnica": job.get("kp_fichaTecnica"),
                        "op": job.get("op"),
                        "_kf_macho": job.get("_kf_macho"),
                        "release_date_index": job.get("release_date_index"),
                        "deadline_date_index": job.get("deadline_date_index"),
                        "config": job.get("config") or job.get("gap", {}).get("config"),
                        "assigned_day": alloc,
                        "processing_minutes": int(
                            job.get("processing_minutes")
                            or job.get("gap", {}).get("processing_minutes")
                            or 0
                        ),
                        "machine_id": job["assigned_machine_id"],
                        "has_lateness": job.get("has_lateness"),
                        "release_date_index": int(job["release_date_slot"] // 288),   # converte slots → dia
                        "deadline_date_index": int(job["deadline_slot"] // 288)+1,

                        #"release_date_index": job.get("release_date_slot"),
                        #"deadline_index": job.get("deadline_slot") - 1,


                    }
                )
            else:
                not_sat.append(
                    {
                        "job_id": j,
                        "job_register_id": job.get("job_register_id"),
                        "op": job.get("op"),
                        "_kf_macho": job.get("_kf_macho"),
                        "Status_Processed": "Nao foi possivel processar o registro",
                    }
                )

        machine_sched = {
            "machine_id": machine_id,
            "machine_name": machine_name,
            "jobs": jobs_on_machine,
        }
        return (machine_sched, not_sat, result)

    def run(self, data: dict, output_dir: Path):
        metrics_all = []

        jobs = [j for j in data["jobs"] if not j.get("Status_Processed")]
        setup_data = data.get("setup_data", [])
        pairs_not_setup = data.get("pairs_not_setup", [])
        setup_info = self._build_setup_index(setup_data)
        machines = data["machines"]

        optimization_summaries = {}
        machines_scheduling = []
        not_satisfied = []

        t0 = time.time()
        for machine in machines:

            if machine.get("job_capacity", 1) > 1:
                # GAP só para máquinas de capacidade 1
                continue

            start = time.time()
            sched, not_ok, result = self.solve_model(
                machine=machine,
                machine_jobs=[j for j in jobs if j["assigned_machine_id"] == machine["machine_id"]],
                pairs_not_setup_data=pairs_not_setup,
                setup_info=setup_info,
                days=[],  # calculado internamente pelos "work_time_per_day"
                output_dir=output_dir,
            )

            if result:
                machines_scheduling.append(sched)
                not_satisfied += not_ok
                optimization_summaries[machine["machine_id"]] = {
                    "count_jobs_missed": len(not_ok),
                    "machine_name": machine["machine_name"],
                    "status": str(result.solver.status),
                    "elapsed_time_seconds": round(time.time() - start, 2),
                    "gap": getattr(result.solver, "gap", None),
                }

        out = {
            "optimization_summary": optimization_summaries,
            "machines_scheduling": machines_scheduling,
            "not_satisfied_jobs": not_satisfied,
        }

        logger.info("Tempo total GAP: %.2f min", (time.time() - t0) / 60.0)
        # Não salva arquivo separado — retorno ao chamador
        if metrics_all:
            solver_name = "highs" if isinstance(self, GapOptimizer) else "gurobi_persistent"
            metrics_file = output_dir / f"model_metrics_{solver_name}.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics_all, f, indent=2, ensure_ascii=False)
            logger.info(f" Métricas {solver_name} consolidadas em {metrics_file}")

        return out

# ==========================================================
#      JOB SCHEDULING (capacidade > 1) — idêntico ao modelo detalhado
# ==========================================================
class JobSchedulingOptimizer:
    """
    Modelo por slots com múltiplas submáquinas (capacidade > 1),
    com MESMAS variáveis, restrições e FO do modelo detalhado:
      - x[J, s, k]  binária (job J inicia no slot s na submáquina k)
      - z[r, k]     binária (submáquina k ativa no range de slots r)
      - y[r, k, g]  binária (grupo g=(kf_macho, caixa) ativo no range r na sub k)
      - slack_min_use_range[r, k] >= 0 (folga para capacidade mínima)
      - eps[J]      binária (job não alocado, para fase 1)
    Restrições:
      - one_start, parallel, unique_kf_macho_caixa, slot_range_submachine_capacity,
        activate_use_machine, z_order, setup_conflict (two-pointers).
    FO faseada com GurobiPersistent (igual ao detalhado).
    """

    @staticmethod
    def get_slots_used(idx_of: Dict[int, int], start_slots: List[int], s: int, p: int):
        sidx = idx_of.get(s)
        eidx = min(sidx + int(p), len(start_slots))
        return start_slots[sidx:eidx]

    @staticmethod
    def calculate_upperbound_T(jobs, slots_per_day, max_cap_day, start_slots, sub_count):
        total_p = sum(int(j.get("processing_slots") or j.get("time-index", {}).get("processing_slots") or 0) for j in jobs)
        last_rel = max(int(j.get("release_date_slot") or j.get("time-index", {}).get("release_date_slot") or 0) for j in jobs) if jobs else 0
        if max_cap_day and sub_count:
            days_need = math.ceil(total_p / (max_cap_day * sub_count))
        else:
            days_need = 0
        cap_end = days_need * slots_per_day - 1 if days_need > 0 else 0
        last_rel_day_end = ((last_rel // slots_per_day) + 1) * slots_per_day - 1
        ub = max(cap_end, last_rel_day_end, last_rel)
        return min(ub, start_slots[-1] if start_slots else ub)

    def solve_machine(
        self,
        machine: dict,
        machine_to_jobs: Dict[int, List[dict]],
        setups: List[dict],
        slots_per_day: int,
        time_index_of: Dict[int, Dict[int, int]],
        count_time_slots: int,
        output_dir: Path,
    ):
        mid = int(machine["machine_id"])
        jobs_all = [j for j in machine_to_jobs.get(mid, []) if not j.get("Status_Processed")]
        sub_count = int(machine.get("job_capacity", 1))

        

        # Scheduling só para máquinas com múltiplas submáquinas
        if not jobs_all or sub_count <= 1:
            return ({}, [], {})

        # Garante time_capacity
        if "time_capacity" not in machine:
            ti = machine.get("time-index", {})
            if isinstance(ti, dict) and "time_capacity" in ti:
                machine["time_capacity"] = ti["time_capacity"]
            else:
                machine["time_capacity"] = {"slots": [[0, slots_per_day]], "slots_capacity": [[0, slots_per_day]]}

        start_slots = list(machine.get("start_slots") or machine.get("time-index", {}).get("start_slots", []))
        idx_of = time_index_of[mid]

        # Upper bound do horizonte
        first_cap = machine["time_capacity"]["slots_capacity"][0] if machine["time_capacity"].get("slots_capacity") else [0, slots_per_day]
        max_cap_day = int(first_cap[1]) if isinstance(first_cap, (list, tuple)) and len(first_cap) >= 2 else slots_per_day
        ub = self.calculate_upperbound_T(jobs_all, slots_per_day, max_cap_day, start_slots, sub_count)
        start_slots = [s for s in start_slots if s <= ub]

        # === Grupos por (kf_macho, caixa) ===
        
        jobs_by_group = defaultdict(list)
        for j in jobs_all:
            g_key = (j.get("_kf_macho"), j.get("caixa"))
            jobs_by_group[g_key].append(j)
        groups = list(jobs_by_group.keys())

        # Conjunto de starts válidos por job (respeitando release/deadline e não atravessar o dia)
        job_to_slots = defaultdict(list)
        for job in jobs_all:
            p = int(job.get("processing_slots") or job.get("time-index", {}).get("processing_slots") or 0)
            rel = int(job.get("release_date_slot") or job.get("time-index", {}).get("release_date_slot") or 0)
            dead = int(job.get("deadline_slot") or job.get("time-index", {}).get("deadline_slot") or count_time_slots)
            dead_day_end = dead + slots_per_day - 1  # fim inclusivo do dia da deadline

            for s in start_slots:
                si = idx_of.get(s)
                if si is None:
                    continue
                ei = si + p - 1
                if ei >= len(start_slots):
                    continue
                used = self.get_slots_used(idx_of, start_slots, s, p)
                if not used:
                    continue
                last = used[-1]
                cur_day_end = ((s // slots_per_day) + 1) * slots_per_day - 1
                if s >= rel and last <= min(cur_day_end, dead_day_end):
                    job_to_slots[job["job_id"]].append(s)

        # Pré-computa used-slots
        jobslot_to_used = {}
        for j in jobs_all:
            p = int(j.get("processing_slots") or j.get("time-index", {}).get("processing_slots") or 0)
            for s in job_to_slots.get(j["job_id"], []):
                jobslot_to_used[(j["job_id"], s)] = self.get_slots_used(idx_of, start_slots, s, p)

        # Ranges ATIVOS (capacidade > 0)
        time_capacity = machine.get("time_capacity", {})
        active_ranges = []
        for slot_range, cap in zip(time_capacity.get("slots", []), time_capacity.get("slots_capacity", [])):
            rs, re = int(slot_range[0]), int(slot_range[1])
            rmin, rmax = int(cap[0]), int(cap[1])
            if rmax and re > rs:
                active_ranges.append((rs, re, rmin, rmax))

        # Setups (somente desta máquina com S_ij > 0)
        setups_machine = [s for s in setups if int(s.get("machine_id", -1)) == mid and int(s.get("setup_time", 0)) > 0]
        jobs_dict = {j["job_id"]: j for j in jobs_all}

        # ----------------- Modelo -----------------
        m = ConcreteModel()
        m.JOBS = Set(initialize=[j["job_id"] for j in jobs_all])
        m.TIME = Set(initialize=start_slots)
        m.SUB_MACHINES = Set(initialize=range(sub_count))
        m.JOB_SLOTS_K = Set(
            dimen=3,
            initialize=[(j["job_id"], s, k) for j in jobs_all for s in job_to_slots.get(j["job_id"], []) for k in m.SUB_MACHINES]
        )
        m.slot_ranges = Set(dimen=2, initialize=[(a, b) for (a, b, _, _) in active_ranges])
        m.GROUPS = Set(initialize=groups)  # grupos (kf_macho, caixa)

        # Variáveis (idênticas)
        m.x = Var(m.JOB_SLOTS_K, domain=Binary)                  # job inicia no slot s da sub k
        m.z = Var(m.slot_ranges, m.SUB_MACHINES, domain=Binary)  # submáquina k ativa no range r
        m.y = Var(m.slot_ranges, m.SUB_MACHINES, m.GROUPS, domain=Binary)  # grupo g ativo no range r na sub k
        m.slack_min_use_range = Var(m.slot_ranges, m.SUB_MACHINES, domain=NonNegativeReals)
        m.eps = Var(m.JOBS, domain=Binary)  # fallback p/ job não alocado (fase 1)

        # one_start (com eps)
        m.one_start = ConstraintList()
        for j in jobs_all:
            J = j["job_id"]
            valid = job_to_slots.get(J, [])
            if valid:
                m.one_start.add(sum(m.x[J, s, k] for s in valid for k in m.SUB_MACHINES) + m.eps[J] == 1)
            else:
                m.one_start.add(m.eps[J] == 1)

        # Capacidade por slot e submáquina (paralelismo)
        m.parallel = ConstraintList()
        for t in start_slots:
            for k in m.SUB_MACHINES:
                terms = []
                for j in jobs_all:
                    J = j["job_id"]
                    for s in job_to_slots.get(J, []):
                        used = jobslot_to_used[(J, s)]
                        if t in used:
                            terms.append(m.x[J, s, k])
                if terms:
                    m.parallel.add(sum(terms) <= 1)

        # === Unicidade por (kf_macho, caixa) via y[r,k,g] ===
        # 1) sum_k y[r,k,g] <= 1   (grupo g em uma única sub por range r)
        m.unique_kf_macho_caixa = ConstraintList()
        for r in m.slot_ranges:
            for g in m.GROUPS:
                m.unique_kf_macho_caixa.add(sum(m.y[r, k, g] for k in m.SUB_MACHINES) <= 1)

        # 2) x[j,s,k] => y[r,k,g] quando (j,s) sobrepõe o range r
        #    (computa g do job j)
        def _group_of(job):
            return (jobs_dict[job]["_kf_macho"], jobs_dict[job].get("caixa"))

        for (rs, re, _, _) in active_ranges:
            r = (rs, re)
            for k in m.SUB_MACHINES:
                for g, group_jobs in jobs_by_group.items():
                    for job in group_jobs:
                        J = job["job_id"]
                        for s in job_to_slots.get(J, []):
                            used = jobslot_to_used[(J, s)]
                            if any(rs <= ts < re for ts in used):
                                m.unique_kf_macho_caixa.add(m.x[J, s, k] <= m.y[r, k, g])

        # Capacidade por range (min/max) + slack_min_use_range (idêntico)
        m.slot_range_submachine_capacity = ConstraintList()
        for (rs, re, rmin, rmax) in active_ranges:
            r = (rs, re)
            for k in m.SUB_MACHINES:
                expr_terms = []
                for j in jobs_all:
                    J = j["job_id"]
                    for s in job_to_slots.get(J, []):
                        used = jobslot_to_used[(J, s)]
                        overlap = [ts for ts in used if rs <= ts < re]
                        if overlap:
                            expr_terms.append(len(overlap) * m.x[J, s, k])
                if expr_terms:
                    expr = sum(expr_terms)
                    m.slot_range_submachine_capacity.add(expr <= rmax)
                    m.slot_range_submachine_capacity.add(expr + m.slack_min_use_range[r, k] >= rmin)

        # Ativa z: x <= z quando (J,s) sobrepõe o range r
        m.activate_use_machine = ConstraintList()
        for (rs, re, _, _) in active_ranges:
            r = (rs, re)
            for k in m.SUB_MACHINES:
                for j in jobs_all:
                    J = j["job_id"]
                    for s in job_to_slots.get(J, []):
                        used = jobslot_to_used[(J, s)]
                        if any(rs <= ts < re for ts in used):
                            m.activate_use_machine.add(m.x[J, s, k] <= m.z[r, k])

        # Sequenciamento de submáquinas: z[r,k] <= z[r,k-1]
        m.z_order = ConstraintList()
        for r in m.slot_ranges:
            for k in range(1, sub_count):
                m.z_order.add(m.z[r, k] <= m.z[r, k - 1])

        # Setup conflict (two-pointers) — idêntico
        m.setup_conflict = ConstraintList()
        n = len(start_slots)
        def _sidx(x): return time_index_of[mid][x]

        for setup in setups_machine:
            iJ = setup["from_job_id"]
            jJ = setup["to_job_id"]
            Sij = int(setup.get("setup_time", 0))
            if iJ not in jobs_dict or jJ not in jobs_dict or Sij <= 0:
                continue
            Pi = int(jobs_dict[iJ].get("processing_slots") or 0)
            Pj = int(jobs_dict[jJ].get("processing_slots") or 0)
            Hi = job_to_slots.get(iJ, [])
            Hj = job_to_slots.get(jJ, [])
            if not Hi or not Hj:
                continue

            dur_i = Pi + Sij
            dur_j = Pj
            Hi_valid = [s for s in Hi if _sidx(s) + dur_i <= n]
            Hj_valid = [s for s in Hj if _sidx(s) + dur_j <= n]
            if not Hi_valid or not Hj_valid:
                continue

            last_i_idx = {s: _sidx(s) + dur_i - 1 for s in Hi_valid}
            last_j_idx = {s: _sidx(s) + dur_j - 1 for s in Hj_valid}
            rel_idx = sorted(set([_sidx(s) for s in Hi_valid] + list(last_i_idx.values())
                                 + [_sidx(s) for s in Hj_valid] + list(last_j_idx.values())))
            Hi_sorted = sorted(Hi_valid, key=_sidx)
            Hj_sorted = sorted(Hj_valid, key=_sidx)

            qi, qj = deque(), deque()
            pi = pj = 0
            prev_key = None
            for h in rel_idx:
                while pi < len(Hi_sorted) and _sidx(Hi_sorted[pi]) <= h:
                    s = Hi_sorted[pi]
                    if last_i_idx[s] >= h:
                        qi.append(s)
                    pi += 1
                while qi and last_i_idx[qi[0]] < h:
                    qi.popleft()

                while pj < len(Hj_sorted) and _sidx(Hj_sorted[pj]) <= h:
                    s = Hj_sorted[pj]
                    if last_j_idx[s] >= h:
                        qj.append(s)
                    pj += 1
                while qj and last_j_idx[qj[0]] < h:
                    qj.popleft()

                if not qi or not qj:
                    prev_key = None
                    continue

                wi = tuple(qi)
                wj = tuple(qj)
                key = (wi, wj)
                if key == prev_key:
                    continue
                prev_key = key

                for k in m.SUB_MACHINES:
                    m.setup_conflict.add(
                        sum(m.x[iJ, si, k] for si in wi) + sum(m.x[jJ, sj, k] for sj in wj) <= 1
                    )

        # ===== Objetivos (duas fases, idêntico) =====
        # Fase 1: minimizar jobs não alocados
        m.obj1 = Objective(expr=sum(m.eps[J] for J in m.JOBS), sense=minimize)

        opt = GurobiPersistent()
        opt.set_instance(m)
        opt.set_objective(m.obj1)
        opt.options.update(CONFIG_PHASE1)
        t0 = time.time()
        result = opt.solve(tee=TEE)
        miss = value(sum(m.eps[J] for J in m.JOBS))

        # Fixa miss
        m.obj1.deactivate()
        if hasattr(m, "obj1"):
            m.del_component(m.obj1)
        m.fix_miss = Constraint(expr=sum(m.eps[J] for J in m.JOBS) == miss)
        opt.add_constraint(m.fix_miss)

        # Fase 2: mesma FO ponderada do seu arquivo
        slot_day = {s: s // slots_per_day for s in start_slots}
        day_release = {
            j["job_id"]: int(j.get("release_date_slot") or j.get("time-index", {}).get("release_date_slot") or 0) // slots_per_day
            for j in jobs_all
        }

        alpha = 2.0
        w_k = {k: alpha ** k for k in m.SUB_MACHINES}
        B_start = len(jobs_all) * count_time_slots
        Zmax_day = (alpha ** sub_count - 1) / (alpha - 1) if sub_count > 1 else 1.0

        Wz = 1.0
        eps_w = min(1e-3, 1.0 / (B_start + 1))
        Wday = Wz * Zmax_day + eps_w * B_start + 1.0

        m.day_shift_sum = sum(
            (slot_day[s] - day_release[J]) * sum(m.x[J, s, k] for k in m.SUB_MACHINES)
            for J in m.JOBS for s in job_to_slots.get(J, [])
        )
        m.weighted_machine_use = sum(w_k[k] * m.z[r, k] for r in m.slot_ranges for k in m.SUB_MACHINES)
        m.sum_start_time = sum(
            s * sum(m.x[J, s, k] for k in m.SUB_MACHINES)
            for J in m.JOBS for s in job_to_slots.get(J, [])
        )

        m.obj2 = Objective(
            expr=Wday * m.day_shift_sum + Wz * m.weighted_machine_use + eps_w * m.sum_start_time,
            sense=minimize
        )
        opt.set_objective(m.obj2)
        opt.options.update(CONFIG_PHASE2)
        result = opt.solve(tee=TEE)

        elapsed = time.time() - t0
        mip_gap = getattr(opt._solver_model, "MIPGap", None)

        # Coleta
        jobs_on_machine, not_ok = [], []
        for j in jobs_all:
            J = j["job_id"]
            start_sel, sub_sel = None, None
            rel_slot = int(j.get("release_date_slot") or j.get("time-index", {}).get("release_date_slot") or 0)
            dead_slot = int(j.get("deadline_slot") or j.get("time-index", {}).get("deadline_slot") or 0)
            rel_idx = rel_slot // slots_per_day
            dead_idx = dead_slot // slots_per_day

            for s in job_to_slots.get(J, []):
                found = False
                for k in m.SUB_MACHINES:
                    if (J, s, k) in m.JOB_SLOTS_K and value(m.x[J, s, k]) > 0.5:
                        start_sel, sub_sel = s, k
                        found = True
                        break
                if found:
                    break

            if start_sel is not None:
                used = jobslot_to_used[(J, start_sel)]
                end_slot = used[-1] if used else start_sel
                jobs_on_machine.append({
                    "op": j.get("op"),
                    "job_register_id": j.get("job_register_id"),
                    "job_id": J,
                    "start": start_sel,
                    "end": end_slot,
                    "processing_slots": used,
                    "sub_machine": sub_sel,
                    "status_integration_id": j.get("status_integration_id"),
                    "Status_Processed": "",
                    "release_date_index": rel_idx,
                    "deadline_date_index": dead_idx
                })
            else:
                not_ok.append({
                    "op": j.get("op"),
                    "job_register_id": j.get("job_register_id"),
                    "job_id": J,
                    "start": None,
                    "end": None,
                    "processing_slots": None,
                    "sub_machine": None,
                    "status_integration_id": j.get("status_integration_id"),
                    "Status_Processed": "Nao foi possivel processar o registro",
                    "release_date_index": rel_idx,
                    "deadline_date_index": dead_idx
                })

        opt.close()
        del m, opt

        summary = {
            "machine_name": machine["machine_name"],
            "missed_jobs": miss,
            "status": str(result.solver.status),
            "elapsed_time_seconds": round(elapsed, 2),
            "elapsed_time_minutes": round(elapsed / 60.0, 2),
            "MIPGap": mip_gap,
        }

        return (jobs_on_machine, not_ok, summary)

    def run(self, data: dict, output_dir: Path):
        machines = data["machines"]
        jobs = [j for j in data["jobs"] if not j.get("Status_Processed")]
        setups = data.get("setups", [])
        slots_per_day = int(data.get("slots_per_day", 288))
        count_time_slots = int(data.get("count_time_slots", slots_per_day * 7))

        # Mapeia slots -> idx para cada máquina
        time_index_of = {}
        machine_to_jobs = defaultdict(list)
        for m in machines:
            mid = int(m["machine_id"])
            ss = m.get("start_slots") or m.get("time-index", {}).get("start_slots", [])
            time_index_of[mid] = {s: i for i, s in enumerate(ss)}
        for j in jobs:
            machine_to_jobs[int(j["assigned_machine_id"])].append(j)

        sched_machines, not_satisfied, summaries = [], [], {}
        t0 = time.time()
        for machine in machines:
            if int(machine.get("job_capacity", 1)) <= 1:
                continue  # só cap > 1 aqui

            jobs_on_machine, not_ok, summary = self.solve_machine(
                machine=machine,
                machine_to_jobs=machine_to_jobs,
                setups=setups,
                slots_per_day=slots_per_day,
                time_index_of=time_index_of,
                count_time_slots=count_time_slots,
                output_dir=output_dir,
            )
            if jobs_on_machine:
                sched_machines.append({
                    "machine_id": machine["machine_id"],
                    "machine_name": machine["machine_name"],
                    "jobs": jobs_on_machine,
                })
                not_satisfied.extend(not_ok)
                summaries[machine["machine_id"]] = summary

        out = {
            "optimization_summary": summaries,
            "machines_scheduling": sched_machines,
            "not_satisfied_jobs": not_satisfied,
        }

        logger.info("Tempo total Scheduling: %.2f min", (time.time() - t0) / 60.0)
        return out


# ==========================================================
#                    CLI / Main
# ==========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roda GAP (cap=1) e Job Scheduling (cap>1) lendo APENAS macharia.json."
    )
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--dt", required=True, help="Data (YYYY-MM-DD).")
    parser.add_argument(
        "--raw-root", type=Path, default=default_root.joinpath("data", "raw"),
        help="Diretório raiz com lotes importados (default: data/raw)."
    )
    parser.add_argument(
        "--trusted-root", type=Path, default=default_root.joinpath("data", "trusted"),
        help="Diretório raiz das saídas (default: data/trusted)."
    )
    parser.add_argument(
        "--model-config-dir", type=Path, default=default_root.joinpath("data", "raw", "model_config"),
        help="Diretório dos parâmetros do modelo (default: data/raw/model_config)."
    )
    parser.add_argument(
        "--only-status", nargs="+", default=None,
        help="Filtra pelos status (subpastas). Se omitido, processa todos."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        dt = datetime.strptime(args.dt, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Data inválida para --dt: {args.dt}") from exc

    date_slug = dt.strftime("%d%m%Y")
    raw_root: Path = args.raw_root
    trusted_root: Path = args.trusted_root
    model_config_dir: Path = args.model_config_dir
    date_dir = raw_root / date_slug

    logger.info("Base data dir: %s", raw_root)
    logger.info("Output dir: %s", trusted_root)

    ensure_exists(date_dir, kind="dir")

    status_dirs = [
        (p.name, p) for p in sorted(date_dir.iterdir())
        if p.is_dir() and (p / "demanda").is_dir()
    ]
    if not status_dirs:
        raise SystemExit(
            f"Nenhum lote encontrado em {date_dir}. Execute a ingestão antes."
        )

    if args.only_status:
        req = {s.strip().lower() for s in args.only_status if s.strip()}
        found = {name.lower(): path for name, path in status_dirs}
        missing = sorted(req - set(found.keys()))
        if missing:
            raise SystemExit("Status não encontrados: " + ", ".join(missing))
        status_to_process = [(name, found[name.lower()]) for name in args.only_status]
    else:
        status_to_process = status_dirs

    possible_param_srcs = [date_dir / "parameters", raw_root / "parameters"]
    ensure_model_config_directory(model_config_dir, possible_param_srcs)

    for status, dataset_dir in status_to_process:
        output_dir = trusted_root / date_slug / status
        output_dir.mkdir(parents=True, exist_ok=True)

        # === Lê APENAS macharia.json ===
        macharia_file = output_dir / "macharia_input.json"
        ensure_exists(macharia_file, kind="file")
        with open(macharia_file, "r") as f:
            data = json.load(f)
        logger.info("Loaded unified input from %s", macharia_file)

        # Normalizações rápidas para chaves esperadas
        for m in data.get("machines", []):
            if "start_slots" not in m:
                m["start_slots"] = m.get("time-index", {}).get("start_slots", [])
            if "time_capacity" not in m:
                m["time_capacity"] = m.get("time-index", {}).get("time_capacity", {})
        for j in data.get("jobs", []):
            ti = j.get("time-index", {})
            gap = j.get("gap", {})
            j.setdefault("processing_slots", ti.get("processing_slots"))
            j.setdefault("release_date_slot", ti.get("release_date_slot"))
            j.setdefault("deadline_slot", ti.get("deadline_slot"))
            j.setdefault("processing_minutes", gap.get("processing_minutes"))
            j.setdefault("release_date_index", gap.get("release_date_index"))
            j.setdefault("deadline_date_index", gap.get("deadline_date_index"))

        # --------- Roda modelos ---------
        gap_out = GapOptimizer().run(data, output_dir)
        js_out = JobSchedulingOptimizer().run(data, output_dir)

        final_output = {
            "gap_optimization": gap_out,
            "job_scheduling_optimization": js_out,
        }

        out_file = output_dir / "macharia_output.json"
        with open(out_file, "w") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        logger.info("✅ Resultados unificados salvos em %s", out_file)

        # ------ Consolidação de métricas ------
        machine_capacity = {m["machine_name"]: m.get("job_capacity", 1) for m in data["machines"]}

        try:
            machine_capacity = {m["machine_name"]: m.get("job_capacity", 1) for m in data["machines"]}
            metrics_highs, metrics_gurobi = [], []

            # ---- HiGHS (máquinas com job_capacity == 1) ----
            for f in Path(output_dir).glob("gap_*_metrics.json"):
                with open(f, "r") as fh:
                    dd = json.load(fh)
                    if isinstance(dd, list):
                        for m in dd:
                            name = m.get("model_label", "").replace("gap_", "")
                            # 🔹 ignora máquinas multi-submáquina
                            if machine_capacity.get(name, 1) > 1:
                                continue
                            metrics_highs.append({
                                "machine_name": name,
                                "solver": m.get("solver"),
                                "objective_value": m.get("objective_value"),
                                "num_variables": m.get("num_variables"),
                                "num_constraints": m.get("num_constraints"),
                                "timestamp": m.get("timestamp"),
                            })

            # ---- Gurobi (máquinas com job_capacity > 1) ----
            for f in Path(output_dir).glob("job_scheduling_*_metrics.json"):
                with open(f, "r") as fh:
                    dd = json.load(fh)
                    if isinstance(dd, list):
                        for m in dd:
                            name = m.get("model_label", "").replace("job_scheduling_", "")
                            #  ignora máquinas simples
                            if machine_capacity.get(name, 1) == 1:
                                continue
                            metrics_gurobi.append({
                                "machine_name": name,
                                "solver": m.get("solver"),
                                "objective_value": m.get("objective_value"),
                                "num_variables": m.get("num_variables"),
                                "num_constraints": m.get("num_constraints"),
                                "timestamp": m.get("timestamp"),
                            })

            if metrics_highs:
                with open(output_dir / "model_metrics_highs.json", "w") as f:
                    json.dump(metrics_highs, f, indent=2, ensure_ascii=False)
                logger.info("Métricas HiGHS consolidadas.")

            if metrics_gurobi:
                with open(output_dir / "model_metrics_gurobi.json", "w") as f:
                    json.dump(metrics_gurobi, f, indent=2, ensure_ascii=False)
                logger.info("Métricas Gurobi consolidadas.")

        except Exception as e:
            logger.warning(f"Falha ao consolidar métricas: {e}")



if __name__ == "__main__":
    main()
