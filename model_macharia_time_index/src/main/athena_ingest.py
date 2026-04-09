from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

import boto3
from botocore.exceptions import ClientError


LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("athena-ingest")

DEFAULT_INPUT_PREFIX = "s3://harumi-production-data-pipeline/afm/input/demanda"
ATHENA_POLL_SECONDS = 2
STATUS_COLUMN = "StatusIntegrationId"
def ensure_model_config_directory(model_config_dir: Path, possible_sources: Iterable[Path]) -> Path:
    model_config_dir.mkdir(parents=True, exist_ok=True)

    existing_files = {item.name for item in model_config_dir.iterdir()} if model_config_dir.exists() else set()

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
class IngestionResult:
    datasets: List[Tuple[Path, str, Path]]
    model_config_dir: Path
    batch_slug: str
    date_dir: Path


def sanitize_for_s3_folder(label: str) -> str:
    if not label:
        return "batch"
    normalized = label.strip().replace(" ", "T")
    normalized = re.sub(r"[^0-9A-Za-z._-]", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    normalized = normalized.strip("-_")
    return normalized or "batch"


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"URI S3 inválida: {uri}")
    key = parsed.path.lstrip("/")
    return parsed.netloc, key


def derive_processing_prefix(input_prefix: str, override: Optional[str]) -> str:
    if override:
        return override.rstrip("/")
    candidate = input_prefix.rstrip("/")
    if "/input/" in candidate:
        return candidate.replace("/input/", "/processing/")
    return f"{candidate}-processing"


def iter_s3_objects(client, bucket: str, prefix: str) -> Iterable[str]:
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def move_partition_objects(
    *,
    client,
    input_prefix: str,
    processing_prefix: str,
    dt: str,
    batch_slug: str,
    dry_run: bool,
) -> Tuple[int, int, str]:
    src_bucket, src_base = parse_s3_uri(input_prefix)
    dst_bucket, dst_base = parse_s3_uri(processing_prefix)

    src_base = src_base.rstrip("/")
    dst_base = dst_base.rstrip("/")

    src_prefix_dt = f"{src_base}/dt={dt}/"
    dst_prefix_dt = f"{dst_base}/dt={dt}/batch={batch_slug}/"

    total = 0
    moved = 0

    for key in iter_s3_objects(client, src_bucket, src_prefix_dt):
        if not key.startswith(src_prefix_dt):
            continue
        if key.endswith("/"):
            continue
        total += 1
        relative = key[len(src_prefix_dt) :]
        dest_key = f"{dst_prefix_dt}{relative}"
        if dry_run:
            logger.info("[dry-run] mover s3://%s/%s -> s3://%s/%s", src_bucket, key, dst_bucket, dest_key)
            moved += 1
            continue

        client.copy_object(
            Bucket=dst_bucket,
            Key=dest_key,
            CopySource={"Bucket": src_bucket, "Key": key},
        )
        client.delete_object(Bucket=src_bucket, Key=key)
        moved += 1

    destination_uri = f"s3://{dst_bucket}/{dst_prefix_dt}"
    return moved, total, destination_uri


def resolve_region(explicit_region: Optional[str]) -> Optional[str]:
    if explicit_region:
        return explicit_region
    env_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if env_region:
        logger.debug("Usando região do ambiente: %s", env_region)
        return env_region
    fallback = "us-east-2"
    logger.debug(
        "Nenhuma região informada; utilizando fallback %s (Ohio). Informe --region ou configure AWS_REGION.",
        fallback,
    )
    return fallback


def resolve_output_location(
    *,
    client,
    workgroup: str,
    explicit_location: Optional[str],
) -> Optional[str]:
    if explicit_location:
        return explicit_location.rstrip("/")

    try:
        response = client.get_work_group(WorkGroup=workgroup)
    except ClientError as exc:
        logger.debug("Não foi possível obter configuração do workgroup %s: %s", workgroup, exc)
        return None

    configuration = response.get("WorkGroup", {}).get("Configuration", {})
    result_config = configuration.get("ResultConfiguration", {})
    output_location = result_config.get("OutputLocation")

    if output_location:
        logger.debug("Usando output location do workgroup %s: %s", workgroup, output_location)
        return output_location.rstrip("/")

    logger.debug("Workgroup %s não possui OutputLocation configurado.", workgroup)
    return None


def build_athena_query(table: str, dt: str) -> str:
    sanitized_dt = dt.replace("'", "''")
    return f"SELECT * FROM {table} WHERE dt = '{sanitized_dt}'"


def start_athena_query(
    *,
    client,
    query: str,
    database: str,
    workgroup: str,
    output_location: Optional[str],
) -> str:
    params = {
        "QueryString": query,
        "QueryExecutionContext": {"Database": database},
        "WorkGroup": workgroup,
    }
    if output_location:
        params["ResultConfiguration"] = {"OutputLocation": output_location}
    response = client.start_query_execution(**params)
    return response["QueryExecutionId"]


def wait_for_query(client, execution_id: str, poll_seconds: int = ATHENA_POLL_SECONDS) -> dict:
    while True:
        result = client.get_query_execution(QueryExecutionId=execution_id)
        status = result["QueryExecution"]["Status"]["State"]
        if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return result
        time.sleep(poll_seconds)


def fetch_query_dataframe(client, execution_id: str) -> pd.DataFrame:
    paginator = client.get_paginator("get_query_results")
    header: Optional[List[str]] = None
    rows: List[List[str]] = []

    for page in paginator.paginate(QueryExecutionId=execution_id):
        result_rows = page.get("ResultSet", {}).get("Rows", [])
        for row in result_rows:
            values = [col.get("VarCharValue", "") for col in row.get("Data", [])]
            if not header:
                header = values
                continue
            if len(values) != len(header):
                continue
            rows.append(values)

    if not header:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=header)


def sanitize_status_label(raw_value: Optional[str]) -> str:
    value = (raw_value or "sem-status").strip()
    if not value:
        value = "sem-status"
    normalized = re.sub(r"[^0-9A-Za-z._-]", "-", value)
    normalized = re.sub(r"-+", "-", normalized)
    normalized = normalized.strip("-_")
    return normalized or "sem-status"


def write_demand_partitions(
    df: pd.DataFrame,
    *,
    date_dir: Path,
    status_column: str = STATUS_COLUMN,
) -> List[Tuple[Path, str, Path]]:
    if status_column not in df.columns:
        candidates = [col for col in df.columns if col.lower() == status_column.lower()]
        if candidates:
            status_column = candidates[0]
        else:
            raise KeyError(
                f"Coluna {status_column} não encontrada no resultado da consulta Athena."
            )

    datasets: List[Tuple[Path, str, Path]] = []
    date_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby(status_column, dropna=False)
    for status_value, subset in grouped:
        status_label = sanitize_status_label(str(status_value) if status_value is not None else "sem-status")
        dataset_dir = date_dir / status_label
        demanda_dir = dataset_dir / "demanda"
        demanda_dir.mkdir(parents=True, exist_ok=True)

        part_name = f"{status_label}-part-0000.parquet"
        part_path = demanda_dir / part_name
        subset.to_parquet(part_path, engine="pyarrow", index=False)

        flat_part_path = date_dir / part_name
        shutil.copy2(part_path, flat_part_path)

        datasets.append((dataset_dir, status_label, part_path))

    return datasets


def extract_latest_process_timestamp(df: pd.DataFrame) -> Optional[datetime]:
    for candidate in ("dataHoraProcess", "datahoraprocess"):
        if candidate in df.columns:
            column = candidate
            break
    else:
        return None
    try:
        timestamps = pd.to_datetime(df[column], errors="coerce")
    except Exception:
        return None
    if timestamps.isna().all():
        return None
    return timestamps.max().to_pydatetime()


def ingest_from_athena(
    *,
    dt: str,
    raw_root: Path,
    model_config_dir: Path,
    database: str,
    table: str,
    workgroup: str,
    output_location: Optional[str],
    region: Optional[str],
    input_prefix: str,
    processing_prefix: Optional[str],
    dry_run: bool,
    skip_move: bool,
) -> IngestionResult:
    try:
        dt_obj = datetime.strptime(dt, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"Data inválida para --dt: {dt}") from exc

    date_slug = dt_obj.strftime("%d%m%Y")
    date_dir = raw_root / date_slug
    date_dir.mkdir(parents=True, exist_ok=True)

    region_name = resolve_region(region)
    logger.info("Região AWS utilizada: %s", region_name)

    try:
        athena_client = boto3.client("athena", region_name=region_name)
    except ClientError as exc:
        raise SystemExit(f"Erro ao inicializar cliente Athena: {exc}") from exc

    output_location = resolve_output_location(
        client=athena_client,
        workgroup=workgroup,
        explicit_location=output_location,
    )

    query = build_athena_query(table, dt)
    logger.info("Iniciando consulta Athena: %s", query)

    try:
        execution_id = start_athena_query(
            client=athena_client,
            query=query,
            database=database,
            workgroup=workgroup,
            output_location=output_location,
        )
    except ClientError as exc:
        error_message = getattr(exc, "response", {}).get("Error", {}).get("Message", str(exc))
        logger.error("Falha ao iniciar consulta: %s", error_message)
        if "No output location provided" in error_message and not output_location:
            logger.error(
                "Nenhum output location foi detectado automaticamente; verifique as permissões ou a configuração do workgroup."
            )
        raise SystemExit(1) from exc

    status = wait_for_query(athena_client, execution_id)
    state = status["QueryExecution"]["Status"]["State"]
    if state != "SUCCEEDED":
        reason = status["QueryExecution"]["Status"].get("StateChangeReason", "")
        logger.error("Consulta finalizada com status %s: %s", state, reason)
        raise SystemExit(1)

    df = fetch_query_dataframe(athena_client, execution_id)
    if df.empty:
        logger.warning("Nenhum registro retornado para dt=%s.", dt)
        return IngestionResult(datasets=[], model_config_dir=model_config_dir, batch_slug="", date_dir=date_dir)

    logger.info("Demandas recuperadas: %d linhas", len(df))

    datasets = write_demand_partitions(df, date_dir=date_dir, status_column=STATUS_COLUMN)
    logger.info("Lotes de demanda gerados: %d", len(datasets))

    possible_parameter_sources = [date_dir / "parameters", raw_root / "parameters"]
    model_config_dir = ensure_model_config_directory(model_config_dir, possible_parameter_sources)

    latest_ts = extract_latest_process_timestamp(df)
    if latest_ts:
        batch_slug = sanitize_for_s3_folder(latest_ts.strftime("%Y%m%dT%H%M%S"))
    else:
        batch_slug = sanitize_for_s3_folder(dt)

    if not skip_move:
        try:
            s3_client = boto3.client("s3", region_name=region_name)
        except ClientError as exc:
            raise SystemExit(f"Erro ao inicializar cliente S3: {exc}") from exc

        target_prefix = derive_processing_prefix(input_prefix, processing_prefix)
        moved, total, destination_uri = move_partition_objects(
            client=s3_client,
            input_prefix=input_prefix,
            processing_prefix=target_prefix,
            dt=dt,
            batch_slug=batch_slug,
            dry_run=dry_run,
        )

        if total == 0:
            logger.info("Nenhum objeto encontrado para mover em %s/dt=%s/.", input_prefix, dt)
        else:
            if dry_run:
                logger.info("[dry-run] %d objetos seriam movidos para %s", total, destination_uri)
            else:
                logger.info("Movidos %d de %d objetos para %s", moved, total, destination_uri)

    return IngestionResult(datasets=datasets, model_config_dir=model_config_dir, batch_slug=batch_slug, date_dir=date_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consulta demandas no Athena e organiza arquivos locais por status_integration_id."
    )
    default_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--dt", required=True, help="Partição dt a ser consultada (formato YYYY-MM-DD).")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=default_root.joinpath("data", "raw"),
        help="Diretório raiz local para armazenar os lotes importados (default: data/raw).",
    )
    parser.add_argument(
        "--model-config-dir",
        type=Path,
        default=default_root.joinpath("data", "raw", "model_config"),
        help="Diretório compartilhado para os parâmetros do modelo (default: data/raw/model_config).",
    )
    parser.add_argument(
        "--database",
        default="production_afm_data",
        help="Nome do banco de dados no Athena/Glue (default: production_afm_data).",
    )
    parser.add_argument(
        "--table",
        default="afm_demanda",
        help="Nome da tabela de demanda no Athena (default: afm_demanda).",
    )
    parser.add_argument(
        "--workgroup",
        default="production_afm_wg",
        help="Workgroup do Athena a ser utilizado (default: production_afm_wg).",
    )
    parser.add_argument(
        "--output-location",
        dest="output_location",
        default="s3://harumi-production-data-pipeline/afm/athena-results/",
        help="URI S3 para salvar os resultados da consulta. Se omitido, usa a configuração do workgroup.",
    )
    parser.add_argument(
        "--region",
        default="us-east-2",
        help="Região AWS (ex.: us-east-1). Se omitido, usa a configuração do ambiente ou fallback.",
    )
    parser.add_argument(
        "--input-prefix",
        default=DEFAULT_INPUT_PREFIX,
        help="Prefixo S3 de entrada contendo partições dt=YYYY-MM-DD.",
    )
    parser.add_argument(
        "--processing-prefix",
        default=None,
        help="Prefixo S3 de destino para mover os lotes processados. Se omisso, usa input -> processing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Executa consultas e grava arquivos sem mover objetos no S3.",
    )
    parser.add_argument(
        "--skip-move",
        action="store_true",
        help="Não move os arquivos do S3 para a pasta de processing após o download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ingestion = ingest_from_athena(
        dt=args.dt,
        raw_root=args.raw_root,
        model_config_dir=args.model_config_dir,
        database=args.database,
        table=args.table,
        workgroup=args.workgroup,
        output_location=args.output_location,
        region=args.region,
        input_prefix=args.input_prefix,
        processing_prefix=args.processing_prefix,
        dry_run=args.dry_run,
        skip_move=args.skip_move,
    )

    if not ingestion.datasets:
        logger.warning("Nenhum conjunto de dados para processar.")
        return

    logger.info(
        "Demandas baixadas para %s. Processamento local deve ser executado manualmente via data_input_process.py.",
        ingestion.date_dir,
    )


if __name__ == "__main__":
    main()
