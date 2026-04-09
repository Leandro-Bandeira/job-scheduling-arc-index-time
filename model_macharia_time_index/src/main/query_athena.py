#!/usr/bin/env python3
"""Consulta e organiza registros da tabela afm_demanda no Athena.

O script agora também movimenta os arquivos parquet do lote consultado
para uma pasta de *processing* no S3, deixando a pasta de demanda vazia.

Exemplo de uso:
    python query_athena.py --dt 2024-09-01 --database production_afm_data
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import boto3
from botocore.exceptions import ClientError

DEFAULT_INPUT_PREFIX = "s3://harumi-production-data-pipeline/afm/input/demanda"
TIMESTAMP_FORMATS = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa consulta Athena na demanda AFM.")
    parser.add_argument("--dt", required=True, help="Partição dt a ser consultada (formato YYYY-MM-DD).")
    parser.add_argument(
        "--database",
        default="production_afm_data",
        help="Nome do banco de dados registrado no Glue/Athena.",
    )
    parser.add_argument(
        "--table",
        default="afm_demanda",
        help="Nome da tabela externa do Glue/Athena.",
    )
    parser.add_argument(
        "--workgroup",
        default="production_afm_wg",
        help="Workgroup do Athena a ser utilizado (default: primary).",
    )
    parser.add_argument(
        "--output-location",
        dest="output_location",
        help=(
            "URI S3 para salvar os resultados da consulta. Se omitido, usa a configuração padrão do workgroup."
        ),
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Região AWS (ex.: us-east-1). Se omitido, usa a configuração padrão do ambiente.",
    )
    parser.add_argument(
        "--max-rows",
        dest="max_rows",
        type=int,
        default=50,
        help="Número máximo de linhas a serem exibidas (default: 50).",
    )
    parser.add_argument(
        "--input-prefix",
        default=DEFAULT_INPUT_PREFIX,
        help="Prefixo S3 de entrada (parquet) contendo as partições dt=YYYY-MM-DD.",
    )
    parser.add_argument(
        "--processing-prefix",
        default=None,
        help="Prefixo S3 de destino para mover os lotes processados. Se omisso, usa o input substituindo '/input/' por '/processing/'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exibe quais objetos seriam movidos sem efetivar a cópia/remoção.",
    )
    parser.add_argument(
        "--list-workgroups",
        action="store_true",
        help="Exibe os workgroups disponíveis no Athena e encerra a execução.",
    )
    parser.add_argument(
        "--show-workgroup",
        help="Exibe a configuração completa do workgroup informado e encerra a execução.",
    )
    return parser.parse_args()


def build_query(table: str, dt: str, max_rows: int) -> str:
    return (
        f"SELECT * FROM {table} "
        f"WHERE dt = '{dt}' "
        f"ORDER BY dataHoraProcess DESC "
        f"LIMIT {max_rows}"
    )


def parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    sanitized = candidate.replace("T", " ").replace("/", "-")
    for pattern in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(sanitized, pattern)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(candidate.replace(" ", "T"))
    except ValueError:
        return None


def summarise_by_process_date(
    header: Sequence[str], rows: Iterable[Sequence[str]]
) -> Tuple[Dict[str, int], List[datetime]]:
    try:
        idx = header.index("dataHoraProcess")
    except ValueError:
        return {}, []

    summary: Dict[str, int] = defaultdict(int)
    timestamps: List[datetime] = []
    for row in rows:
        if idx >= len(row):
            continue
        raw = row[idx]
        if not raw:
            continue
        parsed = parse_timestamp(raw)
        if parsed:
            summary[parsed.date().isoformat()] += 1
            timestamps.append(parsed)
        else:
            key = raw.strip() or "desconhecido"
            summary[key] += 1
    return dict(summary), timestamps


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"URI S3 inválida: {uri}")
    key = parsed.path.lstrip("/")
    return parsed.netloc, key


def sanitize_for_s3_folder(label: str) -> str:
    if not label:
        return "batch"
    normalized = label.strip().replace(" ", "T")
    normalized = re.sub(r"[^0-9A-Za-z._-]", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    normalized = normalized.strip("-_")
    return normalized or "batch"


def derive_processing_prefix(input_prefix: str, override: Optional[str]) -> str:
    if override:
        return override.rstrip("/")
    candidate = input_prefix.rstrip("/")
    if "/input/" in candidate:
        return candidate.replace("/input/", "/processing/")
    return f"{candidate}-processing"


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
            print(f"[dry-run] mover s3://{src_bucket}/{key} -> s3://{dst_bucket}/{dest_key}")
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


def resolve_output_location(
    *, client, workgroup: str, explicit_location: Optional[str]
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
        logger.debug(
            "Usando output location do workgroup %s: %s", workgroup, output_location
        )
        return output_location.rstrip("/")

    logger.debug(
        "Workgroup %s não possui OutputLocation configurado.",
        workgroup,
    )
    return None


def list_workgroups(client) -> List[dict]:
    workgroups: List[dict] = []
    next_token: Optional[str] = None

    while True:
        params = {}
        if next_token:
            params["NextToken"] = next_token

        response = client.list_work_groups(**params)
        workgroups.extend(response.get("WorkGroups", []))

        next_token = response.get("NextToken")
        if not next_token:
            break

    return workgroups


def describe_workgroup(client, name: str) -> Optional[dict]:
    try:
        response = client.get_work_group(WorkGroup=name)
    except ClientError as exc:
        logger.error("Falha ao obter configuração do workgroup %s: %s", name, exc)
        return None

    return response.get("WorkGroup")


def start_query(
    *,
    client,
    query: str,
    database: str,
    workgroup: str,
    output_location: Optional[str] = None,
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


def wait_for_query(client, execution_id: str, poll_seconds: int = 2) -> dict:
    while True:
        result = client.get_query_execution(QueryExecutionId=execution_id)
        status = result["QueryExecution"]["Status"]["State"]
        if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return result
        time.sleep(poll_seconds)


def fetch_rows(client, execution_id: str) -> Iterable[List[str]]:
    paginator = client.get_paginator("get_query_results")
    for page in paginator.paginate(QueryExecutionId=execution_id):
        rows = page.get("ResultSet", {}).get("Rows", [])
        for row in rows:
            yield [col.get("VarCharValue", "") for col in row.get("Data", [])]


def main() -> int:
    args = parse_args()
    query = build_query(args.table, args.dt, args.max_rows)
    input_prefix = args.input_prefix.rstrip("/")
    processing_prefix = derive_processing_prefix(input_prefix, args.processing_prefix)

    summary_by_date: Dict[str, int] = {}
    process_timestamps: List[datetime] = []

    region = resolve_region(args.region)

    try:
        athena_client = boto3.client("athena", region_name=region)
    except ClientError as exc:
        print(f"Erro ao inicializar cliente Athena: {exc}", file=sys.stderr)
        return 1

    if args.list_workgroups:
        try:
            workgroups = list_workgroups(athena_client)
        except ClientError as exc:
            print(f"Falha ao listar workgroups: {exc}", file=sys.stderr)
            return 1

        if not workgroups:
            print("Nenhum workgroup encontrado.")
        else:
            print("Workgroups disponíveis:")
            for item in workgroups:
                name = item.get("Name", "<desconhecido>")
                state = item.get("State", "<sem estado>")
                config = item.get("Configuration", {}) or {}
                result_conf = config.get("ResultConfiguration", {}) or {}
                output_location = result_conf.get("OutputLocation", "<não configurado>")
                enforced = config.get("EnforceWorkGroupConfiguration", False)
                publish_cloudwatch = config.get("PublishCloudWatchMetricsEnabled", False)

                print(f"  - {name} (estado: {state})")
                print(f"      output_location: {output_location}")
                print(f"      enforce_configuration: {enforced}")
                print(f"      publish_cloudwatch_metrics: {publish_cloudwatch}")
        return 0

    if args.show_workgroup:
        workgroup_info = describe_workgroup(athena_client, args.show_workgroup)
        if not workgroup_info:
            print(f"Não foi possível obter informações do workgroup {args.show_workgroup}", file=sys.stderr)
            return 1

        print("Configuração do workgroup:")
        print(json.dumps(workgroup_info, indent=2, sort_keys=True, default=str))
        return 0

    output_location = resolve_output_location(
        client=athena_client,
        workgroup=args.workgroup,
        explicit_location=args.output_location,
    )

    try:
        execution_id = start_query(
            client=athena_client,
            query=query,
            database=args.database,
            workgroup=args.workgroup,
            output_location=output_location,
        )
    except ClientError as exc:
        error_message = getattr(exc, "response", {}).get("Error", {}).get("Message", str(exc))
        print(f"Falha ao iniciar consulta: {error_message}", file=sys.stderr)
        if "No output location provided" in error_message:
            print(
                "Informe --output-location ou habilite 'Query result location' no workgroup do Athena.",
                file=sys.stderr,
            )
            if not output_location:
                print(
                    "Nenhum output location foi detectado automaticamente; verifique as permissões ou a configuração do workgroup.",
                    file=sys.stderr,
                )
        return 1

    status = wait_for_query(athena_client, execution_id)
    state = status["QueryExecution"]["Status"]["State"]
    if state != "SUCCEEDED":
        reason = status["QueryExecution"]["Status"].get("StateChangeReason", "")
        print(f"Consulta finalizada com status {state}: {reason}", file=sys.stderr)
        return 1

    print(f"Consulta ({execution_id}) concluída com sucesso. Exibindo resultados:")
    rows = fetch_rows(athena_client, execution_id)
    results = list(rows)
    if not results:
        print("Nenhum dado retornado.")
        return 0

    header, *data_rows = results
    widths = [len(col) for col in header]
    for row in data_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def format_row(values: List[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    print(format_row(header))
    print("-" * (sum(widths) + (3 * (len(header) - 1))))
    for row in data_rows:
        print(format_row(row))

    summary_by_date, process_timestamps = summarise_by_process_date(header, data_rows)

    if summary_by_date:
        print("\nResumo por data de processamento:")
        for key in sorted(summary_by_date.keys()):
            print(f"  {key}: {summary_by_date[key]} registros")
    else:
        print("\nColuna dataHoraProcess ausente ou sem valores para agrupar.")

    if args.dry_run:
        print("\n[dry-run] Nenhuma operação S3 será executada.")

    try:
        s3_client = boto3.client("s3", region_name=region)
    except ClientError as exc:
        print(f"Erro ao inicializar cliente S3: {exc}", file=sys.stderr)
        return 1

    if process_timestamps:
        latest_ts = max(process_timestamps)
        batch_slug = sanitize_for_s3_folder(latest_ts.strftime("%Y%m%dT%H%M%S"))
    elif summary_by_date:
        batch_slug = sanitize_for_s3_folder(sorted(summary_by_date.keys())[-1])
    else:
        batch_slug = sanitize_for_s3_folder(datetime.utcnow().strftime("%Y%m%dT%H%M%S"))

    try:
        moved, total, destination_uri = move_partition_objects(
            client=s3_client,
            input_prefix=input_prefix,
            processing_prefix=processing_prefix,
            dt=args.dt,
            batch_slug=batch_slug,
            dry_run=args.dry_run,
        )
    except (ClientError, ValueError) as exc:
        print(f"Falha ao mover arquivos do S3: {exc}", file=sys.stderr)
        return 1

    if total == 0:
        print(f"Nenhum objeto encontrado em {input_prefix}/dt={args.dt}/ para mover.")
    else:
        if args.dry_run:
            print(f"[dry-run] {total} objetos seriam movidos para {destination_uri}")
        else:
            print(f"Movidos {moved} de {total} objetos para {destination_uri}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
