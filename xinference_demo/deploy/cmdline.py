import click
import logging
from typing import Optional

from .utils import get_config_dict, get_log_file, get_timestamp_ms
from ..constants import (
    XINFERENCE_DEFAULT_ENDPOINT_PORT,
    XINFERENCE_DEFAULT_LOCAL_HOST,
    XINFERENCE_LOG_BACKUP_COUNT,
    XINFERENCE_LOG_MAX_BYTES,
)

    # XINFERENCE_AUTH_DIR,
    # XINFERENCE_DEFAULT_DISTRIBUTED_HOST,
    # XINFERENCE_DEFAULT_ENDPOINT_PORT,
    # ,
    # XINFERENCE_ENV_ENDPOINT,

def start_local_cluster(
    log_level: str,
    host: str,
    port: int,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    auth_config_file: Optional[str] = None,
):
    from .local import main

    dict_config = get_config_dict(
        log_level,
        get_log_file(f"local_{get_timestamp_ms()}"),
        XINFERENCE_LOG_BACKUP_COUNT,
        XINFERENCE_LOG_MAX_BYTES,
    )
    logging.config.dictConfig(dict_config)  # type: ignore

    main(
        host=host,
        port=port,
        metrics_exporter_host=metrics_exporter_host,
        metrics_exporter_port=metrics_exporter_port,
        logging_conf=dict_config,
        auth_config_file=auth_config_file,
    )


@click.command(help="Starts an Xinference local cluster.")
@click.option(
    "--log-level",
    default="INFO",
    type=str,
    help="""Set the logger level. Options listed from most log to least log are:
              DEBUG > INFO > WARNING > ERROR > CRITICAL (Default level is INFO)""",
)
@click.option(
    "--host",
    "-H",
    default=XINFERENCE_DEFAULT_LOCAL_HOST,
    type=str,
    help="Specify the host address for the Xinference server.",
)
@click.option(
    "--port",
    "-p",
    default=XINFERENCE_DEFAULT_ENDPOINT_PORT,
    type=int,
    help="Specify the port number for the Xinference server.",
)
@click.option(
    "--metrics-exporter-host",
    "-MH",
    default=None,
    type=str,
    help="Specify the host address for the Xinference metrics exporter server, default is the same as --host.",
)
@click.option(
    "--metrics-exporter-port",
    "-mp",
    type=int,
    help="Specify the port number for the Xinference metrics exporter server.",
)
@click.option(
    "--auth-config",
    type=str,
    help="Specify the auth config json file.",
)
def local(
    log_level: str,
    host: str,
    port: int,
    metrics_exporter_host: Optional[str],
    metrics_exporter_port: Optional[int],
    auth_config: Optional[str],
):
    if metrics_exporter_host is None:
        metrics_exporter_host = host
    start_local_cluster(
        log_level=log_level,
        host=host,
        port=port,
        metrics_exporter_host=metrics_exporter_host,
        metrics_exporter_port=metrics_exporter_port,
        auth_config_file=auth_config,
    )