# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import fire
import atexit
import signal
import logging
import importlib
import unitorch.cli
from pathlib import Path
from unitorch import hf_cached_path
from unitorch.cli import CoreConfigureParser
from unitorch.services import registered_service
from unitorch_cli import get_config_file, load_template


def _sigterm_handler(signo, frame):
    raise SystemExit(1)


def daemonize(pid_file, name):
    if os.path.exists(pid_file):
        raise RuntimeError(f"unitorch-auto-service {name} already running")
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    os.chdir("/")
    os.setsid()
    os.umask(0)

    _pid = os.fork()
    if _pid:
        sys.exit(0)

    sys.stdout.flush()
    sys.stderr.flush()

    stdin_file = f"/tmp/unitorch_auto_service_{name}.stdin.log"
    stdout_file = f"/tmp/unitorch_auto_service_{name}.stdout.log"

    Path(stdin_file).touch()
    Path(stdout_file).touch()

    with open(stdin_file, "rb") as read_file, open(stdout_file, "ab") as write_file:
        os.dup2(read_file.fileno(), sys.stdin.fileno())
        os.dup2(write_file.fileno(), sys.stdout.fileno())
        os.dup2(write_file.fileno(), sys.stderr.fileno())

    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    atexit.register(lambda: os.remove(pid_file))
    signal.signal(signal.SIGTERM, _sigterm_handler)


def start(name, inst, daemon_mode):
    name = name.replace("/", "_")
    pid_file = f"/tmp/unitorch_auto_service_{name}.pid"
    if daemon_mode:
        daemonize(pid_file, name)
    inst.start()


def stop(name, inst):
    name = name.replace("/", "_")
    pid_file = f"/tmp/unitorch_auto_service_{name}.pid"
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            os.kill(int(f.read()), signal.SIGTERM)


def restart(name, inst):
    stop(name, inst)
    start(name, inst)


@fire.decorators.SetParseFn(str)
def service(service_action: str, service_path_or_dir: str, **kwargs):
    config_file = kwargs.pop("config_file", "config.ini")

    if service_path_or_dir and os.path.isdir(service_path_or_dir):
        config_path = os.path.join(service_path_or_dir, config_file)
        sys.path.insert(0, service_path_or_dir)
        for f in os.listdir(service_path_or_dir):
            fpath = os.path.normpath(os.path.join(service_path_or_dir, f))
            if (
                not f.startswith("_")
                and not f.startswith(".")
                and (f.endswith(".py") or os.path.isdir(fpath))
            ):
                fname = f[:-3] if f.endswith(".py") else f
                module = importlib.import_module(f"{fname}")
    elif not service_path_or_dir.endswith(".ini"):
        load_template(service_path_or_dir)
        config_path = os.path.join(service_path_or_dir, config_file)
        config_path = get_config_file(config_path)
        if config_path is None:
            config_path = get_config_file(config_file)
    else:
        config_path = get_config_file(service_path_or_dir)

    params = []
    for k, v in kwargs.items():
        if k.count("@") > 0:
            k0 = k.split("@")[0]
            k1 = "@".join(k.split("@")[1:])
        else:
            k0 = "core/cli"
            k1 = k
        params.append((k0, k1, v))

    if config_path is not None:
        config = CoreConfigureParser(config_path, params=params)
    else:
        config = CoreConfigureParser(params=params)

    depends_templates = config.getdefault("core/cli", "depends_templates", None)

    if depends_templates:
        for template in depends_templates:
            load_template(template)

    daemon_mode = config.getdefault("core/cli", "daemon_mode", True)
    service_name = config.getdefault("core/cli", "service_name", None)
    assert service_name is not None
    main_service_cls = registered_service.get(service_name)
    if main_service_cls is None:
        raise ValueError(f"service {service_name} not found")
    service_inst = main_service_cls["obj"](config)

    if service_action == "start":
        start(service_name, service_inst, daemon_mode)
    elif service_action == "stop":
        stop(service_name, service_inst)
    elif service_action == "restart":
        restart(service_name, service_inst)
    else:
        raise ValueError(f"service action {service_action} not found")


def cli_main():
    fire.Fire(service)
