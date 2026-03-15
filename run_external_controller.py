#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _eprint(msg: str) -> None:
    sys.stderr.write(msg.rstrip("\n") + "\n")
    sys.stderr.flush()


def _parse_dotenv_value(raw: str) -> str:
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] == "`":
        return v[1:-1]
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        q = v[0]
        inner = v[1:-1]
        if q == "'":
            return inner
        out = []
        i = 0
        while i < len(inner):
            c = inner[i]
            if c != "\\":
                out.append(c)
                i += 1
                continue
            i += 1
            if i >= len(inner):
                break
            esc = inner[i]
            i += 1
            if esc == "n":
                out.append("\n")
            elif esc == "r":
                out.append("\r")
            elif esc == "t":
                out.append("\t")
            elif esc in ('"', "\\"):
                out.append(esc)
            else:
                out.append(esc)
        return "".join(out)
    return v


def load_dotenv(env_path: Path) -> tuple[int, list[str]]:
    if not env_path.exists():
        return 0, []
    loaded_keys: list[str] = []
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ[key] = _parse_dotenv_value(value)
        loaded_keys.append(key)
    return len(loaded_keys), loaded_keys


def _pkill(pattern: str) -> None:
    try:
        subprocess.run(["pkill", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        return


def _stream_process(p: subprocess.Popen, *, log_path: Path | None, prefix: str) -> int:
    f = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            f = open(log_path, "a", encoding="utf-8")
        assert p.stdout is not None
        for line in p.stdout:
            out = f"{prefix}{line}"
            sys.stdout.write(out)
            sys.stdout.flush()
            if f is not None:
                f.write(out)
                f.flush()
    finally:
        if f is not None:
            f.close()
    return p.wait()


def main(argv: list[str]) -> int:
    pipeline_root = Path(__file__).resolve().parent

    controller_dir = Path(os.environ.get("AR_CONTROLLER_DIR", "/root/autoresearch-controller"))
    controller_py = controller_dir / "controller.py"
    if not controller_py.exists():
        raise FileNotFoundError(str(controller_py))

    env_file = Path(os.environ.get("AR_CONTROLLER_ENV_FILE", str(controller_dir / ".env")))

    os.environ.setdefault("AR_TARGET_REPO", str(pipeline_root))
    os.environ.setdefault("AR_TARGET_BRANCH", "autoresearch/mar14")
    os.environ.setdefault("AR_RUNS_DIR", str(controller_dir / "runs"))
    os.environ.setdefault("AR_USE_UV", "1")

    if (os.environ.get("AR_STOP_LEGACY_RUNNER", "0") or "0").strip() == "1":
        _eprint("[controller] Stopping legacy runner processes...")
        _pkill("run_until_monday_utc.sh")
        _pkill("python train.py")

    quiet = (os.environ.get("AR_WRAPPER_QUIET", "0") or "0").strip() == "1"
    if not quiet:
        _eprint("[controller] External controller wrapper starting")
        _eprint(f"[controller] pipeline_repo={pipeline_root}")
        _eprint(f"[controller] controller_dir={controller_dir}")
        _eprint(f"[controller] target_repo={os.environ.get('AR_TARGET_REPO')}")
        _eprint(f"[controller] target_branch={os.environ.get('AR_TARGET_BRANCH')}")
        _eprint(f"[controller] runs_dir={os.environ.get('AR_RUNS_DIR')}")

    loaded_n, loaded_keys = load_dotenv(env_file)
    if not quiet:
        if env_file.exists():
            _eprint(f"[controller] loaded_env_file={env_file} (keys={loaded_n})")
        else:
            _eprint(f"[controller] loaded_env_file=none (missing {env_file})")

        llm_base = os.environ.get("AR_LLM_BASE_URL", "").strip()
        llm_model = os.environ.get("AR_LLM_MODEL", "").strip()
        llm_key_set = bool(os.environ.get("AR_LLM_API_KEY", "").strip())
        _eprint(f"[controller] llm_base_url_set={bool(llm_base)} llm_model_set={bool(llm_model)} llm_api_key_set={llm_key_set}")
        if llm_base:
            _eprint(f"[controller] llm_base_url={llm_base}")
        if llm_model:
            _eprint(f"[controller] llm_model={llm_model}")
        if "`" in llm_base:
            _eprint("[controller] WARNING: llm_base_url contains backticks; fix .env to remove them")

    llm_base = os.environ.get("AR_LLM_BASE_URL", "").strip()
    llm_model = os.environ.get("AR_LLM_MODEL", "").strip()
    llm_key_set = bool(os.environ.get("AR_LLM_API_KEY", "").strip())
    allow_no_llm = (os.environ.get("AR_ALLOW_NO_LLM", "0") or "0").strip() == "1"
    if (llm_base or llm_model) and not llm_key_set:
        msg = "[controller] ERROR: AR_LLM_API_KEY is missing/empty (LLM calls will fail)"
        if allow_no_llm:
            if not quiet:
                _eprint(msg + " (continuing because AR_ALLOW_NO_LLM=1)")
        else:
            _eprint(msg + " (set it in the controller .env or export it before launch)")
            return 2

    log_path_raw = os.environ.get("AR_CONTROLLER_LOG_PATH", "").strip()
    if log_path_raw in ("-", "stdout", "STDOUT"):
        log_path = None
    elif log_path_raw:
        log_path = Path(log_path_raw)
    else:
        log_path = controller_dir / "controller.out"
    prefix = os.environ.get("AR_CONTROLLER_LOG_PREFIX", "[controller.py] ")
    if not quiet:
        _eprint(f"[controller] controller_log_path={str(log_path) if log_path else 'stdout'}")
        _eprint(f"[controller] exec={sys.executable} -u {controller_py} {' '.join(argv)}")

    os.chdir(controller_dir)
    p = subprocess.Popen(
        [sys.executable, "-u", str(controller_py), *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return _stream_process(p, log_path=log_path, prefix=prefix)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
