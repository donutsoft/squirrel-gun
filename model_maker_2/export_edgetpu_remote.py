#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, urlencode, urlparse, urlunparse

try:
    import requests
    import websocket
except ModuleNotFoundError as exc:
    CLIENT_IMPORT_ERROR: Optional[ModuleNotFoundError] = exc
else:
    CLIENT_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("model_maker_2/yolo")
DEFAULT_OUTPUT = Path("squirrel-daemon/best_full_integer_quant_edgetpu.tflite")
REMOTE_OUTPUT_NAME = "best_full_integer_quant_edgetpu.tflite"
DEFAULT_JUPYTER_TOKEN = "7eea26785b6ef849673ed712415b9c69a9600f5aa4a537c5"

class JupyterError(RuntimeError):
    pass


class JupyterClient:
    def __init__(self, base_url: str, token: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()
        self._xsrf_token: Optional[str] = None

    def _url(self, path: str, params: Optional[Dict[str, str]] = None, ws: bool = False) -> str:
        parsed = urlparse(self.base_url)
        scheme = parsed.scheme
        if ws:
            scheme = "wss" if scheme == "https" else "ws"

        query: Dict[str, str] = {}
        if self.token:
            query["token"] = self.token
        if params:
            query.update(params)

        base_path = parsed.path.rstrip("/")
        api_path = "/" + path.lstrip("/")
        return urlunparse(
            (
                scheme,
                parsed.netloc,
                base_path + api_path,
                "",
                urlencode(query),
                "",
            )
        )

    def request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        try:
            headers = {"Accept": "application/json"}
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            if method.upper() not in {"GET", "HEAD", "OPTIONS"}:
                headers["X-XSRFToken"] = self.xsrf_token()
            resp = self.session.request(
                method,
                self._url(path, params=params),
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            detail = ""
            response = getattr(exc, "response", None)
            if response is not None:
                detail = f": {response.text}"
            raise JupyterError(f"{method} {path} failed: {exc}{detail}") from exc
        if not resp.content:
            return {}
        return resp.json()

    def xsrf_token(self) -> str:
        if self._xsrf_token:
            return self._xsrf_token

        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        try:
            resp = self.session.get(self._url("/api"), headers=headers, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise JupyterError(f"GET /api failed while establishing XSRF cookie: {exc}") from exc

        token = self.session.cookies.get("_xsrf")
        if not token:
            # Some Jupyter deployments only set the cookie on the browser route.
            try:
                params = {"token": self.token} if self.token else None
                resp = self.session.get(self.base_url + "/", headers=headers, params=params, timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise JupyterError("GET / failed while establishing XSRF cookie.") from exc
            token = self.session.cookies.get("_xsrf")
        if not token:
            raise JupyterError("Jupyter did not set an _xsrf cookie.")

        self._xsrf_token = token
        return token

    def make_dir(self, path: str) -> None:
        try:
            self.request("PUT", f"/api/contents/{quote_path(path)}", {"type": "directory"})
        except JupyterError as exc:
            if "409" not in str(exc):
                raise

    def contents(self, path: str = "", content: bool = True) -> Dict[str, Any]:
        params = {"content": "1" if content else "0"}
        api_path = "/api/contents"
        if path:
            api_path += f"/{quote_path(path)}"
        return self.request("GET", api_path, params=params)

    def discover_writable_base_dir(self, preferred_notebook: str = "ModelConverter.ipynb") -> str:
        root = self.contents("", content=True)
        root_writable = bool(root.get("writable"))
        queue: List[Tuple[str, Dict[str, Any], int]] = [("", root, 0)]
        first_writable = "" if root_writable else None
        max_dirs = 200
        checked = 0

        while queue and checked < max_dirs:
            path, model, depth = queue.pop(0)
            checked += 1
            entries = model.get("content") or []
            if not isinstance(entries, list):
                continue

            if any(entry.get("name") == preferred_notebook for entry in entries):
                if model.get("writable"):
                    return path
                print(f"[WARN] Found {preferred_notebook} in read-only remote directory: {path or '/'}")

            for entry in entries:
                if entry.get("type") != "directory":
                    continue
                entry_path = entry.get("path") or entry.get("name")
                if not entry_path:
                    continue
                if first_writable is None and entry.get("writable"):
                    first_writable = str(entry_path)
                if depth >= 3:
                    continue
                try:
                    child = self.contents(str(entry_path), content=True)
                except JupyterError as exc:
                    print(f"[WARN] Could not inspect remote directory {entry_path}: {exc}")
                    continue
                queue.append((str(entry_path), child, depth + 1))

        if first_writable is not None:
            return first_writable
        raise JupyterError("Could not find a writable remote Jupyter directory.")

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        encoded = base64.b64encode(local_path.read_bytes()).decode("ascii")
        self.request(
            "PUT",
            f"/api/contents/{quote_path(remote_path)}",
            {
                "type": "file",
                "format": "base64",
                "content": encoded,
            },
        )

    def download_file(self, remote_path: str, local_path: Path) -> None:
        payload = self.request(
            "GET",
            f"/api/contents/{quote_path(remote_path)}",
            params={"format": "base64"},
        )
        if payload.get("type") != "file" or payload.get("format") != "base64":
            raise JupyterError(f"Unexpected contents payload for {remote_path}: {payload.get('type')}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(base64.b64decode(payload["content"]))

    def start_kernel(self) -> str:
        payload = self.request("POST", "/api/kernels", {})
        kernel_id = payload.get("id")
        if not kernel_id:
            raise JupyterError(f"Kernel start did not return an id: {payload}")
        return str(kernel_id)

    def shutdown_kernel(self, kernel_id: str) -> None:
        try:
            self.request("DELETE", f"/api/kernels/{quote(kernel_id, safe='')}")
        except Exception as exc:
            print(f"[WARN] Could not shut down remote kernel {kernel_id}: {exc}", file=sys.stderr)

    def execute(self, kernel_id: str, code: str) -> None:
        session_id = uuid.uuid4().hex
        ws_url = self._url(
            f"/api/kernels/{quote(kernel_id, safe='')}/channels",
            params={"session_id": session_id},
            ws=True,
        )
        headers = [f"Authorization: token {self.token}"] if self.token else None
        ws = websocket.create_connection(ws_url, timeout=self.timeout, header=headers)
        try:
            msg_id = uuid.uuid4().hex
            ws.send(
                json.dumps(
                    {
                        "header": {
                            "msg_id": msg_id,
                            "username": "remote-edgetpu-export",
                            "session": session_id,
                            "date": iso_utc_now(),
                            "msg_type": "execute_request",
                            "version": "5.3",
                        },
                        "parent_header": {},
                        "metadata": {},
                        "content": {
                            "code": code,
                            "silent": False,
                            "store_history": False,
                            "user_expressions": {},
                            "allow_stdin": False,
                            "stop_on_error": True,
                        },
                        "channel": "shell",
                    },
                    separators=(",", ":"),
                )
            )

            saw_idle = False
            execute_failed = False
            while not saw_idle:
                msg = json.loads(ws.recv())
                if not msg:
                    continue
                parent = msg.get("parent_header") or {}
                if parent.get("msg_id") != msg_id:
                    continue
                msg_type = (msg.get("header") or {}).get("msg_type")
                content = msg.get("content") or {}
                if msg_type == "stream":
                    text = content.get("text", "")
                    if text:
                        print(text, end="", flush=True)
                elif msg_type in {"display_data", "execute_result"}:
                    data = content.get("data") or {}
                    text = data.get("text/plain") or data.get("text/html")
                    if text:
                        if isinstance(text, list):
                            text = "".join(text)
                        print(text, flush=True)
                elif msg_type == "error":
                    execute_failed = True
                    traceback = content.get("traceback") or []
                    if traceback:
                        print("\n".join(strip_ansi(line) for line in traceback), file=sys.stderr)
                    else:
                        print(f"{content.get('ename')}: {content.get('evalue')}", file=sys.stderr)
                elif msg_type == "execute_reply" and content.get("status") == "error":
                    execute_failed = True
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    saw_idle = True

            if execute_failed:
                raise JupyterError("Remote export failed.")
        finally:
            ws.close()


def quote_path(path: str) -> str:
    return "/".join(quote(part, safe="") for part in path.strip("/").split("/") if part)


def join_remote_path(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def strip_ansi(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        if text[i] == "\x1b":
            i += 1
            while i < len(text) and text[i] not in "ABCDEFGHJKSTfimnsulh":
                i += 1
        else:
            out.append(text[i])
        i += 1
    return "".join(out)


def iso_utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"


def rel(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path.resolve()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_simple_yaml(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def final_epoch(results_csv: Path) -> Optional[int]:
    if not results_csv.exists():
        return None
    with results_csv.open(newline="") as handle:
        rows = list(csv.reader(handle))
    for row in reversed(rows):
        if not row:
            continue
        try:
            return int(float(row[0].strip()))
        except ValueError:
            continue
    return None


def find_latest_completed_weights(runs_dir: Path) -> Path:
    candidates: List[Tuple[float, Path, str]] = []
    for weights in runs_dir.glob("detect/*/weights/best.pt"):
        run_dir = weights.parents[1]
        args = parse_simple_yaml(run_dir / "args.yaml")
        expected_epochs = int(float(args.get("epochs", "0") or "0"))
        last_epoch = final_epoch(run_dir / "results.csv")
        if expected_epochs > 0 and last_epoch is not None and last_epoch < expected_epochs:
            reason = f"skipping incomplete {rel(run_dir)}: epoch {last_epoch}/{expected_epochs}"
            candidates.append((-1.0, weights, reason))
            continue
        candidates.append((weights.stat().st_mtime, weights, ""))

    completed = [(mtime, weights) for mtime, weights, reason in candidates if mtime >= 0]
    for _, _, reason in candidates:
        if reason:
            print(f"[INFO] {reason}")
    if not completed:
        raise FileNotFoundError(f"No completed best.pt found under {runs_dir / 'detect'}")
    return max(completed, key=lambda item: item[0])[1]


def iter_dataset_files(dataset_dir: Path) -> Iterable[Path]:
    excluded_suffixes = {".cache"}
    excluded_names = {".DS_Store"}
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in excluded_names or path.suffix in excluded_suffixes:
            continue
        yield path


def build_bundle(weights: Path, dataset: Path, output_dir: Path) -> Path:
    bundle_path = output_dir / "squirrel_edgetpu_export_bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz", dereference=True) as tar:
        tar.add(weights, arcname="best.pt")
        for file_path in iter_dataset_files(dataset):
            tar.add(file_path, arcname=str(Path("yolo") / file_path.relative_to(dataset)))
    return bundle_path


def remote_export_code(remote_dir: str, archive_name: str, imgsz: int) -> str:
    return f"""
from pathlib import Path
import shutil
import tarfile
import time

print("[REMOTE] Python kernel started", flush=True)
archive_name = {archive_name!r}
remote_dir = Path({remote_dir!r})
candidates = [
    Path.cwd() / remote_dir / archive_name,
    Path.home() / remote_dir / archive_name,
]
archive = next((p for p in candidates if p.exists()), None)
if archive is None:
    matches = list(Path.cwd().rglob(archive_name)) + list(Path.home().rglob(archive_name))
    archive = matches[0] if matches else None
if archive is None:
    raise FileNotFoundError(f"Could not find uploaded archive {{archive_name}} from cwd={{Path.cwd()}} home={{Path.home()}}")

print(f"[REMOTE] Using archive {{archive}}", flush=True)
upload_dir = archive.parent
work = upload_dir / "work"
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
with tarfile.open(archive, "r:gz") as tar:
    tar.extractall(work)
print(f"[REMOTE] Extracted bundle into {{work}}", flush=True)

from ultralytics import YOLO

model_path = work / "best.pt"
data_path = work / "yolo" / "dataset.yaml"
original_yaml = data_path.read_text().splitlines()
rewritten_yaml = []
path_written = False
for line in original_yaml:
    if line.strip().startswith("path:"):
        rewritten_yaml.append(f"path: {{work / 'yolo'}}")
        path_written = True
    else:
        rewritten_yaml.append(line)
if not path_written:
    rewritten_yaml.insert(0, f"path: {{work / 'yolo'}}")
data_path.write_text("\\n".join(rewritten_yaml) + "\\n")
print(f"[REMOTE] Exporting {{model_path}} with data={{data_path}}", flush=True)
started = time.time()
model = YOLO(str(model_path))
result = model.export(
    format="edgetpu",
    imgsz={imgsz},
    nms=False,
    batch=1,
    dynamic=False,
    data=str(data_path),
)
print(f"[REMOTE] export() returned {{result!r}} in {{time.time() - started:.1f}}s", flush=True)

matches = sorted(work.rglob({REMOTE_OUTPUT_NAME!r}), key=lambda p: p.stat().st_mtime, reverse=True)
if not matches:
    raise FileNotFoundError(f"Could not find {REMOTE_OUTPUT_NAME} under {{work}}")
compiled = matches[0]
target = upload_dir / {REMOTE_OUTPUT_NAME!r}
shutil.copy2(compiled, target)
print(f"[REMOTE] Copied compiled model to {{target}} ({{target.stat().st_size}} bytes)", flush=True)
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload local YOLO weights/dataset to a remote Jupyter server, export EdgeTPU TFLite, and download it.",
    )
    parser.add_argument("--server", default=os.environ.get("JUPYTER_URL", "http://192.168.1.2:8888/"))
    parser.add_argument("--token", default=os.environ.get("JUPYTER_TOKEN", DEFAULT_JUPYTER_TOKEN))
    parser.add_argument("--weights", type=Path, help="Path to best.pt. Defaults to latest completed runs/detect/*/weights/best.pt.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--remote-base-dir", help="Existing writable Jupyter directory. Defaults to auto-detect, preferring ModelConverter.ipynb's directory.")
    parser.add_argument("--remote-dir", default=f"squirrel-edgetpu-export-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--keep-kernel", action="store_true", help="Do not shut down the temporary remote kernel.")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: pass --token or set JUPYTER_TOKEN.", file=sys.stderr)
        return 2
    if CLIENT_IMPORT_ERROR is not None:
        print(
            f"ERROR: missing Python package {CLIENT_IMPORT_ERROR.name!r}. "
            "Run this through the model_maker_2 uv environment, e.g. "
            "`cd model_maker_2 && uv run python export_edgetpu_remote.py`, "
            "or install project dependencies with `uv sync`.",
            file=sys.stderr,
        )
        return 2

    weights = resolve_repo_path(args.weights) if args.weights else find_latest_completed_weights(REPO_ROOT / "runs")
    dataset = resolve_repo_path(args.dataset)
    output = resolve_repo_path(args.output)
    if not weights.exists():
        raise FileNotFoundError(weights)
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if not (dataset / "dataset.yaml").exists():
        raise FileNotFoundError(dataset / "dataset.yaml")

    print(f"[LOCAL] Weights: {rel(weights)} ({human_size(weights.stat().st_size)})")
    print(f"[LOCAL] Dataset: {rel(dataset)}")
    print(f"[LOCAL] Output:  {rel(output)}")

    try:
        client = JupyterClient(args.server, args.token, timeout=args.timeout)
        with tempfile.TemporaryDirectory(prefix="squirrel-edgetpu-") as tmp:
            tmp_path = Path(tmp)
            print("[LOCAL] Building upload bundle...")
            bundle = build_bundle(weights, dataset, tmp_path)
            print(f"[LOCAL] Bundle: {human_size(bundle.stat().st_size)}")

            if args.remote_base_dir is None:
                remote_base_dir = client.discover_writable_base_dir()
                print(f"[LOCAL] Remote writable base: {remote_base_dir or '/'}")
            else:
                remote_base_dir = args.remote_base_dir.strip("/")
            remote_dir = join_remote_path(remote_base_dir, args.remote_dir)
            remote_archive = join_remote_path(remote_dir, bundle.name)
            print(f"[LOCAL] Uploading bundle to {args.server.rstrip('/')}/files/{remote_archive}")
            client.make_dir(remote_dir)
            client.upload_file(bundle, remote_archive)
            print("[LOCAL] Upload complete.")

            kernel_id = client.start_kernel()
            print(f"[LOCAL] Started remote kernel {kernel_id}")
            try:
                client.execute(kernel_id, remote_export_code(remote_dir, bundle.name, args.imgsz))
            finally:
                if args.keep_kernel:
                    print(f"[LOCAL] Leaving remote kernel running: {kernel_id}")
                else:
                    client.shutdown_kernel(kernel_id)
                    print("[LOCAL] Remote kernel shut down.")

            remote_model = join_remote_path(remote_dir, REMOTE_OUTPUT_NAME)
            print(f"[LOCAL] Downloading {remote_model} -> {rel(output)}")
            client.download_file(remote_model, output)
            print(f"[LOCAL] Downloaded {human_size(output.stat().st_size)}")
            print(f"[LOCAL] sha256: {sha256(output)}")
    except JupyterError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
