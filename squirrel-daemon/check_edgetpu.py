#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import grp
import importlib.metadata as metadata
import os
import pwd
import shutil
import stat
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple


CORAL_USB_IDS = {
    ("18d1", "9302"): "Google Coral USB Accelerator",
    ("1a6e", "089a"): "Coral USB Accelerator bootloader",
}


def print_section(title: str) -> None:
    print()
    print(f"== {title} ==")


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def run_command(args: list[str]) -> tuple[int, str]:
    if shutil.which(args[0]) is None:
        return 127, f"{args[0]} not found"
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as exc:
        return 126, f"{' '.join(args)} failed: {exc}"
    return proc.returncode, proc.stdout.strip()


def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def user_and_groups() -> None:
    uid = os.getuid()
    gid = os.getgid()
    try:
        user = pwd.getpwuid(uid).pw_name
    except KeyError:
        user = str(uid)
    try:
        group = grp.getgrgid(gid).gr_name
    except KeyError:
        group = str(gid)

    groups = []
    for group_id in os.getgroups():
        try:
            groups.append(grp.getgrgid(group_id).gr_name)
        except KeyError:
            groups.append(str(group_id))
    print(f"user={user} uid={uid} primary_group={group} gid={gid}")
    print(f"groups={','.join(groups) if groups else '(none)'}")


def iter_sysfs_usb_devices() -> Iterable[dict[str, object]]:
    root = Path("/sys/bus/usb/devices")
    if not root.exists():
        return
    for entry in sorted(root.iterdir()):
        vendor = read_text(entry / "idVendor")
        product = read_text(entry / "idProduct")
        if not vendor or not product:
            continue
        busnum = read_text(entry / "busnum")
        devnum = read_text(entry / "devnum")
        node = None
        if busnum and devnum:
            try:
                node = Path(f"/dev/bus/usb/{int(busnum):03d}/{int(devnum):03d}")
            except ValueError:
                node = None
        yield {
            "sysfs": entry,
            "vendor": vendor.lower(),
            "product": product.lower(),
            "busnum": busnum,
            "devnum": devnum,
            "node": node,
        }


def describe_device_node(path: Path) -> str:
    try:
        st = path.stat()
    except OSError as exc:
        return f"{path} stat failed: {exc}"

    try:
        owner = pwd.getpwuid(st.st_uid).pw_name
    except KeyError:
        owner = str(st.st_uid)
    try:
        group = grp.getgrgid(st.st_gid).gr_name
    except KeyError:
        group = str(st.st_gid)

    access = []
    if os.access(path, os.R_OK):
        access.append("read")
    if os.access(path, os.W_OK):
        access.append("write")
    access_text = ",".join(access) if access else "no read/write"
    return f"{path} {stat.filemode(st.st_mode)} {owner}:{group} access={access_text}"


def print_usb_info() -> list[Path]:
    print_section("USB")
    code, output = run_command(["lsusb"])
    if output:
        print(output)
    if code not in (0, 127):
        print(f"lsusb exited with {code}")

    coral_nodes: list[Path] = []
    print()
    print("Coral devices from sysfs:")
    found = False
    for device in iter_sysfs_usb_devices():
        key = (str(device["vendor"]), str(device["product"]))
        label = CORAL_USB_IDS.get(key)
        if not label:
            continue
        found = True
        node = device["node"]
        print(
            f"{label} {key[0]}:{key[1]} "
            f"bus={device['busnum']} dev={device['devnum']} sysfs={device['sysfs']}"
        )
        if isinstance(node, Path):
            coral_nodes.append(node)
            print(f"  {describe_device_node(node)}")
    if not found:
        print("No known Coral USB device IDs found in /sys/bus/usb/devices.")
    return coral_nodes


def print_runtime_info() -> None:
    print_section("Python and Packages")
    print(f"python={sys.executable}")
    print(f"version={sys.version.split()[0]}")
    for name in ("pycoral", "tflite-runtime", "ultralytics", "numpy"):
        print(f"{name}={package_version(name)}")

    print_section("Edge TPU Shared Library")
    found = ctypes.util.find_library("edgetpu")
    print(f"find_library('edgetpu')={found or 'not found'}")
    candidates = [candidate for candidate in (found, "libedgetpu.so.1") if candidate]
    loaded = False
    for candidate in candidates:
        try:
            ctypes.CDLL(candidate)
        except OSError as exc:
            print(f"FAIL loading {candidate}: {exc}")
        else:
            print(f"PASS loaded {candidate}")
            loaded = True
            break
    if not loaded:
        print("No Edge TPU runtime library could be loaded by ctypes.")


def print_process_info(device_nodes: list[Path]) -> None:
    print_section("Processes")
    code, output = run_command(["ps", "-eo", "pid,user,stat,comm,args"])
    if code != 0:
        print(output or f"ps exited with {code}")
    else:
        lines = output.splitlines()
        if lines:
            print(lines[0])
            needles = ("squirrel", "python", "uv", "flask", "tflite", "edge", "ultralytics")
            for line in lines[1:]:
                if any(needle in line.lower() for needle in needles):
                    print(line)

    if not device_nodes:
        return

    print()
    print("Device users from fuser:")
    for node in device_nodes:
        code, output = run_command(["fuser", "-v", str(node)])
        if code == 127:
            print(output)
            return
        if output:
            print(output)
        else:
            print(f"{node}: no process reported by fuser")


def import_edgetpu_helpers() -> Optional[Tuple[Any, Any, Any]]:
    try:
        from pycoral.utils.edgetpu import list_edge_tpus, load_edgetpu_delegate, make_interpreter
    except Exception as exc:
        print(f"FAIL importing pycoral Edge TPU helpers: {exc}")
        traceback.print_exc(file=sys.stdout)
        return None
    return list_edge_tpus, load_edgetpu_delegate, make_interpreter


def test_delegate_open(device: Optional[str]) -> bool:
    print_section("PyCoral Device and Delegate")
    helpers = import_edgetpu_helpers()
    if helpers is None:
        return False
    list_edge_tpus, load_edgetpu_delegate, _make_interpreter = helpers

    try:
        devices = list_edge_tpus()
    except Exception as exc:
        print(f"FAIL list_edge_tpus(): {exc}")
        traceback.print_exc(file=sys.stdout)
        return False
    print(f"list_edge_tpus={devices}")

    options = {"device": device} if device else {}
    try:
        load_edgetpu_delegate(options)
    except Exception as exc:
        print(f"FAIL load_edgetpu_delegate({options}): {exc}")
        traceback.print_exc(file=sys.stdout)
        return False
    print(f"PASS load_edgetpu_delegate({options})")
    return True


def test_model_interpreter(model_path: Path, device: Optional[str]) -> bool:
    print_section("Model Interpreter Smoke Test")
    print(f"model={model_path}")
    if device:
        print(f"device={device}")
    if not model_path.exists():
        print(f"FAIL model file does not exist: {model_path}")
        return False

    helpers = import_edgetpu_helpers()
    if helpers is None:
        return False
    _list_edge_tpus, _load_edgetpu_delegate, make_interpreter = helpers

    try:
        if device:
            interpreter = make_interpreter(str(model_path), device)
        else:
            interpreter = make_interpreter(str(model_path))
        interpreter.allocate_tensors()
        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
    except Exception as exc:
        print(f"FAIL creating/allocating Edge TPU interpreter: {exc}")
        traceback.print_exc(file=sys.stdout)
        return False

    def shape_value(item: dict[str, object]) -> object:
        shape = item.get("shape")
        return shape.tolist() if hasattr(shape, "tolist") else shape

    print("PASS Edge TPU interpreter allocated")
    print(f"inputs={[(item.get('name'), shape_value(item)) for item in inputs]}")
    print(f"outputs={[(item.get('name'), shape_value(item)) for item in outputs]}")
    return True


def print_hints() -> None:
    print_section("Hints")
    print("If list_edge_tpus finds the Coral but load_edgetpu_delegate fails, the issue is below the model/Ultralytics layer.")
    print("If the Coral node exists but access is not read,write, fix udev/group permissions or test with sudo once.")
    print("If libedgetpu cannot load, install the Edge TPU runtime package on the Pi.")
    print("If fuser or ps shows another Python/squirrel process using the USB node, stop that process and retry.")
    print("If the device is 1a6e:089a only, it is still in bootloader mode and the runtime did not initialize it.")


def parse_args() -> argparse.Namespace:
    default_model = Path(__file__).resolve().parent / "best_full_integer_quant_edgetpu.tflite"
    parser = argparse.ArgumentParser(description="Diagnose Coral Edge TPU delegate loading.")
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--device", default=None, help="Optional PyCoral device string, for example usb:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser()
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path

    print_section("User")
    user_and_groups()
    device_nodes = print_usb_info()
    print_runtime_info()
    print_process_info(device_nodes)
    delegate_ok = test_delegate_open(args.device)
    interpreter_ok = test_model_interpreter(model_path, args.device) if delegate_ok else False
    print_hints()
    return 0 if delegate_ok and interpreter_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
