"""Small, dependency-free hardware inventory for explanatory UI labels."""

from dataclasses import dataclass
import platform
import re
import shlex
import subprocess
import sys


@dataclass(frozen=True)
class HardwareInfo:
    cpu: str = ""
    nvidia_gpus: tuple[str, ...] = ()
    amd_gpus: tuple[str, ...] = ()


def _run(args, timeout=2) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def _cpu_name() -> str:
    if sys.platform != "win32":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read(), re.M)
            if match:
                return match.group(1).strip()
        except OSError:
            pass
    return platform.processor().strip()


def _linux_gpus() -> tuple[list[str], list[str]]:
    nvidia, amd = [], []
    for line in _run(["lspci", "-mm"]).splitlines():
        try:
            fields = shlex.split(line)
        except ValueError:
            continue
        if len(fields) < 4 or not re.search(r"VGA|3D|Display", fields[1], re.I):
            continue
        vendor, product = fields[2], fields[3]
        product = re.sub(r"^[^[]+\[([^]]+)\]$", r"\1", product).strip()
        if "nvidia" in vendor.lower():
            nvidia.append(product)
        elif "amd" in vendor.lower() or "advanced micro" in vendor.lower():
            amd.append(product)
    return nvidia, amd


def _windows_gpus() -> tuple[list[str], list[str]]:
    output = _run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
    ])
    names = [line.strip() for line in output.splitlines() if line.strip()]
    return ([name for name in names if "nvidia" in name.lower()],
            [name for name in names if re.search(r"amd|radeon", name, re.I)])


def detect_hardware() -> HardwareInfo:
    nvidia, amd = _windows_gpus() if sys.platform == "win32" else _linux_gpus()
    return HardwareInfo(_cpu_name(), tuple(nvidia), tuple(amd))


def execution_device_labels(hardware: HardwareInfo) -> dict[str, str]:
    """Only return runnable backends; detected-but-unusable GPUs belong in the note."""
    nvidia = hardware.nvidia_gpus[0] if hardware.nvidia_gpus else "CUDA"
    cpu = re.sub(r"\s+\d+-Core Processor$", "", hardware.cpu) or "processador"
    return {
        "auto": ("Automático — prioriza a GPU NVIDIA" if hardware.nvidia_gpus
                 else "Automático — usa a CPU"),
        "cuda": f"GPU NVIDIA — {nvidia}",
        "cpu": f"CPU — {cpu}",
    }


def execution_hardware_note(hardware: HardwareInfo) -> str:
    if hardware.amd_gpus:
        amd = hardware.amd_gpus[0]
        return (f"GPU AMD detectada: {amd}. Ela fica livre: esta instalação usa CUDA na NVIDIA "
                "ou CPU; AMD exigiria um ambiente ROCm separado e não se soma à RTX.")
    return "Automático usa a NVIDIA quando disponível; CPU é a alternativa independente."
