import glob
import logging
import subprocess

from constants import MAX_CPU_TEMP, MAX_GPU_TEMP

logger = logging.getLogger("thermal_guard")


def get_gpu_temperature():
    """Returns the maximum GPU temperature in Celsius, or None if unavailable."""
    try:
        gpu_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            timeout=2.0,
        )
        temps = [float(x.strip()) for x in gpu_out.strip().split("\n") if x.strip()]
        return max(temps) if temps else None
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def get_cpu_temperature():
    """Returns the maximum CPU temperature in Celsius from Linux sysfs sensors, or None if unavailable."""
    temps = []
    # 1. Check /sys/class/hwmon
    for p in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
        try:
            with open(p) as f:
                temps.append(float(f.read().strip()) / 1000.0)
        except (OSError, ValueError):
            continue

    # 2. Check /sys/class/thermal
    for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(p) as f:
                temps.append(float(f.read().strip()) / 1000.0)
        except (OSError, ValueError):
            continue

    return max(temps) if temps else None


def check_thermal_guard(max_gpu=MAX_GPU_TEMP, max_cpu=MAX_CPU_TEMP):
    """Monitors hardware temperatures. If GPU or CPU exceeds safety threshold, raises SystemExit."""
    gpu_temp = get_gpu_temperature()
    cpu_temp = get_cpu_temperature()

    if gpu_temp is not None and gpu_temp > max_gpu:
        msg = (
            f"\n{'!' * 60}\n"
            f"[CRITICAL THERMAL ALERT] GPU Temp reached {gpu_temp:.1f}°C (Threshold: {max_gpu:.1f}°C)!\n"
            f"Halting training immediately to protect hardware.\n"
            f"{'!' * 60}\n"
        )
        logger.critical(msg)
        print(msg)
        raise SystemExit(f"EMERGENCY SHUTDOWN: GPU temperature exceeded {max_gpu}°C.")

    if cpu_temp is not None and cpu_temp > max_cpu:
        msg = (
            f"\n{'!' * 60}\n"
            f"[CRITICAL THERMAL ALERT] CPU Temp reached {cpu_temp:.1f}°C (Threshold: {max_cpu:.1f}°C)!\n"
            f"Halting training immediately to protect hardware.\n"
            f"{'!' * 60}\n"
        )
        logger.critical(msg)
        print(msg)
        raise SystemExit(f"EMERGENCY SHUTDOWN: CPU temperature exceeded {max_cpu}°C.")

    return gpu_temp, cpu_temp
