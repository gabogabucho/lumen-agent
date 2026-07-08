"""RSS measurement smoke test for the perfil-core acceptance criterion.

Spawns fresh Python subprocesses that import Lumen's headless core runtime
and reports their idle RSS. Acceptance (spec sdd/perfil-core/spec, domain
performance): the core profile (no litellm import) must stay under
RSS_LIMIT_MIB. A litellm scenario is measured alongside for comparison.

Usage:
    python scripts/measure_rss.py            # human-readable report
    python scripts/measure_rss.py --check    # exit 1 if core scenario exceeds limit

No third-party dependencies: reads /proc/self/status on Linux and uses
GetProcessMemoryInfo via ctypes on Windows.
"""

from __future__ import annotations

import subprocess
import sys

RSS_LIMIT_MIB = 100.0

_RSS_SNIPPET = r"""
import sys

def rss_mib():
    if sys.platform == "win32":
        import ctypes, ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        kernel32 = ctypes.windll.kernel32
        fn = getattr(kernel32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
        fn.restype = wt.BOOL
        current_process = wt.HANDLE(-1)  # GetCurrentProcess() pseudo-handle
        if not fn(current_process, ctypes.byref(pmc), pmc.cb):
            raise ctypes.WinError()
        return pmc.WorkingSetSize / (1024 * 1024)
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("unsupported platform")
"""

SCENARIOS = {
    "core (headless, no litellm)": (
        "import lumen.core.brain, lumen.core.distiller, lumen.core.llm_client\n"
        "assert 'litellm' not in sys.modules, 'litellm leaked into the core profile'\n"
    ),
    "litellm imported (full-profile comparison)": "import litellm\n",
}


def measure(scenario_code: str) -> float:
    script = _RSS_SNIPPET + scenario_code + "print(rss_mib())\n"
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return float(out.stdout.strip().splitlines()[-1])


def main() -> int:
    check = "--check" in sys.argv
    failed = False
    for name, code in SCENARIOS.items():
        rss = measure(code)
        is_core = name.startswith("core")
        verdict = ""
        if is_core:
            ok = rss < RSS_LIMIT_MIB
            failed |= not ok
            verdict = f"  [{'OK' if ok else 'FAIL'} limit {RSS_LIMIT_MIB:.0f} MiB]"
        print(f"{name:45s} {rss:8.1f} MiB{verdict}")
    return 1 if (check and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
