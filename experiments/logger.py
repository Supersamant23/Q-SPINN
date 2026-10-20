"""
experiments/logger.py
======================
Shared structured logging utility for Q-SPINN experiments.

Tees all stdout output simultaneously to:
  1. The terminal (live, coloured prefix)
  2. A timestamped .log file in the experiment's output directory

Usage in a train.py:
    from logger import ExperimentLogger
    log = ExperimentLogger(OUT_DIR / "train.log", experiment_name="Exp A")
    log.section("Loading data")
    log.info("Loaded 2M rows")
    log.metric("val_f1", 0.523)
    log.epoch(5, 25, loss=0.312, val_f1=0.523, spike_rate=0.037, eta_s=120)
    log.done()
"""

import sys
import time
from datetime import datetime
from pathlib import Path
import io

# Force UTF-8 output on Windows (avoids CP1252 UnicodeEncodeError for box-draw chars)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ── ANSI colour codes (stripped in the file, kept in terminal) ─────────────────
_RESET  = "\033[0m\033[37m"  # Explicitly reset to White to fix black-on-black invisible text
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_DIM    = "\033[90m"  # Changed from \033[2m to Bright Black (Dark Gray) for Windows compatibility
_BOLD   = "\033[1m"


def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r'\033\[[0-9;]*m', '', s)


class ExperimentLogger:
    """
    Structured logger that writes simultaneously to stdout and a log file.

    Log file format is plain-text with timestamps — no ANSI codes.
    Terminal output includes colour and a live progress bar.
    """

    WIDTH = 68   # total separator width

    def __init__(self, log_path: Path, experiment_name: str = "Experiment"):
        self.log_path        = log_path
        self.experiment_name = experiment_name
        self._start_time     = time.time()
        self._file           = open(log_path, "w", encoding="utf-8", buffering=1)

        self._banner(experiment_name)

    # ── internal helpers ──────────────────────────────────────────────────────
    def _ts(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _elapsed(self) -> str:
        s = int(time.time() - self._start_time)
        h, rem = divmod(s, 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _write(self, terminal_line: str, file_line: str | None = None):
        """Print to terminal (with ANSI) and write clean version to file."""
        try:
            print(terminal_line)
        except (OSError, UnicodeEncodeError):
            pass
        try:
            clean = _strip_ansi(file_line if file_line is not None else terminal_line)
            self._file.write(clean + "\n")
            self._file.flush()
        except Exception:
            pass

    def _sep(self, char: str = "─") -> str:
        return char * self.WIDTH

    # ── public API ────────────────────────────────────────────────────────────
    def _banner(self, name: str):
        ts = self._ts()
        self._write(
            f"{_BOLD}{_CYAN}{'═' * self.WIDTH}{_RESET}\n"
            f"{_BOLD}{_CYAN}  Q-SPINN — {name}{_RESET}\n"
            f"{_DIM}  Started : {ts}{_RESET}\n"
            f"{_DIM}  Log     : {self.log_path}{_RESET}\n"
            f"{_BOLD}{_CYAN}{'═' * self.WIDTH}{_RESET}",
            file_line=(
                f"{'=' * self.WIDTH}\n"
                f"  Q-SPINN — {name}\n"
                f"  Started : {ts}\n"
                f"  Log     : {self.log_path}\n"
                f"{'=' * self.WIDTH}"
            ),
        )

    def section(self, title: str):
        """Print a clearly visible section header."""
        ts = self._ts()
        self._write(
            f"\n{_BOLD}{_CYAN}{self._sep('─')}{_RESET}\n"
            f"{_BOLD}{_CYAN}  [{ts}]  {title}{_RESET}\n"
            f"{_BOLD}{_CYAN}{self._sep('─')}{_RESET}",
            file_line=(
                f"\n{self._sep('─')}\n"
                f"  [{ts}]  {title}\n"
                f"{self._sep('─')}"
            ),
        )

    def info(self, msg: str):
        """General informational line."""
        ts = self._ts()
        self._write(
            f"  {_DIM}[{ts}]{_RESET}  {msg}",
            file_line=f"  [{ts}]  {msg}",
        )

    def metric(self, key: str, value, unit: str = ""):
        """Highlight a single key metric."""
        ts  = self._ts()
        val = f"{value:.4f}" if isinstance(value, float) else str(value)
        self._write(
            f"  {_DIM}[{ts}]{_RESET}  {_GREEN}{key}{_RESET} = {_BOLD}{val}{_RESET}{unit}",
            file_line=f"  [{ts}]  {key} = {val}{unit}",
        )

    def warn(self, msg: str):
        ts = self._ts()
        self._write(
            f"  {_YELLOW}[{ts}]  ⚠  {msg}{_RESET}",
            file_line=f"  [{ts}]  WARN  {msg}",
        )

    def epoch(
        self,
        ep: int,
        total: int,
        *,
        loss: float,
        val_f1: float,
        extra: dict | None = None,
        eta_s: float = 0.0,
    ):
        """
        Compact, fixed-width epoch progress line.

        Example terminal output:
          [2026-09-18 20:14:01]  Ep  5/25  ▓▓▓▓▓░░░░░░░░░░░░░░░  20%  loss=0.3124  F1=0.523  ETA 04:12
        """
        ts      = self._ts()
        pct     = ep / total
        filled  = int(pct * 20)
        bar_t   = _GREEN + "▓" * filled + _DIM + "░" * (20 - filled) + _RESET
        bar_f   = "█" * filled + "░" * (20 - filled)

        eta_str = ""
        if eta_s > 0:
            em, es  = divmod(int(eta_s), 60)
            eta_str = f"  ETA {em:02d}:{es:02d}"

        extras_t = ""
        extras_f = ""
        if extra:
            for k, v in extra.items():
                fv = f"{v:.4f}" if isinstance(v, float) else str(v)
                extras_t += f"  {_DIM}{k}{_RESET}={_BOLD}{fv}{_RESET}"
                extras_f += f"  {k}={fv}"

        line_t = (f"  {_DIM}[{ts}]{_RESET}"
                  f"  Ep {ep:3d}/{total}"
                  f"  {bar_t}  {int(pct*100):3d}%"
                  f"  {_BOLD}loss={loss:.4f}{_RESET}"
                  f"  {_GREEN}F1={val_f1:.3f}{_RESET}"
                  f"{extras_t}{eta_str}")

        line_f = (f"  [{ts}]"
                  f"  Ep {ep:3d}/{total}"
                  f"  {bar_f}  {int(pct*100):3d}%"
                  f"  loss={loss:.4f}"
                  f"  F1={val_f1:.3f}"
                  f"{extras_f}{eta_str}")

        self._write(line_t, file_line=line_f)

    def stage(self, name: str, result: dict):
        """Print a stage completion summary (Stage A / B / C)."""
        ts = self._ts()
        pairs_t = "  ".join(
            f"{_DIM}{k}{_RESET}={_BOLD}{v:.4f}{_RESET}" if isinstance(v, float)
            else f"{_DIM}{k}{_RESET}={v}"
            for k, v in result.items()
        )
        pairs_f = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in result.items()
        )
        self._write(
            f"  {_DIM}[{ts}]{_RESET}  {_GREEN}✓ {name}{_RESET}  {pairs_t}",
            file_line=f"  [{ts}]  ✓ {name}  {pairs_f}",
        )

    def table(self, headers: list[str], rows: list[tuple], title: str = ""):
        """Print a formatted results table."""
        if title:
            self.section(title)

        # compute column widths
        col_w = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_w[i] = max(col_w[i], len(str(cell)))

        def fmt_row(cells, bold=False):
            parts = [str(c).ljust(col_w[i]) for i, c in enumerate(cells)]
            return "  " + "  |  ".join(parts)

        sep_line = "  " + "-" * (sum(col_w) + 5 * (len(headers) - 1) + 2)
        header_t = f"{_BOLD}{fmt_row(headers)}{_RESET}"
        header_f = fmt_row(headers)

        lines_t, lines_f = [header_t, sep_line], [header_f, sep_line]
        for row in rows:
            lines_t.append(fmt_row(row))
            lines_f.append(fmt_row(row))

        for lt, lf in zip(lines_t, lines_f):
            self._write(lt, file_line=lf)

    def done(self, summary: dict | None = None):
        """Print the final completion banner."""
        elapsed   = self._elapsed()
        ts        = self._ts()
        wall_time = time.time() - self._start_time

        lines_t = [
            f"\n{_BOLD}{_GREEN}{'═' * self.WIDTH}{_RESET}",
            f"{_BOLD}{_GREEN}  ✅  {self.experiment_name} — COMPLETE{_RESET}",
            f"{_DIM}  Finished : {ts}  |  Wall time : {elapsed}{_RESET}",
        ]
        lines_f = [
            f"\n{'=' * self.WIDTH}",
            f"  ✅  {self.experiment_name} — COMPLETE",
            f"  Finished : {ts}  |  Wall time : {elapsed}",
        ]

        if summary:
            for k, v in summary.items():
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                lines_t.append(f"  {_GREEN}{k:<24}{_RESET} {_BOLD}{val}{_RESET}")
                lines_f.append(f"  {k:<24} {val}")

        lines_t.append(f"{_BOLD}{_GREEN}{'═' * self.WIDTH}{_RESET}")
        lines_f.append(f"{'=' * self.WIDTH}")

        for lt, lf in zip(lines_t, lines_f):
            self._write(lt, file_line=lf)

        self._file.close()

    def close(self):
        if not self._file.closed:
            self._file.close()

    # ── context manager support ───────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
