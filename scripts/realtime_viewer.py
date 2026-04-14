"""
realtime_viewer.py — Interactive EEG Pipeline Viewer
=====================================================
Stand-alone matplotlib script.  No Streamlit / Dash required.

Usage
-----
From the project root::

    python scripts/realtime_viewer.py                           # auto-discover
    python scripts/realtime_viewer.py --raw  data/processed/real_awake7.fif \\
                                      --filt data/processed/real_awake_final_clean.fif

Controls
--------
  ← / →  or  A / D   scroll one window left / right
  - / +                narrow / widen the time window
  ↑ / ↓               increase / decrease displayed channel count
  P                    toggle PSD side panel
  S                    save current view to PNG
  Q / Esc             quit
  Mouse drag slider   scroll to any position
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # works on macOS/Linux without extra deps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy import signal as sp_signal
import mne

mne.set_log_level("ERROR")

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ── Constants ─────────────────────────────────────────────────────────────────
DBS_FREQ_DEFAULT = 7.0
WINDOW_SEC_DEFAULT = 10.0
MAX_CH_DISPLAY = 8
SCALE_DEFAULT = 60.0       # µV per row
FIG_SIZE = (16, 9)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_fif(path: str):
    """Return (data_µV, ch_names, sfreq, n_samples)."""
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    data = raw.get_data() * 1e6   # V → µV
    return data, raw.ch_names, float(raw.info["sfreq"])


def discover_fif(proc_dir: pathlib.Path) -> dict[str, str]:
    """Return {display_name: abs_path} for all .fif files in proc_dir."""
    result = {}
    for p in sorted(proc_dir.glob("*.fif")):
        name = p.stem.replace("_", " ")
        result[name] = str(p)
    return result


# ── PSD helper ────────────────────────────────────────────────────────────────

def psd_segment(data_1d: np.ndarray, sfreq: float, fmax: float = 80.0):
    nperseg = min(512, len(data_1d))
    f, p = sp_signal.welch(data_1d, fs=sfreq, nperseg=nperseg)
    mask = (f >= 0.5) & (f <= fmax)
    return f[mask], 10 * np.log10(p[mask] + 1e-30)


# ── Viewer class ──────────────────────────────────────────────────────────────

class EEGViewer:
    RAW_COLOR   = "#888888"
    FILT_COLOR  = "#1a6faf"
    DIFF_COLOR  = "#cc4444"

    def __init__(
        self,
        raw_data:   np.ndarray,
        filt_data:  np.ndarray,
        ch_names:   list[str],
        sfreq:      float,
        dbs_freq:   float = DBS_FREQ_DEFAULT,
        window_sec: float = WINDOW_SEC_DEFAULT,
        scale_uv:   float = SCALE_DEFAULT,
        raw_label:  str = "Raw",
        filt_label: str = "Filtered",
    ):
        self.raw    = raw_data
        self.filt   = filt_data
        self.names  = ch_names
        self.sfreq  = sfreq
        self.dbs_f  = dbs_freq
        self.win    = window_sec
        self.scale  = scale_uv
        self.t0     = 0.0
        self.n_ch_show = min(MAX_CH_DISPLAY, len(ch_names))
        self.ch_offset = 0          # first channel index shown
        self.show_psd  = True
        self.show_diff = False

        self.n_ch, self.n_samp = raw_data.shape
        self.total_sec = self.n_samp / sfreq

        self.raw_label  = raw_label
        self.filt_label = filt_label

        self._build_figure()
        self._draw()
        self._connect_keys()

    # ── Figure layout ─────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=FIG_SIZE, facecolor="#1c1c1c")
        self.fig.canvas.manager.set_window_title("EEG Pipeline Viewer")

        # Main grid: [traces | psd]  [slider row]  [button row]
        outer = gridspec.GridSpec(
            3, 1, figure=self.fig,
            height_ratios=[10, 0.7, 0.5],
            hspace=0.08, left=0.07, right=0.97, top=0.94, bottom=0.04,
        )

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0],
            width_ratios=[4, 1] if self.show_psd else [1, 0],
            wspace=0.04,
        )
        self.ax_traces = self.fig.add_subplot(inner[0])
        self.ax_psd    = self.fig.add_subplot(inner[1])

        # Slider
        ax_slider = self.fig.add_subplot(outer[1])
        ax_slider.set_facecolor("#2a2a2a")
        self.slider = Slider(
            ax=ax_slider,
            label="t₀ (s)",
            valmin=0.0,
            valmax=max(0.0, self.total_sec - self.win),
            valinit=0.0,
            valstep=self.win / 4,
            color="#1a6faf",
        )
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")
        self.slider.on_changed(self._on_slider)

        # Button row
        btn_ax = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer[2], wspace=0.05,
        )
        self.btn_prev = Button(
            self.fig.add_subplot(btn_ax[0]), "◀ Prev",
            color="#2a2a2a", hovercolor="#3a3a3a",
        )
        self.btn_next = Button(
            self.fig.add_subplot(btn_ax[1]), "Next ▶",
            color="#2a2a2a", hovercolor="#3a3a3a",
        )
        self.btn_wider = Button(
            self.fig.add_subplot(btn_ax[2]), "+ Window",
            color="#2a2a2a", hovercolor="#3a3a3a",
        )
        self.btn_narrower = Button(
            self.fig.add_subplot(btn_ax[3]), "− Window",
            color="#2a2a2a", hovercolor="#3a3a3a",
        )
        for btn in (self.btn_prev, self.btn_next, self.btn_wider, self.btn_narrower):
            btn.label.set_color("white")

        self.btn_prev.on_clicked(lambda _: self._scroll(-1))
        self.btn_next.on_clicked(lambda _: self._scroll(+1))
        self.btn_wider.on_clicked(lambda _: self._resize_window(+5))
        self.btn_narrower.on_clicked(lambda _: self._resize_window(-5))

        for ax in self.fig.axes:
            for spine in ax.spines.values():
                spine.set_color("#444444")

    def _connect_keys(self):
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self):
        self._draw_traces()
        self._draw_psd()
        self._update_title()
        self.fig.canvas.draw_idle()

    def _draw_traces(self):
        ax = self.ax_traces
        ax.cla()
        ax.set_facecolor("#1c1c1c")

        i0 = int(self.t0 * self.sfreq)
        i1 = min(i0 + int(self.win * self.sfreq), self.n_samp)
        t_rel = np.arange(i1 - i0) / self.sfreq + self.t0

        ch_end = min(self.ch_offset + self.n_ch_show, self.n_ch)
        ch_range = list(range(self.ch_offset, ch_end))

        ytick_pos, ytick_lbl = [], []

        for row, ci in enumerate(ch_range):
            offset = (len(ch_range) - 1 - row) * self.scale

            raw_seg  = self.raw[ci, i0:i1]
            filt_seg = self.filt[ci, i0:i1]

            # Normalize to ±scale/2 for tidy stacking
            def _norm(x):
                pk = np.abs(x).max()
                return x / (pk + 1e-30) * (self.scale * 0.45)

            ax.plot(t_rel, _norm(raw_seg)  + offset,
                    color=self.RAW_COLOR,  lw=0.7, alpha=0.65)
            ax.plot(t_rel, _norm(filt_seg) + offset,
                    color=self.FILT_COLOR, lw=1.0)

            if self.show_diff:
                diff = raw_seg - filt_seg
                ax.fill_between(
                    t_rel,
                    offset, _norm(diff) + offset,
                    color=self.DIFF_COLOR, alpha=0.30,
                )

            ytick_pos.append(offset)
            ytick_lbl.append(self.names[ci])

        # DBS harmonic lines
        for k in range(1, 20):
            h_t = self.t0 + np.arange(0, self.win, 1.0 / self.dbs_f)
            # Draw very faint vertical ticks at each DBS cycle
            # (only first harmonic to keep it clean)
            if k == 1:
                for ht in h_t:
                    if ht <= self.t0 + self.win:
                        ax.axvline(ht, color="#ff8c00", lw=0.3, alpha=0.25)
                break

        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_lbl, fontsize=7.5, color="white")
        ax.tick_params(axis="x", colors="white", labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(self.t0, self.t0 + self.win)
        ax.set_xlabel("Time (s)", color="white", fontsize=9)
        ax.spines[:].set_color("#444444")
        ax.set_facecolor("#1c1c1c")

        # Legend
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=self.RAW_COLOR,  lw=1.2, label=self.raw_label),
            Line2D([0], [0], color=self.FILT_COLOR, lw=1.4, label=self.filt_label),
        ]
        if self.show_diff:
            handles.append(
                Line2D([0], [0], color=self.DIFF_COLOR, lw=1.2, label="Residual (raw−filt)")
            )
        ax.legend(
            handles=handles, loc="upper right", fontsize=7.5,
            facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white",
        )

    def _draw_psd(self):
        ax = self.ax_psd
        ax.cla()
        ax.set_facecolor("#1c1c1c")

        if not self.show_psd:
            ax.set_visible(False)
            return
        ax.set_visible(True)

        i0 = int(self.t0 * self.sfreq)
        i1 = min(i0 + int(self.win * self.sfreq), self.n_samp)

        ch_end = min(self.ch_offset + self.n_ch_show, self.n_ch)
        ch_range = list(range(self.ch_offset, ch_end))

        # Average PSD over displayed channels
        raw_mean  = self.raw [ch_range, i0:i1].mean(axis=0)
        filt_mean = self.filt[ch_range, i0:i1].mean(axis=0)

        fr, pr = psd_segment(raw_mean,  self.sfreq)
        ff, pf = psd_segment(filt_mean, self.sfreq)

        ax.fill_betweenx(fr, pr, pf,
                         where=pr > pf,
                         color=self.RAW_COLOR,   alpha=0.15, interpolate=True)
        ax.plot(pr, fr, color=self.RAW_COLOR,  lw=0.8, alpha=0.7, label=self.raw_label)
        ax.plot(pf, ff, color=self.FILT_COLOR, lw=1.2, label=self.filt_label)

        for k in range(1, 15):
            h = k * self.dbs_f
            if h > fr[-1]:
                break
            ax.axhline(h, color="#ff8c00", lw=0.5, ls=":", alpha=0.75,
                       label="DBS harmonic" if k == 1 else "")

        ax.set_ylim(0.5, 80)
        ax.set_ylabel("Frequency (Hz)", color="white", fontsize=7)
        ax.set_xlabel("PSD (dB)", color="white", fontsize=7)
        ax.tick_params(colors="white", labelsize=6)
        ax.spines[:].set_color("#444444")
        ax.set_title("PSD (window)", color="white", fontsize=7.5)

    def _update_title(self):
        i0 = int(self.t0 * self.sfreq)
        i1 = min(i0 + int(self.win * self.sfreq), self.n_samp)
        ch_range = list(range(self.ch_offset,
                               min(self.ch_offset + self.n_ch_show, self.n_ch)))
        diff_rms = float(np.sqrt(((self.raw[ch_range, i0:i1] -
                                    self.filt[ch_range, i0:i1]) ** 2).mean()) * 1e6)
        self.fig.suptitle(
            f"EEG Pipeline Viewer  |  "
            f"t = {self.t0:.1f} – {self.t0 + self.win:.1f} s  |  "
            f"Residual RMS = {diff_rms:.2f} µV  |  "
            f"DBS f₀ = {self.dbs_f:.1f} Hz  |  "
            f"[← →] scroll   [+/-] window   [P] PSD   [D] diff   [S] save   [Q] quit",
            color="white", fontsize=8.5, y=0.985,
        )
        self.ax_traces.set_facecolor("#1c1c1c")
        self.fig.set_facecolor("#1c1c1c")

    # ── Interaction handlers ───────────────────────────────────────────────────

    def _scroll(self, direction: int):
        step = self.win * 0.5
        new_t0 = np.clip(self.t0 + direction * step, 0.0,
                         max(0.0, self.total_sec - self.win))
        if new_t0 != self.t0:
            self.t0 = new_t0
            self.slider.set_val(self.t0)
            self._draw()

    def _resize_window(self, delta_sec: float):
        new_win = np.clip(self.win + delta_sec, 2.0, min(60.0, self.total_sec))
        if new_win != self.win:
            self.win = new_win
            # Update slider max
            self.slider.valmax = max(0.0, self.total_sec - self.win)
            self.t0 = min(self.t0, self.slider.valmax)
            self.slider.set_val(self.t0)
            self._draw()

    def _on_slider(self, val):
        self.t0 = float(val)
        self._draw()

    def _on_key(self, event):
        key = event.key
        if key in ("left", "a"):
            self._scroll(-1)
        elif key in ("right", "d"):
            self._scroll(+1)
        elif key in ("up",):
            self.n_ch_show = min(self.n_ch_show + 1, self.n_ch)
            self._draw()
        elif key in ("down",):
            self.n_ch_show = max(1, self.n_ch_show - 1)
            self._draw()
        elif key in ("+", "="):
            self._resize_window(+5)
        elif key in ("-", "_"):
            self._resize_window(-5)
        elif key.lower() == "p":
            self.show_psd = not self.show_psd
            self._draw()
        elif key.lower() == "d":
            self.show_diff = not self.show_diff
            self._draw()
        elif key.lower() == "s":
            out = pathlib.Path("eeg_viewer_snapshot.png")
            self.fig.savefig(str(out), dpi=150, facecolor=self.fig.get_facecolor())
            print(f"Saved: {out}")
        elif key in ("q", "escape"):
            plt.close(self.fig)


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive EEG Pipeline Viewer (matplotlib)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--raw",  default=None,
                   help="Path to raw/contaminated .fif file")
    p.add_argument("--filt", default=None,
                   help="Path to filtered/clean .fif file")
    p.add_argument("--processed-dir", default="data/processed",
                   help="Directory to scan for .fif files when --raw/--filt not given")
    p.add_argument("--dbs-freq", type=float, default=DBS_FREQ_DEFAULT,
                   help="DBS fundamental frequency in Hz (default: 7)")
    p.add_argument("--window", type=float, default=WINDOW_SEC_DEFAULT,
                   help="Initial time-window length in seconds (default: 10)")
    p.add_argument("--scale", type=float, default=SCALE_DEFAULT,
                   help="Per-channel amplitude scale in µV (default: 60)")
    return p.parse_args()


def _pick_default_pair(fif_map: dict[str, str]) -> tuple[str, str]:
    """Return (raw_path, filt_path) based on heuristics."""
    keys = list(fif_map.keys())
    raw_key  = next((k for k in keys if "awake7" in k and "clean" not in k
                                                       and "dbs" not in k), keys[0])
    filt_key = next((k for k in keys if "final clean" in k or "awake final" in k),
                    keys[-1])
    return fif_map[raw_key], fif_map[filt_key]


def main():
    args = parse_args()

    proc = pathlib.Path(args.processed_dir)

    if args.raw and args.filt:
        raw_path  = args.raw
        filt_path = args.filt
    else:
        fif_map = discover_fif(proc)
        if not fif_map:
            print(f"ERROR: No .fif files found in {proc}")
            sys.exit(1)
        print("Available .fif files:")
        for k, v in fif_map.items():
            print(f"  [{k}]  {v}")
        raw_path, filt_path = _pick_default_pair(fif_map)
        print(f"\nAuto-selected:")
        print(f"  raw  → {raw_path}")
        print(f"  filt → {filt_path}")

    print("Loading signals …")
    raw_data,  ch_names, sfreq = load_fif(raw_path)
    filt_data, ch_names2, _   = load_fif(filt_path)

    if raw_data.shape[0] != filt_data.shape[0]:
        print(f"WARNING: channel count mismatch ({raw_data.shape[0]} vs {filt_data.shape[0]}). "
              "Cropping to common channels.")
        n_common = min(raw_data.shape[0], filt_data.shape[0])
        raw_data  = raw_data[:n_common]
        filt_data = filt_data[:n_common]
        ch_names  = ch_names[:n_common]

    # Crop to shorter signal
    min_samp  = min(raw_data.shape[1], filt_data.shape[1])
    raw_data  = raw_data[:,  :min_samp]
    filt_data = filt_data[:, :min_samp]

    raw_label  = pathlib.Path(raw_path).stem.replace("_", " ")
    filt_label = pathlib.Path(filt_path).stem.replace("_", " ")

    print(f"Loaded: {raw_data.shape[1]/sfreq:.1f} s  ×  {raw_data.shape[0]} channels  "
          f"@ {sfreq:.0f} Hz")
    print("Controls: ← → scroll | +/- window | P toggle PSD | D toggle diff | S save | Q quit")

    viewer = EEGViewer(
        raw_data, filt_data, ch_names, sfreq,
        dbs_freq=args.dbs_freq,
        window_sec=args.window,
        scale_uv=args.scale,
        raw_label=raw_label,
        filt_label=filt_label,
    )
    plt.show()


if __name__ == "__main__":
    main()
