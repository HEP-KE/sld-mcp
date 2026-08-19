from __future__ import annotations
from pathlib import Path

import csv
import glob
from datetime import datetime
import os

import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jazelle
from dataclasses import replace

from mcp.server.fastmcp import FastMCP
from sld_resurrect.kinematics import build_particles
import sld_resurrect.kinematics as kin
from sld_resurrect.selector import EventSelector
from sld_resurrect import selector_presets
from sld_resurrect.event_view import EventView

mcp = FastMCP("sld-mcp-server")


def _year_mask(data, year_group: str = "all"):
    times = ak.to_numpy(data["IEVENTH"].evttime)
    years = np.array([datetime.utcfromtimestamp(int(t)).year for t in times])

    if year_group == "all":
        return np.ones(len(data), dtype=bool)
    if year_group == "1996":
        return years == 1996
    if year_group == "1997":
        return years == 1997
    if year_group == "1998":
        return years == 1998
    if year_group == "1997_1998":
        return (years == 1997) | (years == 1998)

    raise ValueError(f"Unknown year_group: {year_group}")



def _output_dir() -> Path:
    return Path(os.environ.get("MCP_OUTPUT_DIR", ".")).resolve()


def _compute_visible_mass_values(data, preset: str):
    particles = build_particles(data)
    selector = EventSelector.from_preset(preset, data, particles)
    mask = selector.mask()

    selected_particles = particles[mask]
    event_p4 = ak.sum(selected_particles, axis=1)
    masses = ak.to_numpy(event_p4.mass)

    return mask, masses


# ---------------------------------------------------------------------------
# Data cache: holds the most recent full load so consecutive tool calls
# with the same glob + max_files don't re-read parquet from disk.
# The cache stores the *full* dataset (no max_events slice); the
# max_events trim is applied after retrieval so that a 10k-event call
# doesn't evict a full-statistics cache entry.
# ---------------------------------------------------------------------------
_DATA_CACHE: dict = {}  # key: (path_glob, n_files) -> (files, files_to_load, data)


def _load_sld_data(path_glob: str, max_files: int = 1, max_events: int = -1):
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No parquet files matched: {path_glob}")

    if max_files < 0:
        files_to_load = files
    else:
        files_to_load = files[:max_files]

    cache_key = (path_glob, len(files_to_load))

    if cache_key in _DATA_CACHE:
        cached_files, cached_ftl, cached_data = _DATA_CACHE[cache_key]
        if cached_ftl == files_to_load:
            data = cached_data
            if max_events > 0:
                data = data[:max_events]
            return cached_files, cached_ftl, data

    # Cache miss -- read from disk
    arrays = []
    for path in files_to_load:
        chunk = jazelle.from_parquet(path)
        arrays.append(chunk)

    data = ak.concatenate(arrays) if len(arrays) > 1 else arrays[0]

    # Store the full dataset in cache (no max_events slice)
    _DATA_CACHE.clear()  # keep only one entry to bound memory
    _DATA_CACHE[cache_key] = (files, files_to_load, data)

    if max_events > 0:
        data = data[:max_events]

    return files, files_to_load, data



def _apply_cut_overrides(cuts, track_quality, overrides):
    """Apply user overrides to a (Selection, TrackQualityCuts) pair."""
    tq_fields = {f.name for f in track_quality.__dataclass_fields__.values()}
    tq_overrides = {}
    cut_overrides = {}

    for key, value in overrides.items():
        if key in tq_fields:
            tq_overrides[key] = value
        else:
            cut_overrides[key] = value

    if tq_overrides:
        track_quality = track_quality.with_overrides(**tq_overrides)

    if cut_overrides:
        cuts = _override_cut_tree(cuts, cut_overrides)

    return cuts, track_quality


def _override_cut_tree(cuts, overrides):
    """Recursively walk the cut tree and replace thresholds by CutSpec name."""
    new_cuts = []
    for cut in cuts:
        if hasattr(cut, "members"):
            new_members = _override_cut_tree(cut.members, overrides)
            new_cuts.append(replace(cut, members=new_members))
        else:
            if cut.name in overrides:
                new_cuts.append(replace(cut, threshold=overrides[cut.name]))
            else:
                new_cuts.append(cut)
    return new_cuts


def _describe_cut_tree(cuts, track_quality, indent=0):
    """Return a flat list of strings describing the full cut configuration."""
    lines = []
    for cut in cuts:
        prefix = "  " * indent
        if hasattr(cut, "members"):
            lines.append(f"{prefix}CutGroup: {cut.name} (combine={cut.combine})")
            lines.extend(_describe_cut_tree(cut.members, None, indent + 1))
        else:
            lines.append(f"{prefix}{cut.name}: {cut.quantity} {cut.op} {cut.threshold}")
    if track_quality is not None and indent == 0:
        lines.append("Track quality:")
        lines.append(f"  min_pt: {track_quality.min_pt}")
        lines.append(f"  max_abs_cos_theta: {track_quality.max_abs_cos_theta}")
        lines.append(f"  max_r: {track_quality.max_r}")
        lines.append(f"  max_abs_z: {track_quality.max_abs_z}")
        lines.append(f"  max_d3: {track_quality.max_d3}")
    return lines


@mcp.tool()
def inspect_sld_dataset(path_glob: str, max_files: int = 1, year_group: str = "all") -> str:
    """
    Inspect SLD parquet shards using the same loading path as sld-resurrect.

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        max_files: Maximum number of parquet shards to load for inspection.

    Returns:
        A text summary of matched files, loaded files, event count, and
        top-level SLD banks, or an error message if loading fails.
    """
    try:
        files, files_to_load, data = _load_sld_data(path_glob, max_files=max_files)
        data = data[_year_mask(data, year_group)]

        lines = []
        lines.append(f"Matched files: {len(files)}")
        lines.append(f"Loaded files: {len(files_to_load)}")
        lines.append(f"Year group: {year_group}")
        lines.append(f"Events loaded: {len(data)}")
        lines.append("Top-level fields:")
        for field in ak.fields(data):
            lines.append(f"  - {field}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to inspect SLD dataset: {e}"


@mcp.tool()
def run_sld_selection(
    path_glob: str,
    preset: str = "hadronic_default",
    max_files: int = 1,
    max_events: int = 10000,
    year_group: str = "all",
) -> str:
    """
    Load SLD parquet shards, build particles, and apply a named selection preset.

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        preset: Selection preset name, e.g. "hadronic_default".
        max_files: Maximum number of parquet shards to load.
        max_events: Maximum number of events to process. Use -1 for all loaded events.

    Returns:
        A text summary of loaded events, selected events, selection fraction,
        and preset name, or an error message if processing fails.
    """
    try:
        files, files_to_load, data = _load_sld_data(
            path_glob,
            max_files=max_files,
            max_events=max_events,
        )

        data = data[_year_mask(data, year_group)]

        particles = build_particles(data)
        selector = EventSelector.from_preset(preset, data, particles)
        mask = selector.mask()

        n_total = len(data)
        n_selected = int(mask.sum())
        frac = (100.0 * n_selected / n_total) if n_total > 0 else 0.0

        lines = []
        lines.append(f"Preset: {preset}")
        lines.append(f"Year group: {year_group}")
        lines.append(f"Matched files: {len(files)}")
        lines.append(f"Loaded files: {len(files_to_load)}")
        lines.append(f"Events processed: {n_total}")
        lines.append(f"Events selected: {n_selected}")
        lines.append(f"Selection fraction: {frac:.2f}%")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to run SLD selection: {e}"



@mcp.tool()
def list_sld_presets() -> str:
    """
    List the available named SLD event-selection presets from sld-resurrect.

    Args:
        None.

    Returns:
        A text list of available preset names, one per line, or an error message if lookup fails.
    """
    try:
        names = sorted(selector_presets.PRESETS.keys())
        lines = []
        lines.append(f"Available SLD selection presets: {len(names)}")
        for name in names:
            lines.append(f" - {name}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: Failed to list SLD presets: {e}"


@mcp.tool()
def describe_sld_preset(preset: str) -> str:
    """
    Describe a named SLD event-selection preset from sld-resurrect.

    Args:
        preset: Name of the preset to describe, for example "hadronic_default" or "leptonic_default".

    Returns:
        A text summary of whether the preset exists, its underlying function name,
        any available docstring, and simple known alias/reference notes, or an
        error message if lookup fails.
    """
    try:
        if preset not in selector_presets.PRESETS:
            names = sorted(selector_presets.PRESETS.keys())
            lines = []
            lines.append(f"Error: Unknown preset: {preset}")
            lines.append("Available presets:")
            for name in names:
                lines.append(f" - {name}")
            return "\n".join(lines)

        func = selector_presets.PRESETS[preset]
        doc = getattr(func, "__doc__", None)
        func_name = getattr(func, "__name__", "<unknown>")

        notes = {
            "hadronic_default": "Alias of alr_2000; default hadronic selection used in the SLD pipeline.",
            "alr_2000": "Published high-precision A_LR hadronic selection.",
            "alr_1994": "Published earlier A_LR hadronic selection.",
            "alphas_1995": "Published event-shape / alpha_s selection.",
            "rb_1998": "Published R_b selection.",
            "abc_2005": "Published A_b / A_c selection.",
            "leptonic_default": "Default leptonic selection family.",
            "leptonic_ee": "Leptonic e+e- channel selection.",
            "leptonic_mumu": "Leptonic mu+mu- channel selection.",
            "leptonic_tautau": "Leptonic tau+tau- channel selection.",
            "leptonic_1997_ee": "Legacy 1997 e+e- leptonic selection.",
            "leptonic_1997_mumu": "Legacy 1997 mu+mu- leptonic selection.",
            "leptonic_1997_tautau": "Legacy 1997 tau+tau- leptonic selection.",
        }

        lines = []
        lines.append(f"Preset: {preset}")
        lines.append(f"Function name: {func_name}")
        if preset in notes:
            lines.append(f"Note: {notes[preset]}")

        if doc and doc.strip():
            lines.append("Docstring:")
            lines.append(doc.strip())
        else:
            lines.append("Docstring: None available.")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: Failed to describe SLD preset {preset}: {e}"


@mcp.tool()
def run_sld_selection_with_cutflow(
    path_glob: str,
    preset: str = "hadronic_default",
    max_files: int = 1,
    max_events: int = 10000,
    year_group: str = "all",
) -> str:
    """
    Load SLD parquet shards, apply a named selection preset, and return cutflow details.

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        preset: Selection preset name, e.g. "hadronic_default".
        max_files: Maximum number of parquet shards to load.
        max_events: Maximum number of events to process. Use -1 for all loaded events.

    Returns:
        A text summary of processed events, selected events, selection fraction,
        and the cutflow reported by the EventSelector, or an error message if processing fails.
    """
    try:
        files, files_to_load, data = _load_sld_data(
            path_glob,
            max_files=max_files,
            max_events=max_events,
        )

        data = data[_year_mask(data, year_group)]

        particles = build_particles(data)
        selector = EventSelector.from_preset(preset, data, particles)
        mask = selector.mask()

        n_total = len(data)
        n_selected = int(mask.sum())
        frac = (100.0 * n_selected / n_total) if n_total > 0 else 0.0

        lines = []
        lines.append(f"Preset: {preset}")
        lines.append(f"Year group: {year_group}")
        lines.append(f"Matched files: {len(files)}")
        lines.append(f"Loaded files: {len(files_to_load)}")
        lines.append(f"Events processed: {n_total}")
        lines.append(f"Events selected: {n_selected}")
        lines.append(f"Selection fraction: {frac:.2f}%")
        lines.append("")
        lines.append("Cutflow:")

        try:
            cf = selector.cutflow()
            if isinstance(cf, str):
                lines.append(cf)
            else:
                lines.append(str(cf))
        except TypeError:
            cf = selector.cutflow
            lines.append(str(cf))
        except Exception as e:
            lines.append(f"Could not extract structured cutflow: {e}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to run SLD selection with cutflow: {e}"


@mcp.tool()
def inspect_sld_bank_schema(
    path_glob: str,
    bank_name: str,
    max_files: int = 1,
    max_list_fields: int = 100,
    year_group: str = "all",
) -> str:
    """
    Inspect the schema of a specific top-level SLD bank in the parquet dataset.

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        bank_name: Name of the top-level SLD bank to inspect, e.g. "PHBM" or "PHCHRG".
        max_files: Maximum number of parquet shards to load.
        max_list_fields: Maximum number of subfields to list.

    Returns:
        A text summary of whether the bank exists, its visible subfields, and a
        simple type/structure preview, or an error message if inspection fails.
    """
    try:
        files, files_to_load, data = _load_sld_data(path_glob, max_files=max_files)
        data = data[_year_mask(data, year_group)]

        top_fields = ak.fields(data)
        if bank_name not in top_fields:
            lines = []
            lines.append(f"Error: Bank {bank_name} not found.")
            lines.append("Available top-level banks:")
            for name in top_fields:
                lines.append(f" - {name}")
            return "\n".join(lines)

        bank = data[bank_name]

        lines = []
        lines.append(f"Bank: {bank_name}")
        lines.append(f"Matched files: {len(files)}")
        lines.append(f"Loaded files: {len(files_to_load)}")
        lines.append(f"Year group: {year_group}")
        lines.append(f"Events loaded: {len(data)}")

        try:
            lines.append(f"Awkward type: {ak.type(bank)}")
        except Exception as e:
            lines.append(f"Awkward type: <unavailable: {e}>")

        subfields = []
        try:
            subfields = list(ak.fields(bank))
        except Exception:
            subfields = []

        if subfields:
            lines.append(f"Subfields ({len(subfields)}):")
            for name in subfields[:max_list_fields]:
                lines.append(f" - {name}")
            if len(subfields) > max_list_fields:
                lines.append(f" ... ({len(subfields) - max_list_fields} more)")
        else:
            lines.append("Subfields: None visible (bank may be scalar, list-like, or opaque at this level).")

        try:
            preview = bank[:1]
            lines.append("")
            lines.append("One-event preview:")
            lines.append(str(preview))
        except Exception as e:
            lines.append("")
            lines.append(f"One-event preview unavailable: {e}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to inspect SLD bank schema for {bank_name}: {e}"


@mcp.tool()
def plot_sld_visible_mass_histogram(
    path_glob: str,
    preset: str = "hadronic_default",
    max_files: int = -1,
    max_events: int = -1,
    mass_min: float = 0.0,
    mass_max: float = 125.0,
    bin_width: float = 5.0,
    output_name: str = "sld_visible_mass_histogram.png",
    year_group: str = "all",
) -> str:
    """
    Plot a visible invariant-mass histogram for selected SLD events and save it as a PNG.

    The visible mass is computed by building per-event particles with
    sld-resurrect, summing the particle four-vectors within each selected
    event, and taking the invariant mass of the summed four-vector.

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        preset: Selection preset name, e.g. "hadronic_default".
        max_files: Maximum number of parquet shards to load. Use -1 for all matched files.
        max_events: Maximum number of events to process. Use -1 for all loaded events.
        mass_min: Lower edge of the visible-mass histogram in GeV.
        mass_max: Upper edge of the visible-mass histogram in GeV.
        bin_width: Histogram bin width in GeV.
        output_name: Output PNG filename.

    Returns:
        A text summary with selection counts, visible-mass summary statistics,
        and the saved plot path, or an error message if plotting fails.
    """
    try:
        files = sorted(glob.glob(path_glob))
        if not files:
            return f"Error: No parquet files matched: {path_glob}"

        files_to_load = files if max_files < 0 else files[:max_files]

        arrays = []
        n_loaded = 0
        for path in files_to_load:
            chunk = jazelle.from_parquet(path)
            arrays.append(chunk)
            n_loaded += len(chunk)
            if max_events > 0 and n_loaded >= max_events:
                break

        data = ak.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        if max_events > 0:
            data = data[:max_events]

        data = data[_year_mask(data, year_group)]

        particles = build_particles(data)
        selector = EventSelector.from_preset(preset, data, particles)
        mask = selector.mask()

        selected_particles = particles[mask]
        event_p4 = ak.sum(selected_particles, axis=1)
        masses = ak.to_numpy(event_p4.mass)

        bin_edges = np.arange(mass_min, mass_max + bin_width, bin_width)
        counts, edges = np.histogram(masses, bins=bin_edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        unc = np.sqrt(counts)

        output_dir = os.environ.get("MCP_OUTPUT_DIR", ".")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)

        plt.figure(figsize=(7, 5))
        plt.stairs(counts, edges, linewidth=2, color="blue", label=r"$Z \to q\bar{q}$")

        # Draw vertical Poisson statistical uncertainties only
        plt.errorbar(
            centers,
            counts,
            yerr=unc,
            fmt="none",
            ecolor="blue",
            elinewidth=1,
            capsize=0,
        )

        # Keep axis styling generic rather than forcing the paper settings
        plt.xlabel("Visible invariant mass [GeV]")
        plt.ylabel(f"Events / {bin_width:g} GeV")
        plt.title(f"SLD visible mass spectrum ({preset})")
        plt.xlim(mass_min, mass_max)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        n_total = len(data)
        n_selected = int(mask.sum())
        frac = (100.0 * n_selected / n_total) if n_total > 0 else 0.0

        lines = []
        lines.append(f"Preset: {preset}")
        lines.append(f"Year group: {year_group}")
        lines.append(f"Matched files: {len(files)}")
        lines.append(f"Loaded files: {len(files_to_load)}")
        lines.append(f"Events processed: {n_total}")
        lines.append(f"Events selected: {n_selected}")
        lines.append(f"Selection fraction: {frac:.2f}%")
        lines.append(f"Histogrammed masses: {len(masses)}")

        if len(masses) > 0:
            lines.append(f"Visible mass mean [GeV]: {float(np.mean(masses)):.6f}")
            lines.append(f"Visible mass min [GeV]: {float(np.min(masses)):.6f}")
            lines.append(f"Visible mass max [GeV]: {float(np.max(masses)):.6f}")

        lines.append(f"Events above {mass_max} GeV: {int(np.sum(masses > mass_max))}")
        lines.append(f"Saved plot: {output_path}")
        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to plot SLD visible-mass histogram: {e}"


@mcp.tool()
def plot_sld_leptonic_visible_mass_histograms(
    path_glob: str,
    max_files: int = -1,
    max_events: int = -1,
    mass_min: float = 0.0,
    mass_max: float = 125.0,
    bin_width: float = 5.0,
    output_name: str = "sld_leptonic_visible_mass_histograms.png",
    year_group: str = "all",
) -> str:
    """
    Plot overlapping visible invariant-mass histograms for the leptonic SLD selections
    Z->e+e-, Z->mu+mu-, and Z->tau+tau-, and save the plot as a PNG.

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        max_files: Maximum number of parquet shards to load. Use -1 for all matched files.
        max_events: Maximum number of events to process. Use -1 for all loaded events.
        mass_min: Lower edge of the visible-mass histogram in GeV.
        mass_max: Upper edge of the visible-mass histogram in GeV.
        bin_width: Histogram bin width in GeV.
        output_name: Output PNG filename.

    Returns:
        A text summary with event counts for the three leptonic channels and
        the saved plot path, or an error message if plotting fails.
    """
    try:
        files = sorted(glob.glob(path_glob))
        if not files:
            return f"Error: No parquet files matched: {path_glob}"

        files_to_load = files if max_files < 0 else files[:max_files]

        arrays = []
        n_loaded = 0
        for path in files_to_load:
            chunk = jazelle.from_parquet(path)
            arrays.append(chunk)
            n_loaded += len(chunk)
            if max_events > 0 and n_loaded >= max_events:
                break

        data = ak.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        if max_events > 0:
            data = data[:max_events]

        data = data[_year_mask(data, year_group)]

        particles = build_particles(data)

        channel_info = [
            ("leptonic_ee", "Z->e+e-", "tab:blue"),
            ("leptonic_mumu", "Z->mu+mu-", "tab:orange"),
            ("leptonic_tautau", "Z->tau+tau-", "tab:green"),
        ]

        bin_edges = np.arange(mass_min, mass_max + bin_width, bin_width)

        output_dir = os.environ.get("MCP_OUTPUT_DIR", ".")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)

        plt.figure(figsize=(8, 6))

        summary_lines = []
        summary_lines.append(f"Year group: {year_group}")
        summary_lines.append(f"Matched files: {len(files)}")
        summary_lines.append(f"Loaded files: {len(files_to_load)}")
        summary_lines.append(f"Events processed: {len(data)}")

        for preset, label, color in channel_info:
            selector = EventSelector.from_preset(preset, data, particles)
            mask = selector.mask()
            selected_particles = particles[mask]

            event_p4 = ak.sum(selected_particles, axis=1)
            masses = ak.to_numpy(event_p4.mass)

            counts, edges = np.histogram(masses, bins=bin_edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            unc = np.sqrt(counts)

            # Use filled histograms with transparency so overlaps blend colors
            plt.hist(
                masses,
                bins=bin_edges,
                histtype="stepfilled",
                alpha=0.35,
                color=color,
                label=label,
            )

            # Add vertical Poisson statistical uncertainties
            plt.errorbar(
                centers,
                counts,
                yerr=unc,
                fmt="none",
                ecolor=color,
                elinewidth=1,
                capsize=0,
            )

            summary_lines.append(f"{label} selected events: {len(masses)}")
            if len(masses) > 0:
                summary_lines.append(f"{label} mean visible mass [GeV]: {float(np.mean(masses)):.6f}")

        # Keep axis styling generic
        plt.xlabel("Visible invariant mass [GeV]")
        plt.ylabel(f"Events / {bin_width:g} GeV")
        plt.title("SLD leptonic visible mass spectra")
        plt.xlim(mass_min, mass_max)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        summary_lines.append(f"Saved plot: {output_path}")
        return "\n".join(summary_lines)

    except Exception as e:
        return f"Error: Failed to plot SLD leptonic visible-mass histograms: {e}"


@mcp.tool()
def compute_sld_visible_mass(
    path_glob: str,
    preset: str = "hadronic_default",
    overrides: dict | None = None,
    max_files: int = 68,
    save_csv: bool = True,
    year_group: str = "all",
) -> str:
    """Load SLD parquet shards, apply a selection preset, and compute per-event visible invariant mass.

    The visible mass is computed by summing the four-vectors of all
    particles in each selected event and taking the invariant mass of
    the sum.  Results are returned as summary statistics and optionally
    saved to a CSV artifact with columns [event_index, visible_mass].

    Accepts optional cut overrides (same format as
    run_sld_selection_with_overrides).

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        preset: Selection preset name, e.g. "hadronic_default" or "leptonic_ee".
        overrides: Optional dict of cut name to new threshold value.
        max_files: Maximum number of parquet shards to load. Use -1 for all matched files.
        save_csv: Whether to save per-event visible-mass values to a CSV artifact.
        year_group: Year group filter: "all", "1996", "1997", "1998", or "1997_1998".

    Returns:
        A text summary with event counts, visible-mass summary statistics
        (min, max, mean, median), and the saved CSV path if requested,
        or an error message if processing fails.
    """
    try:
        files, files_to_load, data = _load_sld_data(path_glob, max_files=max_files)

        data = data[_year_mask(data, year_group)]

        particles = build_particles(data)
        cuts, track_quality = selector_presets.PRESETS[preset]()
        if overrides:
            cuts, track_quality = _apply_cut_overrides(cuts, track_quality, overrides)
        selector = EventSelector(data, particles, cuts, track_quality=track_quality)
        mask = selector.mask()

        selected_particles = particles[mask]
        event_p4 = ak.sum(selected_particles, axis=1)
        masses = ak.to_numpy(event_p4.mass)

        n_total = len(data)
        n_selected = int(mask.sum())

        lines = [
            f"Preset: {preset}",
            f"Year group: {year_group}",
        ]
        if overrides:
            lines.append(f"Overrides applied: {overrides}")
        lines.extend([
            f"Matched files: {len(files)}",
            f"Loaded files: {len(files_to_load)}",
            f"Events processed: {n_total}",
            f"Events selected: {n_selected}",
            f"Visible-mass values: {len(masses)}",
        ])

        if len(masses) > 0:
            lines.extend(
                [
                    f"Visible mass min: {float(np.min(masses)):.3f} GeV",
                    f"Visible mass max: {float(np.max(masses)):.3f} GeV",
                    f"Visible mass mean: {float(np.mean(masses)):.3f} GeV",
                    f"Visible mass median: {float(np.median(masses)):.3f} GeV",
                ]
            )

        if save_csv:
            output_dir = _output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / f"visible_mass_{preset}.csv"

            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["event_index", "visible_mass"])
                for i, value in enumerate(masses):
                    writer.writerow([i, float(value)])

            lines.append(f"Saved CSV: {csv_path}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to compute SLD visible mass: {e}"



@mcp.tool()
def plot_overlay_histograms_from_csvs(
    csv_paths: list[str],
    value_column: str = "visible_mass",
    labels: list[str] | None = None,
    bins: int = 60,
    title: str = "Overlay histogram",
    xlabel: str = "Value",
    ylabel: str = "Events",
    output_name: str = "overlay_histogram.png",
    show_errors: bool = True,
    xmin: float | None = None,
    xmax: float | None = None,
) -> str:
    """Plot overlaid histograms from the same numeric column across multiple CSV files.

    Reads the specified column from each CSV, bins the values into a
    common range, and draws semi-transparent filled histograms with
    optional Poisson error bars.  Useful for comparing distributions
    across channels (e.g. leptonic visible mass) or across different
    cut variations of the same observable.

    Args:
        csv_paths: List of paths to CSV files to overlay.
        value_column: Name of the numeric column to histogram.
        labels: Optional display labels, one per CSV. Defaults to the CSV stem.
        bins: Number of histogram bins.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        output_name: Output PNG filename.
        show_errors: Whether to draw Poisson error bars on each histogram.
        xmin: Optional lower bound on the plot range.
        xmax: Optional upper bound on the plot range.

    Returns:
        A text summary with per-CSV entry counts, the plot range, and
        the saved plot path, or an error message if plotting fails.
    """
    try:
        if not csv_paths:
            return "Error: csv_paths is empty."

        if labels is not None and len(labels) != len(csv_paths):
            return "Error: labels must have the same length as csv_paths."

        if labels is None:
            labels = [Path(p).stem for p in csv_paths]

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

        all_plot_values = []
        summary_lines = []

        for i, csv_path in enumerate(csv_paths):
            df = pd.read_csv(csv_path)
            if value_column not in df.columns:
                return f"Error: Column '{value_column}' not found in {csv_path}. Available columns: {list(df.columns)}"

            values = pd.to_numeric(df[value_column], errors="coerce").dropna().to_numpy()
            if len(values) == 0:
                return f"Error: Column '{value_column}' in {csv_path} contains no valid numeric values."

            plot_values = values.copy()
            if xmin is not None:
                plot_values = plot_values[plot_values >= xmin]
            if xmax is not None:
                plot_values = plot_values[plot_values <= xmax]

            if len(plot_values) == 0:
                return f"Error: No values remain after xmin/xmax cuts for {csv_path}."

            all_plot_values.append((csv_path, labels[i], plot_values))
            summary_lines.append(
                f"{labels[i]}: total={len(values)}, plotted={len(plot_values)}, csv={csv_path}"
            )

        lo = xmin if xmin is not None else min(float(np.min(v)) for _, _, v in all_plot_values)
        hi = xmax if xmax is not None else max(float(np.max(v)) for _, _, v in all_plot_values)
        hist_range = (lo, hi)

        plt.figure(figsize=(8, 5))

        for i, (_, label, plot_values) in enumerate(all_plot_values):
            color = colors[i % len(colors)]
            counts, edges = np.histogram(plot_values, bins=bins, range=hist_range)
            centers = 0.5 * (edges[:-1] + edges[1:])
            errors = np.sqrt(counts)

            plt.hist(
                plot_values,
                bins=bins,
                range=hist_range,
                histtype="stepfilled",
                alpha=0.28,
                color=color,
                label=label,
            )

            if show_errors:
                plt.errorbar(
                    centers,
                    counts,
                    yerr=errors,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1,
                    capsize=0,
                    alpha=0.9,
                )

        plt.xlim(lo, hi)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()

        output_path = Path(output_name)
        if not output_path.is_absolute():
            output_path = _output_dir() / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path, dpi=150)
        plt.close()

        lines = [
            f"Value column: {value_column}",
            f"Bins: {bins}",
            f"Plot range: [{lo}, {hi}]",
            *summary_lines,
            f"Saved plot: {output_path}",
        ]
        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to plot overlay histograms from CSVs: {e}"


@mcp.tool()
def compute_sld_event_shapes(
    path_glob: str,
    preset: str = "hadronic_default",
    overrides: dict | None = None,
    max_files: int = 68,
    save_csv: bool = True,
    year_group: str = "all",
) -> str:
    """Load SLD parquet shards, apply a selection preset, compute per-event event-shape
    observables, and optionally save them to a CSV artifact.

    Accepts optional cut overrides (same format as
    run_sld_selection_with_overrides).

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        preset: Selection preset name, e.g. "hadronic_default".
        overrides: Optional dict of cut name to new threshold value.
        max_files: Maximum number of parquet shards to load. Use -1 for all matched files.
        save_csv: Whether to save per-event event-shape values to a CSV artifact.
        year_group: Year group filter: "all", "1996", "1997", "1998", or "1997_1998".

    Returns:
        A text summary with event counts, mean thrust/tau, and the saved
        CSV path if requested, or an error message if processing fails.
    """
    try:
        files, files_to_load, data = _load_sld_data(path_glob, max_files=max_files)

        data = data[_year_mask(data, year_group)]

        particles = build_particles(data)
        cuts, track_quality = selector_presets.PRESETS[preset]()
        if overrides:
            cuts, track_quality = _apply_cut_overrides(cuts, track_quality, overrides)
        selector = EventSelector(data, particles, cuts, track_quality=track_quality)
        mask = selector.mask()
        selected_particles = particles[mask]

        thrust_vals, thrust_vec, cos_theta_t = kin.thrust(selected_particles)
        tau = 1.0 - thrust_vals

        df = pd.DataFrame({
            "event_index": np.arange(len(thrust_vals), dtype=int),
            "thrust": thrust_vals,
            "tau": tau,
            "cos_theta_t": cos_theta_t,
        })

        # Optional extras
        try:
            df["oblateness"] = kin.oblateness(selected_particles, thrust_vec)
        except Exception:
            pass

        try:
            df["sphericity"] = kin.sphericity(selected_particles)
        except Exception:
            pass

        try:
            df["aplanarity"] = kin.aplanarity(selected_particles)
        except Exception:
            pass

        try:
            df["c_parameter"] = kin.c_parameter(selected_particles)
        except Exception:
            pass

        try:
            heavy_jet_mass, _ = kin.heavy_jet_mass(selected_particles, thrust_vec)
            df["heavy_jet_mass"] = heavy_jet_mass
        except Exception:
            pass

        lines = [
            f"Preset: {preset}",
            f"Year group: {year_group}",
        ]
        if overrides:
            lines.append(f"Overrides applied: {overrides}")
        lines.extend([
            f"Matched files: {len(files)}",
            f"Loaded files: {len(files_to_load)}",
            f"Events processed: {len(data)}",
            f"Events selected: {int(mask.sum())}",
            f"Rows written: {len(df)}",
        ])

        if len(df) > 0:
            lines.append(f"Mean thrust: {float(df['thrust'].mean()):.6f}")
            lines.append(f"Mean tau: {float(df['tau'].mean()):.6f}")

        if save_csv:
            output_dir = _output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / f"event_shapes_{preset}.csv"
            df.to_csv(csv_path, index=False)
            lines.append(f"Saved CSV: {csv_path}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to compute SLD event shapes: {e}"


@mcp.tool()
def plot_histogram_from_csv(
    csv_path: str,
    value_column: str = "visible_mass",
    bins: int = 60,
    title: str = "Histogram",
    xlabel: str = "Value",
    ylabel: str = "Events",
    output_name: str = "histogram.png",
    show_errors: bool = True,
    xmin: float | None = None,
    xmax: float | None = None,
    density: bool = False,
    logy: bool = False,
) -> str:
    """Plot a histogram from a numeric column in a CSV file and save it as a PNG.

    Reads a single numeric column from a CSV artifact (e.g. one produced
    by compute_sld_observable or compute_sld_visible_mass), histograms
    the values, and saves the result as a PNG.  Supports optional
    density normalization, log-y scale, Poisson error bars, and
    explicit axis range limits.

    Args:
        csv_path: Path to the input CSV file.
        value_column: Name of the numeric column to histogram.
        bins: Number of histogram bins.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        output_name: Output PNG filename.
        show_errors: Whether to draw Poisson error bars.
        xmin: Optional lower bound on the plot range.
        xmax: Optional upper bound on the plot range.
        density: Whether to normalize the histogram to unit area.
        logy: Whether to use a logarithmic y-axis.

    Returns:
        A text summary with entry counts, bin count, density/log-y flags,
        and the saved plot path, or an error message if plotting fails.
    """
    try:
        df = pd.read_csv(csv_path)
        if value_column not in df.columns:
            return f"Error: Column '{value_column}' not found in CSV. Available columns: {list(df.columns)}"

        values = pd.to_numeric(df[value_column], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            return f"Error: Column '{value_column}' contains no valid numeric values."

        plot_values = values.copy()
        if xmin is not None:
            plot_values = plot_values[plot_values >= xmin]
        if xmax is not None:
            plot_values = plot_values[plot_values <= xmax]

        if len(plot_values) == 0:
            return "Error: No values remain after applying xmin/xmax cuts."

        hist_range = None
        lo = None
        hi = None
        if xmin is not None or xmax is not None:
            lo = xmin if xmin is not None else float(np.min(plot_values))
            hi = xmax if xmax is not None else float(np.max(plot_values))
            hist_range = (lo, hi)

        counts, edges = np.histogram(
            plot_values,
            bins=bins,
            range=hist_range,
            density=density,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_widths = edges[1:] - edges[:-1]

        raw_counts, _ = np.histogram(
            plot_values,
            bins=bins,
            range=hist_range,
            density=False,
        )

        if density:
            errors = np.sqrt(raw_counts) / (len(plot_values) * bin_widths)
        else:
            errors = np.sqrt(raw_counts)

        plt.figure(figsize=(8, 5))
        plt.hist(
            plot_values,
            bins=bins,
            range=hist_range,
            density=density,
            histtype="stepfilled",
            alpha=0.6,
            color="royalblue",
            label=value_column,
        )

        if show_errors:
            plt.errorbar(
                centers,
                counts,
                yerr=errors,
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=0,
            )

        if lo is not None and hi is not None:
            plt.xlim(lo, hi)

        if logy:
            plt.yscale("log")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()

        output_path = Path(output_name)
        if not output_path.is_absolute():
            output_path = _output_dir() / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path, dpi=150)
        plt.close()

        lines = [
            f"CSV path: {csv_path}",
            f"Column: {value_column}",
            f"Entries total: {len(values)}",
            f"Entries plotted: {len(plot_values)}",
            f"Bins: {bins}",
            f"Density: {density}",
            f"Log-y: {logy}",
        ]

        if lo is not None and hi is not None:
            lines.append(f"Plot range: [{lo}, {hi}]")

        lines.append(f"Saved plot: {output_path}")
        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to plot histogram from CSV: {e}"


@mcp.tool()
def compute_sld_leptonic_angles(
    path_glob: str,
    preset: str = "leptonic_mumu",
    overrides: dict | None = None,
    max_files: int = 68,
    save_csv: bool = True,
    year_group: str = "all",
) -> str:
    """Load SLD parquet shards, apply a leptonic selection preset, compute cos(theta_T),
    extract beam polarization from PHBM, and optionally save the result to a CSV artifact.

    Accepts optional cut overrides (same format as
    run_sld_selection_with_overrides).

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        preset: Leptonic selection preset name, e.g. "leptonic_mumu", "leptonic_ee", "leptonic_tautau".
        overrides: Optional dict of cut name to new threshold value.
        max_files: Maximum number of parquet shards to load. Use -1 for all matched files.
        save_csv: Whether to save per-event cos(theta_T) and beam polarization to a CSV artifact.
        year_group: Year group filter: "all", "1996", "1997", "1998", or "1997_1998".

    Returns:
        A text summary with event counts, L/R helicity split, mean
        cos(theta_T), mean beam polarization, and the saved CSV path if
        requested, or an error message if processing fails.
    """
    try:
        files, files_to_load, data = _load_sld_data(path_glob, max_files=max_files)

        data = data[_year_mask(data, year_group)]

        particles = build_particles(data)
        cuts, track_quality = selector_presets.PRESETS[preset]()
        if overrides:
            cuts, track_quality = _apply_cut_overrides(cuts, track_quality, overrides)
        selector = EventSelector(data, particles, cuts, track_quality=track_quality)
        mask = selector.mask()

        selected_data = data[mask]
        selected_particles = particles[mask]

        view = EventView.from_preset(preset, selected_data, selected_particles)
        cos_theta_t = np.asarray(view.get('thrust_vec_charged')[:, 2])
        thrust_vals = np.asarray(view.get('thrust_value'))

        phbm = selected_data["PHBM"]

        # PHBM is a per-event bank stored as a one-element list in this dataset.
        # Flatten the single-entry bank and extract the polarization field.
        beam_pol = ak.to_numpy(ak.firsts(phbm.pol))
        beam_helicity_sign = np.sign(beam_pol).astype(int)

        df = pd.DataFrame({
            "event_index": np.arange(len(cos_theta_t), dtype=int),
            "cos_theta_t": cos_theta_t,
            "beam_polarization": beam_pol,
            "beam_helicity_sign": beam_helicity_sign,
            "thrust": thrust_vals,
        })

        lines = [
            f"Preset: {preset}",
            f"Year group: {year_group}",
        ]
        if overrides:
            lines.append(f"Overrides applied: {overrides}")
        lines.extend([
            f"Matched files: {len(files)}",
            f"Loaded files: {len(files_to_load)}",
            f"Events processed: {len(data)}",
            f"Events selected: {len(selected_data)}",
            f"Rows written: {len(df)}",
        ])

        if len(df) > 0:
            n_left = int(np.sum(df["beam_helicity_sign"] < 0))
            n_right = int(np.sum(df["beam_helicity_sign"] > 0))
            lines.append(f"Left-helicity events: {n_left}")
            lines.append(f"Right-helicity events: {n_right}")
            lines.append(f"Mean cos(theta_T): {float(df['cos_theta_t'].mean()):.6f}")
            lines.append(f"Mean beam polarization: {float(df['beam_polarization'].mean()):.6f}")

        if save_csv:
            output_dir = _output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / f"leptonic_angles_{preset}.csv"
            df.to_csv(csv_path, index=False)
            lines.append(f"Saved CSV: {csv_path}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to compute SLD leptonic angles: {e}"










@mcp.tool()
def plot_sld_leptonic_cos_theta_from_csv(
    csv_path: str,
    channel_label: str = "mu+mu-",
    bins: int = 18,
    output_name: str = "sld_leptonic_cos_theta.png",
    year_group: str = "all",
) -> str:
    """
    Plot the cos(theta_T) distribution for one leptonic channel from a CSV
    produced by compute_sld_leptonic_angles, split by beam helicity sign,
    using point-style markers with Poisson error bars.
    """
    try:
        df = pd.read_csv(csv_path)

        required = {"cos_theta_t", "beam_helicity_sign"}
        missing = required - set(df.columns)
        if missing:
            return f"Error: CSV is missing required columns: {sorted(missing)}"

        values_left = pd.to_numeric(
            df.loc[df["beam_helicity_sign"] < 0, "cos_theta_t"],
            errors="coerce",
        ).dropna().to_numpy()

        values_right = pd.to_numeric(
            df.loc[df["beam_helicity_sign"] > 0, "cos_theta_t"],
            errors="coerce",
        ).dropna().to_numpy()

        edges = np.linspace(-0.9, 0.9, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        counts_left, _ = np.histogram(values_left, bins=edges)
        counts_right, _ = np.histogram(values_right, bins=edges)

        err_left = np.sqrt(counts_left)
        err_right = np.sqrt(counts_right)

        output_path = Path(output_name)
        if not output_path.is_absolute():
            output_path = _output_dir() / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 5))

        plt.errorbar(
            centers,
            counts_left,
            yerr=err_left,
            fmt="o",
            color="tab:blue",
            label="Left-polarized beam",
            markersize=4,
            linewidth=1,
            capsize=0,
        )

        plt.errorbar(
            centers,
            counts_right,
            yerr=err_right,
            fmt="s",
            color="tab:orange",
            label="Right-polarized beam",
            markersize=4,
            linewidth=1,
            capsize=0,
        )

        plt.xlabel("cos(theta_T)")
        plt.ylabel("Events / bin")
        plt.title(f"SLD {channel_label} angular distribution ({year_group})")
        plt.xlim(-0.9, 0.9)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        lines = []
        lines.append(f"Channel: {channel_label}")
        lines.append(f"Year group: {year_group}")
        lines.append(f"CSV path: {csv_path}")
        lines.append(f"Left-helicity events: {len(values_left)}")
        lines.append(f"Right-helicity events: {len(values_right)}")
        lines.append(f"Saved plot: {output_path}")
        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to plot leptonic cos(theta) from CSV: {e}"



@mcp.tool()
def list_sld_tools() -> str:
    """
    List the available SLD MCP tools grouped by purpose.

    Returns:
        A curated text summary of available tools organized by category.
    """
    lines = []

    lines.append("Data Inspection (7)")
    lines.append("  - inspect_sld_dataset — inspect the parquet dataset")
    lines.append("  - inspect_sld_bank_schema — inspect a specific bank's schema")
    lines.append("  - list_sld_presets — list available selection presets")
    lines.append("  - describe_sld_preset — describe a specific preset")
    lines.append("  - describe_sld_preset_cuts — show overridable cut names and thresholds for a preset")
    lines.append("  - list_sld_observables — list all available observable names")
    lines.append("  - list_sld_tools — this tool; list all tools by category")
    lines.append("")

    lines.append("Selection & Cutflow (5)")
    lines.append("  - run_sld_selection — apply a selection preset")
    lines.append("  - run_sld_selection_with_cutflow — apply a selection with cutflow reporting")
    lines.append("  - run_sld_selection_with_overrides — apply a selection with optional cut overrides")
    lines.append("  - scan_cut_threshold — sweep a cut across a range of values, report counts or A_LR")
    lines.append("  - compare_cuts — side-by-side baseline vs. modified selection with deltas")
    lines.append("")

    lines.append("Physics Computation (8)")
    lines.append("  - compute_sld_alr — hadronic A_LR and sin2(theta_W)")
    lines.append("  - compute_sld_leptonic_asymmetry — leptonic asymmetry for one channel and year group")
    lines.append("  - compute_sld_leptonic_summary — combine into A_e, A_mu, A_tau, and universal A_l")
    lines.append("  - compute_sld_measurement_table — combined summary table of selection counts, hadronic A_LR, and leptonic asymmetries across year groups")
    lines.append("  - compute_sld_observable — any of 45+ per-event observables, with CSV output")
    lines.append("  - compute_sld_visible_mass — visible invariant mass distributions")
    lines.append("  - compute_sld_event_shapes — thrust, sphericity, oblateness, C-parameter, etc.")
    lines.append("  - compute_sld_leptonic_angles — cos(theta_T) split by beam helicity")
    lines.append("")

    lines.append("Plotting (7)")
    lines.append("  - plot_sld_visible_mass_histogram — hadronic visible-mass histogram")
    lines.append("  - plot_sld_leptonic_visible_mass_histograms — leptonic visible-mass overlay")
    lines.append("  - plot_sld_leptonic_cos_theta_from_csv — cos(theta_T) by beam helicity with error bars")
    lines.append("  - plot_histogram_from_csv — generic histogram from a CSV column")
    lines.append("  - plot_overlay_histograms_from_csvs — overlay histograms from multiple CSVs")
    lines.append("  - plot_scatter_from_csv — scatter plot of two CSV columns, optional color-by-third")
    lines.append("  - plot_sld_weak_mixing_summary — sin2(theta_W) error-bar summary across channels")

    return "\n".join(lines)



@mcp.tool()
def compute_sld_alr(
    path_glob: str,
    preset: str = "hadronic_default",
    overrides: dict | None = None,
    max_files: int = -1,
    max_events: int = -1,
    year_group: str = "all",
) -> str:
    """
    Compute the counting-based raw hadronic left-right asymmetry A_LR.

    This uses the selected hadronic event counts split by the sign of the
    beam polarization and divides the raw left-right count asymmetry by the
    mean absolute beam polarization. This is the measured raw asymmetry, not
    the electroweak-corrected pole asymmetry.

    Accepts optional cut overrides (same format as
    run_sld_selection_with_overrides). Example:
        overrides={"e_lac_ge_22": 30.0}  -- raise LAC energy cut to 30 GeV

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        preset: Selection preset name.
        overrides: Optional dict of cut name to new threshold value.
        max_files: Maximum number of parquet shards to load. Use -1 for all.
        max_events: Maximum number of events to process. Use -1 for all.
        year_group: Year group filter.
    """
    try:
        files, files_to_load, data = _load_sld_data(
            path_glob,
            max_files=max_files,
            max_events=max_events,
        )

        data = data[_year_mask(data, year_group)]

        particles = build_particles(data)

        cuts, track_quality = selector_presets.PRESETS[preset]()
        if overrides:
            cuts, track_quality = _apply_cut_overrides(cuts, track_quality, overrides)
        selector = EventSelector(data, particles, cuts, track_quality=track_quality)
        mask = selector.mask()

        selected_data = data[mask]
        phbm = selected_data["PHBM"]

        beam_pol = ak.to_numpy(ak.firsts(phbm.pol))
        beam_pol = beam_pol[np.isfinite(beam_pol)]

        beam_dpol = ak.to_numpy(ak.firsts(phbm.dpol))
        beam_dpol = beam_dpol[np.isfinite(beam_dpol)]

        n_total = len(data)
        n_selected = len(selected_data)

        n_left = int(np.sum(beam_pol < 0))
        n_right = int(np.sum(beam_pol > 0))
        n_lr = n_left + n_right

        mean_abs_pol = float(np.mean(np.abs(beam_pol))) if len(beam_pol) > 0 else 0.0
        raw_count_asym = ((n_left - n_right) / n_lr) if n_lr > 0 else 0.0
        alr = (raw_count_asym / mean_abs_pol) if mean_abs_pol > 0 else 0.0

        dp_e = float(np.mean(np.abs(beam_dpol))) if len(beam_dpol) > 0 else 0.0
        d_raw_count_asym = np.sqrt(max(0.0, 1.0 - raw_count_asym**2) / n_lr) if n_lr > 0 else 0.0
        d_alr = (
            (1.0 / abs(mean_abs_pol)) * np.sqrt(d_raw_count_asym**2 + (raw_count_asym * dp_e / mean_abs_pol) ** 2)
            if mean_abs_pol > 0
            else 0.0
        )

        lines = []
        lines.append(f"Preset: {preset}")
        lines.append(f"Year group: {year_group}")
        if overrides:
            lines.append(f"Overrides applied: {overrides}")
        lines.append(f"Matched files: {len(files)}")
        lines.append(f"Loaded files: {len(files_to_load)}")
        lines.append(f"Events processed: {n_total}")
        lines.append(f"Events selected: {n_selected}")
        lines.append(f"N_L: {n_left}")
        lines.append(f"N_R: {n_right}")
        lines.append(f"Mean |P_e|: {mean_abs_pol:.6f}")
        lines.append(f"Raw count asymmetry: {raw_count_asym:.6f}")
        lines.append(f"Raw A_LR: {alr:.6f}")
        lines.append(f"Mean |dP_e|: {dp_e:.6f}")
        lines.append(f"dRaw count asymmetry: {d_raw_count_asym:.6f}")
        lines.append(f"dRaw A_LR: {d_alr:.6f}")

        # Derive sin2(theta_W) from A_LR = A_e = 2*ve/(1+ve^2)
        if abs(alr) < 1.0 and alr != 0.0:
            ve = (1.0 - np.sqrt(1.0 - alr**2)) / alr
            sin2tw = (1.0 - ve) / 4.0
            # Propagate uncertainty
            dve_dalr = 1.0 / np.sqrt(1.0 - alr**2)
            dsin2tw = dve_dalr * d_alr / 4.0
            lines.append(f"sin2(theta_W): {sin2tw:.5f}")
            lines.append(f"dsin2(theta_W): {dsin2tw:.5f}")
        else:
            lines.append("sin2(theta_W): nan")
            lines.append("dsin2(theta_W): nan")

        lines.append("Note: This is the counting-based raw measured A_LR, not the ZFITTER-corrected pole asymmetry.")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to compute SLD A_LR: {e}"

@mcp.tool()
def compute_sld_leptonic_asymmetry(
    path_glob: str,
    preset: str = "leptonic_mumu",
    overrides: dict | None = None,
    max_files: int = -1,
    max_events: int = -1,
    year_group: str = "all",
) -> str:
    """
    Compute a counting-based leptonic asymmetry baseline for one leptonic channel.

    This follows the resurrection-paper baseline philosophy: direct event
    counting with a fiducial |cos(theta_T)| cut and simple beam-helicity
    splitting, rather than the original unbinned likelihood method.
    """
    try:
        year_params = {
            "1996": {"cos_theta_max": 0.8, "tau_bias": -0.0182},
            "1997_1998": {"cos_theta_max": 0.9, "tau_bias": -0.0183},
        }

        if year_group not in year_params:
            return f"Error: Unsupported year_group for leptonic asymmetry: {year_group}"

        cos_theta_max = year_params[year_group]["cos_theta_max"]
        tau_bias = year_params[year_group]["tau_bias"]

        files, files_to_load, data = _load_sld_data(
            path_glob,
            max_files=max_files,
            max_events=max_events,
        )

        data = data[_year_mask(data, year_group)]

        particles = build_particles(data)

        cuts, track_quality = selector_presets.PRESETS[preset]()
        if overrides:
            cuts, track_quality = _apply_cut_overrides(cuts, track_quality, overrides)
        selector = EventSelector(data, particles, cuts, track_quality=track_quality)
        mask = selector.mask()

        selected_data = data[mask]
        selected_particles = particles[mask]

        view = EventView.from_preset(preset, selected_data, selected_particles)
        cos_theta_t = np.asarray(view.get("thrust_vec_charged")[:, 2])

        phbm = selected_data["PHBM"]
        beam_pol = ak.to_numpy(ak.firsts(phbm.pol))
        beam_dpol = ak.to_numpy(ak.firsts(phbm.dpol))

        valid = np.isfinite(cos_theta_t)
        valid &= np.isfinite(beam_pol) & np.isfinite(beam_dpol)
        valid &= np.abs(beam_pol) > 0.1
        valid &= np.abs(cos_theta_t) <= cos_theta_max

        cos_theta_t = cos_theta_t[valid]
        beam_pol = beam_pol[valid]
        beam_dpol = beam_dpol[valid]

        left = beam_pol < 0
        right = beam_pol > 0
        forward = cos_theta_t >= 0
        backward = cos_theta_t < 0

        n_l = int(np.sum(left))
        n_r = int(np.sum(right))

        n_fl = int(np.sum(forward & left))
        n_bl = int(np.sum(backward & left))
        n_fr = int(np.sum(forward & right))
        n_br = int(np.sum(backward & right))

        p_e = float(np.mean(np.abs(beam_pol))) if len(beam_pol) > 0 else 0.0
        dp_e = float(np.mean(np.abs(beam_dpol))) if len(beam_dpol) > 0 else 0.0
        f_geom = (3.0 + cos_theta_max**2) / (3.0 * cos_theta_max) if cos_theta_max > 0 else 0.0

        n_lr = n_l + n_r
        a_lr = ((n_l - n_r) / n_lr) if n_lr > 0 else 0.0
        d_a_lr = np.sqrt(max(0.0, 1.0 - a_lr**2) / n_lr) if n_lr > 0 else 0.0

        a_e_channel = (a_lr / p_e) if p_e != 0 else 0.0
        d_a_e_channel = (
            (1.0 / abs(p_e)) * np.sqrt(d_a_lr**2 + (a_lr * dp_e / p_e) ** 2)
            if p_e != 0
            else 0.0
        )

        a_lrfb = ((n_fl - n_bl - n_fr + n_br) / n_lr) if n_lr > 0 else 0.0
        d_a_lrfb = np.sqrt(max(0.0, 1.0 - a_lrfb**2) / n_lr) if n_lr > 0 else 0.0

        a_l = (f_geom * a_lrfb / p_e) if p_e != 0 else 0.0
        if preset == "leptonic_tautau":
            a_l = a_l + tau_bias

        d_a_l = (
            (f_geom / abs(p_e)) * np.sqrt(d_a_lrfb**2 + (a_lrfb * dp_e / p_e) ** 2)
            if p_e != 0
            else 0.0
        )

        lines = []
        lines.append(f"Preset: {preset}")
        lines.append(f"Year group: {year_group}")
        if overrides:
            lines.append(f"Overrides applied: {overrides}")
        lines.append(f"Matched files: {len(files)}")
        lines.append(f"Loaded files: {len(files_to_load)}")
        lines.append(f"Events processed: {len(data)}")
        lines.append(f"Events selected: {len(selected_data)}")
        lines.append(f"Fiducial |cos(theta_T)| max: {cos_theta_max:.3f}")
        lines.append(f"N_L: {n_l}")
        lines.append(f"N_R: {n_r}")
        lines.append(f"N_FL: {n_fl}")
        lines.append(f"N_BL: {n_bl}")
        lines.append(f"N_FR: {n_fr}")
        lines.append(f"N_BR: {n_br}")
        lines.append(f"Mean |P_e|: {p_e:.6f}")
        lines.append(f"Mean |dP_e|: {dp_e:.6f}")
        lines.append(f"Geometric factor f_geom: {f_geom:.6f}")
        lines.append(f"A_LR: {a_lr:.6f}")
        lines.append(f"dA_LR: {d_a_lr:.6f}")
        lines.append(f"A_e_channel: {a_e_channel:.6f}")
        lines.append(f"dA_e_channel: {d_a_e_channel:.6f}")
        lines.append(f"A_LRFB: {a_lrfb:.6f}")
        lines.append(f"dA_LRFB: {d_a_lrfb:.6f}")
        lines.append(f"A_l: {a_l:.6f}")
        lines.append(f"dA_l: {d_a_l:.6f}")
        if preset == "leptonic_tautau":
            lines.append(f"Tau bias applied: {tau_bias:.6f}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to compute SLD leptonic asymmetry: {e}"


@mcp.tool()
def compute_sld_leptonic_summary(path_glob: str, overrides: dict | None = None, max_files: int = -1, max_events: int = -1) -> str:
    """
    Combine SLD leptonic asymmetry results across channels and year groups into
    inverse-variance-weighted A_e, A_mu, A_tau, and universal A_l estimates.
    """
    try:
        def ivw(values, errors):
            values = np.asarray(values, dtype=float)
            errors = np.asarray(errors, dtype=float)
            mask = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
            values = values[mask]
            errors = errors[mask]
            if len(values) == 0:
                return float("nan"), float("nan")
            weights = 1.0 / errors**2
            combined = float(np.sum(weights * values) / np.sum(weights))
            combined_unc = float(np.sqrt(1.0 / np.sum(weights)))
            return combined, combined_unc

        def parse_field(text, label):
            prefix = label + ":"
            for line in text.splitlines():
                if line.startswith(prefix):
                    return float(line[len(prefix):].strip())
            raise ValueError(f"Could not parse {label} from result:\n{text}")

        year_groups = ["1996", "1997_1998"]

        mumu_results = {
            yg: compute_sld_leptonic_asymmetry(
                path_glob,
                preset="leptonic_mumu",
                overrides=overrides,
                max_files=max_files,
                max_events=max_events,
                year_group=yg,
            )
            for yg in year_groups
        }

        tautau_results = {
            yg: compute_sld_leptonic_asymmetry(
                path_glob,
                preset="leptonic_tautau",
                overrides=overrides,
                max_files=max_files,
                max_events=max_events,
                year_group=yg,
            )
            for yg in year_groups
        }

        a_e_values = []
        a_e_errors = []
        a_mu_values = []
        a_mu_errors = []
        for yg in year_groups:
            text = mumu_results[yg]
            a_e_values.append(parse_field(text, "A_e_channel"))
            a_e_errors.append(parse_field(text, "dA_e_channel"))
            a_mu_values.append(parse_field(text, "A_l"))
            a_mu_errors.append(parse_field(text, "dA_l"))

        a_tau_values = []
        a_tau_errors = []
        for yg in year_groups:
            text = tautau_results[yg]
            a_tau_values.append(parse_field(text, "A_l"))
            a_tau_errors.append(parse_field(text, "dA_l"))

        a_e, d_a_e = ivw(a_e_values, a_e_errors)
        a_mu, d_a_mu = ivw(a_mu_values, a_mu_errors)
        a_tau, d_a_tau = ivw(a_tau_values, a_tau_errors)
        a_l_universal, d_a_l_universal = ivw(
            [a_e, a_mu, a_tau],
            [d_a_e, d_a_mu, d_a_tau],
        )

        lines = []
        lines.append(f"Path glob: {path_glob}")
        lines.append(f"Max files: {max_files}")
        lines.append(f"Max events: {max_events}")
        lines.append("")
        lines.append("Muon channel (leptonic_mumu) A_e_channel by year group:")
        for yg in year_groups:
            lines.append(f"  {yg}: A_e_channel={a_e_values[year_groups.index(yg)]:.6f} dA_e_channel={a_e_errors[year_groups.index(yg)]:.6f}")
        lines.append("Muon channel (leptonic_mumu) A_l by year group:")
        for yg in year_groups:
            lines.append(f"  {yg}: A_l={a_mu_values[year_groups.index(yg)]:.6f} dA_l={a_mu_errors[year_groups.index(yg)]:.6f}")
        lines.append("Tau channel (leptonic_tautau) A_l by year group:")
        for yg in year_groups:
            lines.append(f"  {yg}: A_l={a_tau_values[year_groups.index(yg)]:.6f} dA_l={a_tau_errors[year_groups.index(yg)]:.6f}")
        lines.append("")
        lines.append(f"A_e (inverse-variance weighted): {a_e:.6f} +/- {d_a_e:.6f}")
        lines.append(f"A_mu (inverse-variance weighted): {a_mu:.6f} +/- {d_a_mu:.6f}")
        lines.append(f"A_tau (inverse-variance weighted): {a_tau:.6f} +/- {d_a_tau:.6f}")
        lines.append(f"A_l universal (inverse-variance weighted from A_e, A_mu, A_tau): {a_l_universal:.6f} +/- {d_a_l_universal:.6f}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to compute SLD leptonic summary: {e}"


@mcp.tool()
def compute_sld_measurement_table(
    path_glob: str,
    overrides: dict | None = None,
    max_files: int = -1,
    max_events: int = -1,
) -> str:
    """
    Build a markdown summary table of the main SLD measurement results by
    running the underlying selection and asymmetry tools and collecting
    their outputs into one table, broken out by year group (1996, 1997_1998,
    all):

    - Runs run_sld_selection with preset "hadronic_default" to get
      reconstructed Z->qqbar event counts.
    - Runs compute_sld_alr on the same preset to get N_L, N_R, raw A_LR,
      and the derived sin2(theta_W)_eff.
    - Runs run_sld_selection for the "leptonic_ee", "leptonic_mumu", and
      "leptonic_tautau" presets to get per-channel event counts.
    - Runs compute_sld_leptonic_asymmetry per lepton channel and year group
      (1996, 1997_1998 only) to get A_e, A_mu, A_tau.
    - Runs compute_sld_leptonic_summary to get the inverse-variance-weighted
      combined A_e, A_mu, A_tau and the universal A_l, then derives
      sin2(theta_W) from A_l.

    All physics calls accept the same `overrides` dict for cut thresholds,
    and `max_files`/`max_events` are forwarded to each underlying call for
    subsampling.
    """
    try:
        def parse_field(text, label):
            prefix = label + ":"
            for line in text.splitlines():
                if line.startswith(prefix):
                    return line[len(prefix):].strip()
            raise ValueError(f"Could not parse '{label}' from result:\n{text}")

        def parse_int(text, label):
            return int(parse_field(text, label))

        def parse_float(text, label):
            return float(parse_field(text, label))

        def parse_ivw(text, label):
            prefix = label + ":"
            for line in text.splitlines():
                if line.startswith(prefix):
                    rest = line[len(prefix):].strip()
                    val_str, unc_str = rest.split("+/-")
                    return float(val_str.strip()), float(unc_str.strip())
            raise ValueError(f"Could not parse '{label}' from result:\n{text}")

        def fmt_pm(value, unc):
            return f"{value:.6f} ± {unc:.6f}"

        files, files_to_load, data = _load_sld_data(
            path_glob,
            max_files=max_files,
            max_events=max_events,
        )

        n_all_1996 = int(np.sum(_year_mask(data, "1996")))
        n_all_1997_1998 = int(np.sum(_year_mask(data, "1997_1998")))
        n_all_combined = int(np.sum(_year_mask(data, "all")))

        n_l = {}
        n_r = {}
        raw_alr = {}
        d_raw_alr = {}
        for yg in ["1996", "1997_1998"]:
            text = compute_sld_alr(
                path_glob,
                preset="hadronic_default",
                overrides=overrides,
                max_files=max_files,
                max_events=max_events,
                year_group=yg,
            )
            n_l[yg] = parse_int(text, "N_L")
            n_r[yg] = parse_int(text, "N_R")
            raw_alr[yg] = parse_float(text, "Raw A_LR")
            d_raw_alr[yg] = parse_float(text, "dRaw A_LR")

        w = 1.0 / np.array([d_raw_alr["1996"], d_raw_alr["1997_1998"]]) ** 2
        values = np.array([raw_alr["1996"], raw_alr["1997_1998"]])
        raw_alr["all"] = float(np.sum(w * values) / np.sum(w))
        d_raw_alr["all"] = float(np.sqrt(1.0 / np.sum(w)))
        n_l["all"] = n_l["1996"] + n_l["1997_1998"]
        n_r["all"] = n_r["1996"] + n_r["1997_1998"]

        n_had = {}
        for yg in ["1996", "1997_1998", "all"]:
            n_had[yg] = n_l[yg] + n_r[yg]

        sin2_eff = {}
        d_sin2_eff = {}
        for yg in ["1996", "1997_1998", "all"]:
            sin2_eff[yg], d_sin2_eff[yg] = _sin2_theta_from_asymmetry(raw_alr[yg], d_raw_alr[yg])

        n_ee = {}
        n_mumu = {}
        n_tautau = {}
        for yg in ["1996", "1997_1998", "all"]:
            n_ee[yg] = parse_int(
                run_sld_selection(
                    path_glob, preset="leptonic_ee", max_files=max_files, max_events=max_events, year_group=yg
                ),
                "Events selected",
            )
            n_mumu[yg] = parse_int(
                run_sld_selection(
                    path_glob, preset="leptonic_mumu", max_files=max_files, max_events=max_events, year_group=yg
                ),
                "Events selected",
            )
            n_tautau[yg] = parse_int(
                run_sld_selection(
                    path_glob, preset="leptonic_tautau", max_files=max_files, max_events=max_events, year_group=yg
                ),
                "Events selected",
            )

        lepton_year_groups = ["1996", "1997_1998"]

        a_e_by_year = {}
        d_a_e_by_year = {}
        a_mu_by_year = {}
        d_a_mu_by_year = {}
        a_tau_by_year = {}
        d_a_tau_by_year = {}
        for yg in lepton_year_groups:
            mumu_text = compute_sld_leptonic_asymmetry(
                path_glob,
                preset="leptonic_mumu",
                overrides=overrides,
                max_files=max_files,
                max_events=max_events,
                year_group=yg,
            )
            a_e_by_year[yg] = parse_float(mumu_text, "A_e_channel")
            d_a_e_by_year[yg] = parse_float(mumu_text, "dA_e_channel")
            a_mu_by_year[yg] = parse_float(mumu_text, "A_l")
            d_a_mu_by_year[yg] = parse_float(mumu_text, "dA_l")

            tautau_text = compute_sld_leptonic_asymmetry(
                path_glob,
                preset="leptonic_tautau",
                overrides=overrides,
                max_files=max_files,
                max_events=max_events,
                year_group=yg,
            )
            a_tau_by_year[yg] = parse_float(tautau_text, "A_l")
            d_a_tau_by_year[yg] = parse_float(tautau_text, "dA_l")

        summary_text = compute_sld_leptonic_summary(
            path_glob,
            overrides=overrides,
            max_files=max_files,
            max_events=max_events,
        )
        a_e_combined, d_a_e_combined = parse_ivw(summary_text, "A_e (inverse-variance weighted)")
        a_mu_combined, d_a_mu_combined = parse_ivw(summary_text, "A_mu (inverse-variance weighted)")
        a_tau_combined, d_a_tau_combined = parse_ivw(summary_text, "A_tau (inverse-variance weighted)")
        a_l_univ, d_a_l_univ = parse_ivw(
            summary_text,
            "A_l universal (inverse-variance weighted from A_e, A_mu, A_tau)",
        )

        sin2_lep, d_sin2_lep = _sin2_theta_from_asymmetry(a_l_univ, d_a_l_univ)

        rows = [
            ("All reconstructed", n_all_1996, n_all_1997_1998, n_all_combined),
            ("Z→q qbar", n_had["1996"], n_had["1997_1998"], n_had["all"]),
            ("N_L", n_l["1996"], n_l["1997_1998"], n_l["all"]),
            ("N_R", n_r["1996"], n_r["1997_1998"], n_r["all"]),
            ("Z→e+e−", n_ee["1996"], n_ee["1997_1998"], n_ee["all"]),
            ("Z→μ+μ−", n_mumu["1996"], n_mumu["1997_1998"], n_mumu["all"]),
            ("Z→τ+τ−", n_tautau["1996"], n_tautau["1997_1998"], n_tautau["all"]),
            (
                "A_LR",
                fmt_pm(raw_alr["1996"], d_raw_alr["1996"]),
                fmt_pm(raw_alr["1997_1998"], d_raw_alr["1997_1998"]),
                fmt_pm(raw_alr["all"], d_raw_alr["all"]),
            ),
            (
                "sin^2θ_W^eff",
                fmt_pm(sin2_eff["1996"], d_sin2_eff["1996"]),
                fmt_pm(sin2_eff["1997_1998"], d_sin2_eff["1997_1998"]),
                fmt_pm(sin2_eff["all"], d_sin2_eff["all"]),
            ),
            (
                "A_e",
                fmt_pm(a_e_by_year["1996"], d_a_e_by_year["1996"]),
                fmt_pm(a_e_by_year["1997_1998"], d_a_e_by_year["1997_1998"]),
                fmt_pm(a_e_combined, d_a_e_combined),
            ),
            (
                "A_μ",
                fmt_pm(a_mu_by_year["1996"], d_a_mu_by_year["1996"]),
                fmt_pm(a_mu_by_year["1997_1998"], d_a_mu_by_year["1997_1998"]),
                fmt_pm(a_mu_combined, d_a_mu_combined),
            ),
            (
                "A_τ",
                fmt_pm(a_tau_by_year["1996"], d_a_tau_by_year["1996"]),
                fmt_pm(a_tau_by_year["1997_1998"], d_a_tau_by_year["1997_1998"]),
                fmt_pm(a_tau_combined, d_a_tau_combined),
            ),
            ("A_ℓ (univ.)", "—", "—", fmt_pm(a_l_univ, d_a_l_univ)),
            ("sin^2θ_W^lep", "—", "—", fmt_pm(sin2_lep, d_sin2_lep)),
        ]

        df = pd.DataFrame(rows, columns=["Quantity", "1996", "1997–1998", "Combined"])

        output_path = _output_dir() / "sld_measurement_table.csv"
        df.to_csv(output_path, index=False)

        return (
            f"Saved CSV: {output_path}"
            + "\n\n"
            + df.to_string(index=False)
        )

    except Exception as e:
        return f"Error: Failed to compute SLD measurement table: {e}"


def _sin2_theta_from_asymmetry(a: float, da: float) -> tuple[float, float]:
    if not np.isfinite(a) or a <= 0 or a >= 1:
        return (float("nan"), float("nan"))

    v = (1.0 - np.sqrt(1.0 - a**2)) / a
    sin2 = (1.0 - v) / 4.0

    eps = 1e-6
    a_plus = min(a + eps, 1.0 - 1e-9)
    a_minus = max(a - eps, 1e-9)

    v_plus = (1.0 - np.sqrt(1.0 - a_plus**2)) / a_plus
    sin2_plus = (1.0 - v_plus) / 4.0

    v_minus = (1.0 - np.sqrt(1.0 - a_minus**2)) / a_minus
    sin2_minus = (1.0 - v_minus) / 4.0

    dsin2_da = (sin2_plus - sin2_minus) / (a_plus - a_minus)
    d_sin2 = abs(dsin2_da) * da

    return float(sin2), float(d_sin2)


@mcp.tool()
def plot_sld_weak_mixing_summary(
    path_glob: str,
    overrides: dict | None = None,
    max_files: int = -1,
    max_events: int = -1,
    output_name: str = "sld_weak_mixing_summary.png",
) -> str:
    """
    Compute sin^2(theta_W^eff) from the combined hadronic A_LR and the
    combined leptonic asymmetries, and plot a horizontal error-bar summary
    comparing SLD A_LR, A_e, A_mu, A_tau, and the universal A_l.
    """
    try:
        def parse_field(text, label):
            prefix = label + ":"
            for line in text.splitlines():
                if line.startswith(prefix):
                    return float(line[len(prefix):].strip())
            raise ValueError(f"Could not parse '{label}' from result:\n{text}")

        def parse_ivw(text, label):
            prefix = label + ":"
            for line in text.splitlines():
                if line.startswith(prefix):
                    rest = line[len(prefix):].strip()
                    val_str, unc_str = rest.split("+/-")
                    return float(val_str.strip()), float(unc_str.strip())
            raise ValueError(f"Could not parse '{label}' from result:\n{text}")

        alr_text = compute_sld_alr(
            path_glob,
            preset="hadronic_default",
            max_files=max_files,
            max_events=max_events,
            year_group="all",
        )
        raw_alr = parse_field(alr_text, "Raw A_LR")
        d_raw_alr = parse_field(alr_text, "dRaw A_LR")

        summary_text = compute_sld_leptonic_summary(
            path_glob,
            overrides=overrides,
            max_files=max_files,
            max_events=max_events,
        )
        a_e, d_a_e = parse_ivw(summary_text, "A_e (inverse-variance weighted)")
        a_mu, d_a_mu = parse_ivw(summary_text, "A_mu (inverse-variance weighted)")
        a_tau, d_a_tau = parse_ivw(summary_text, "A_tau (inverse-variance weighted)")
        a_l_universal, d_a_l_universal = parse_ivw(
            summary_text,
            "A_l universal (inverse-variance weighted from A_e, A_mu, A_tau)",
        )

        sin2_alr, d_sin2_alr = _sin2_theta_from_asymmetry(raw_alr, d_raw_alr)
        sin2_e, d_sin2_e = _sin2_theta_from_asymmetry(a_e, d_a_e)
        sin2_mu, d_sin2_mu = _sin2_theta_from_asymmetry(a_mu, d_a_mu)
        sin2_tau, d_sin2_tau = _sin2_theta_from_asymmetry(a_tau, d_a_tau)
        sin2_lep, d_sin2_lep = _sin2_theta_from_asymmetry(a_l_universal, d_a_l_universal)

        rows = [
            ("SLD A_LR", sin2_alr, d_sin2_alr),
            ("A_e", sin2_e, d_sin2_e),
            ("A_mu", sin2_mu, d_sin2_mu),
            ("A_tau", sin2_tau, d_sin2_tau),
            ("A_l universal", sin2_lep, d_sin2_lep),
        ]

        labels = [r[0] for r in rows]
        values = [r[1] for r in rows]
        errors = [r[2] for r in rows]
        y_positions = np.arange(len(rows))[::-1]

        output_path = Path(output_name)
        if not output_path.is_absolute():
            output_path = _output_dir() / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 5))
        plt.errorbar(
            values,
            y_positions,
            xerr=errors,
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            markersize=6,
            linewidth=1,
            capsize=3,
        )
        plt.yticks(y_positions, labels)
        plt.xlabel("sin^2(theta_W^eff)")
        plt.title("SLD weak mixing angle summary")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        lines = []
        lines.append(f"Saved plot: {output_path}")
        for label, value, unc in rows:
            lines.append(f"{label}: {value:.6f} +/- {unc:.6f}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to plot SLD weak mixing summary: {e}"



@mcp.tool()
def describe_sld_preset_cuts(preset: str = "hadronic_default") -> str:
    """
    Show the full cut tree and track-quality parameters for a preset,
    with the names and current thresholds that can be passed as overrides
    to run_sld_selection_with_overrides.

    Args:
        preset: Selection preset name.

    Returns:
        Text listing every CutSpec name, its quantity, operator, and
        default threshold, plus the TrackQualityCuts fields.
    """
    try:
        if preset not in selector_presets.PRESETS:
            names = sorted(selector_presets.PRESETS.keys())
            return f"Unknown preset: {preset}\nAvailable: {', '.join(names)}"

        cuts, track_quality = selector_presets.PRESETS[preset]()

        lines = [f"Preset: {preset}", ""]
        lines.append("Event-level cuts (override by CutSpec name):")
        lines.extend(_describe_cut_tree(cuts, None, indent=1))
        lines.append("")
        lines.append("Track-quality parameters (override by field name):")
        lines.append(f"  min_pt: {track_quality.min_pt}")
        lines.append(f"  max_abs_cos_theta: {track_quality.max_abs_cos_theta}")
        lines.append(f"  max_r: {track_quality.max_r}")
        lines.append(f"  max_abs_z: {track_quality.max_abs_z}")
        lines.append(f"  max_d3: {track_quality.max_d3}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to describe preset cuts: {e}"


@mcp.tool()
def run_sld_selection_with_overrides(
    path_glob: str,
    preset: str = "hadronic_default",
    overrides: dict | None = None,
    max_files: int = -1,
    max_events: int = -1,
    year_group: str = "all",
    show_cutflow: bool = True,
) -> str:
    """
    Apply a selection preset with optional cut overrides and return the cutflow.

    Overrides is a flat dictionary. Keys that match TrackQualityCuts fields
    (min_pt, max_abs_cos_theta, max_r, max_abs_z, max_d3) modify track
    quality. Keys that match a CutSpec name in the cut tree replace that
    cut's threshold.

    Example overrides:
        {"e_lac_ge_22": 25.0}           -- raise LAC energy cut from 22 to 25 GeV
        {"eimb_lt_06": 0.4}             -- tighten energy imbalance from 0.6 to 0.4
        {"min_pt": 0.15}                -- raise track pT from 100 to 150 MeV
        {"e_lac_ge_22": 25.0, "min_pt": 0.2} -- both at once

    To see available cut names and their current thresholds, use
    describe_sld_preset_cuts.

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        preset: Selection preset name.
        overrides: Optional dict of cut name to new threshold value.
        max_files: Maximum number of parquet shards to load. Use -1 for all.
        max_events: Maximum number of events to process. Use -1 for all.
        year_group: Year group filter: "all", "1996", "1997", "1998", "1997_1998".
        show_cutflow: Whether to include the cutflow in the output.

    Returns:
        Text summary with event counts, selection fraction, applied overrides,
        and optionally the cutflow.
    """
    try:
        files, files_to_load, data = _load_sld_data(
            path_glob, max_files=max_files, max_events=max_events,
        )
        data = data[_year_mask(data, year_group)]
        particles = build_particles(data)

        cuts, track_quality = selector_presets.PRESETS[preset]()

        if overrides:
            cuts, track_quality = _apply_cut_overrides(cuts, track_quality, overrides)

        selector = EventSelector(data, particles, cuts, track_quality=track_quality)
        mask = selector.mask()

        n_total = len(data)
        n_selected = int(mask.sum())
        frac = (100.0 * n_selected / n_total) if n_total > 0 else 0.0

        lines = []
        lines.append(f"Preset: {preset}")
        lines.append(f"Year group: {year_group}")
        if overrides:
            lines.append(f"Overrides applied: {overrides}")
        lines.append(f"Matched files: {len(files)}")
        lines.append(f"Loaded files: {len(files_to_load)}")
        lines.append(f"Events processed: {n_total}")
        lines.append(f"Events selected: {n_selected}")
        lines.append(f"Selection fraction: {frac:.2f}%")

        if show_cutflow:
            lines.append("")
            lines.append("Cutflow:")
            try:
                cf = selector.cutflow()
                lines.append(str(cf) if not isinstance(cf, str) else cf)
            except TypeError:
                lines.append(str(selector.cutflow))
            except Exception as e:
                lines.append(f"Could not extract cutflow: {e}")

        lines.append("")
        lines.append("Active cut configuration:")
        lines.extend(_describe_cut_tree(cuts, track_quality))

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to run SLD selection with overrides: {e}"



@mcp.tool()
def compute_sld_observable(
    path_glob: str,
    observable: str,
    preset: str = "hadronic_default",
    overrides: dict | None = None,
    max_files: int = -1,
    max_events: int = -1,
    year_group: str = "all",
    save_csv: bool = True,
) -> str:
    """
    Compute any per-event observable on selected SLD events.

    The observable can be any EventView built-in or kinematics shorthand.
    Use list_sld_observables to see all available names.

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        observable: Name of the observable to compute.
        preset: Selection preset name.
        overrides: Optional dict of cut name to new threshold value.
        max_files: Maximum number of parquet shards to load. Use -1 for all.
        max_events: Maximum number of events to process. Use -1 for all.
        year_group: Year group filter.
        save_csv: Whether to save per-event values to a CSV artifact.

    Returns:
        Summary statistics and optionally the path to a saved CSV.
    """
    try:
        files, files_to_load, data = _load_sld_data(
            path_glob, max_files=max_files, max_events=max_events,
        )
        data = data[_year_mask(data, year_group)]
        particles = build_particles(data)

        cuts, track_quality = selector_presets.PRESETS[preset]()
        if overrides:
            cuts, track_quality = _apply_cut_overrides(cuts, track_quality, overrides)
        selector = EventSelector(data, particles, cuts, track_quality=track_quality)
        mask = selector.mask()

        selected_data = data[mask]
        selected_particles = particles[mask]

        view = EventView(selected_data, selected_particles, track_quality=track_quality)

        event_view_keys = {
            "n_charged", "e_vis_charged", "max_charged_p", "charged_mass",
            "thrust_value", "thrust_vec", "abs_cos_theta_thrust",
            "thrust_vec_charged", "abs_cos_theta_thrust_charged",
            "n_charged_beam_fwd", "n_charged_beam_bwd",
            "hem_net_charge_fwd", "hem_net_charge_bwd",
            "hem_charges_opposite_unit", "hem_opening_angle",
            "hem_invariant_mass_max", "hem_invariant_mass_min",
            "hem_top_track_lac_max", "hem_top_track_lac_min",
            "hem_top_track_lac_sum",
            "e_vis_total", "energy_imbalance", "event_year",
            "n_lac_clusters", "lac_total_energy", "lac_em_energy",
            "lac_em_fraction", "n_wic_matches", "wic_total_hits",
            "wic_min_nlayexp", "wic_max_matchChi2",
        }

        kin_particles_only = {
            "visible_mass", "charged_multiplicity", "charged_invariant_mass",
            "max_charged_momentum", "visible_energy",
            "sphericity", "aplanarity", "c_parameter",
            "normalised_energy_imbalance",
        }

        kin_needs_thrust = {
            "oblateness", "heavy_jet_mass", "thrust_major", "thrust_minor",
        }

        values = None

        if observable in event_view_keys:
            result = view.get(observable)
            if result.ndim == 2:
                values = result
            else:
                values = np.asarray(result, dtype=float)

        elif observable == "visible_mass":
            event_p4 = ak.sum(selected_particles, axis=1)
            values = ak.to_numpy(event_p4.mass)

        elif observable == "thrust" or observable == "tau":
            thrust_vals, _, _ = kin.thrust(selected_particles)
            values = (1.0 - thrust_vals) if observable == "tau" else thrust_vals

        elif observable == "charged_multiplicity":
            values = kin.charged_multiplicity(selected_particles)

        elif observable == "charged_invariant_mass":
            values = kin.charged_invariant_mass(selected_particles)

        elif observable == "max_charged_momentum":
            values = kin.max_charged_momentum(selected_particles)

        elif observable == "visible_energy":
            values = kin.visible_energy(selected_particles)

        elif observable == "sphericity":
            values = kin.sphericity(selected_particles)

        elif observable == "aplanarity":
            values = kin.aplanarity(selected_particles)

        elif observable == "c_parameter":
            values = kin.c_parameter(selected_particles)

        elif observable == "normalised_energy_imbalance":
            values = kin.normalised_energy_imbalance(selected_particles)

        elif observable == "oblateness":
            _, thrust_vec, _ = kin.thrust(selected_particles)
            values = kin.oblateness(selected_particles, thrust_vec)

        elif observable == "heavy_jet_mass":
            _, thrust_vec, _ = kin.thrust(selected_particles)
            values, _ = kin.heavy_jet_mass(selected_particles, thrust_vec)

        elif observable == "thrust_major":
            _, thrust_vec, _ = kin.thrust(selected_particles)
            major, _ = kin.thrust_major_minor(selected_particles, thrust_vec)
            values = major

        elif observable == "thrust_minor":
            _, thrust_vec, _ = kin.thrust(selected_particles)
            _, minor = kin.thrust_major_minor(selected_particles, thrust_vec)
            values = minor

        elif observable == "beam_polarization":
            phbm = selected_data["PHBM"]
            values = ak.to_numpy(ak.firsts(phbm.pol))

        elif observable == "beam_energy":
            phbm = selected_data["PHBM"]
            values = ak.to_numpy(ak.firsts(phbm.ecm))

        else:
            try:
                result = view.get(observable)
                values = np.asarray(result, dtype=float) if result.ndim == 1 else result
            except KeyError:
                available = sorted(event_view_keys | kin_particles_only | kin_needs_thrust | {"beam_polarization", "beam_energy"})
                return f"Error: Unknown observable '{observable}'.\nAvailable: {', '.join(available)}"

        values = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]

        lines = [
            f"Observable: {observable}",
            f"Preset: {preset}",
            f"Year group: {year_group}",
        ]
        if overrides:
            lines.append(f"Overrides: {overrides}")
        lines.extend([
            f"Matched files: {len(files)}",
            f"Loaded files: {len(files_to_load)}",
            f"Events processed: {len(data)}",
            f"Events selected: {int(mask.sum())}",
            f"Values computed: {len(values)}",
            f"Finite values: {len(finite_values)}",
        ])

        if len(finite_values) > 0:
            lines.extend([
                f"Mean: {float(np.mean(finite_values)):.6f}",
                f"Std: {float(np.std(finite_values)):.6f}",
                f"Min: {float(np.min(finite_values)):.6f}",
                f"Max: {float(np.max(finite_values)):.6f}",
                f"Median: {float(np.median(finite_values)):.6f}",
                f"25th percentile: {float(np.percentile(finite_values, 25)):.6f}",
                f"75th percentile: {float(np.percentile(finite_values, 75)):.6f}",
            ])

        if save_csv:
            output_dir = _output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / f"observable_{observable}_{preset}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["event_index", observable])
                for i, v in enumerate(values):
                    writer.writerow([i, float(v)])
            lines.append(f"Saved CSV: {csv_path}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to compute observable '{observable}': {e}"


@mcp.tool()
def list_sld_observables() -> str:
    """
    List all available observable names that can be passed to compute_sld_observable.

    Returns:
        Text listing of all available observables grouped by category.
    """
    lines = []
    lines.append("EventView quantities:")
    for name in sorted([
        "n_charged", "e_vis_charged", "max_charged_p", "charged_mass",
        "thrust_value", "abs_cos_theta_thrust",
        "abs_cos_theta_thrust_charged",
        "n_charged_beam_fwd", "n_charged_beam_bwd",
        "hem_net_charge_fwd", "hem_net_charge_bwd",
        "hem_charges_opposite_unit", "hem_opening_angle",
        "hem_invariant_mass_max", "hem_invariant_mass_min",
        "hem_top_track_lac_max", "hem_top_track_lac_min",
        "hem_top_track_lac_sum",
        "e_vis_total", "energy_imbalance", "event_year",
        "n_lac_clusters", "lac_total_energy", "lac_em_energy",
        "lac_em_fraction", "n_wic_matches", "wic_total_hits",
        "wic_min_nlayexp", "wic_max_matchChi2",
    ]):
        lines.append(f"  {name}")
    lines.append("")
    lines.append("Kinematics shorthands:")
    for name in sorted([
        "visible_mass", "thrust", "tau", "charged_multiplicity",
        "charged_invariant_mass", "max_charged_momentum", "visible_energy",
        "sphericity", "aplanarity", "c_parameter",
        "normalised_energy_imbalance", "oblateness", "heavy_jet_mass",
        "thrust_major", "thrust_minor",
    ]):
        lines.append(f"  {name}")
    lines.append("")
    lines.append("Beam quantities:")
    lines.append("  beam_polarization")
    lines.append("  beam_energy")
    return "\n".join(lines)


def main() -> None:
    mcp.run()


@mcp.tool()
def plot_scatter_from_csv(
    csv_path: str,
    x_column: str,
    y_column: str,
    title: str = "Scatter plot",
    xlabel: str | None = None,
    ylabel: str | None = None,
    output_name: str = "scatter.png",
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    color_column: str | None = None,
    alpha: float = 0.3,
    marker_size: float = 2.0,
    max_points: int = 50000,
    colorbar_label: str | None = None,
) -> str:
    """Plot a scatter plot of two numeric columns from a CSV file and save as PNG.

    Reads two columns from a CSV artifact (e.g. one produced by
    compute_sld_observable, compute_sld_event_shapes, or
    compute_sld_leptonic_angles) and draws a scatter plot.  Optionally
    colors points by a third column.  If the CSV has more rows than
    max_points, a random subsample is drawn to keep the plot readable.

    Args:
        csv_path: Path to the input CSV file.
        x_column: Name of the column for the x-axis.
        y_column: Name of the column for the y-axis.
        title: Plot title.
        xlabel: X-axis label. Defaults to x_column if not set.
        ylabel: Y-axis label. Defaults to y_column if not set.
        output_name: Output PNG filename.
        xmin: Optional lower bound on the x-axis.
        xmax: Optional upper bound on the x-axis.
        ymin: Optional lower bound on the y-axis.
        ymax: Optional upper bound on the y-axis.
        color_column: Optional column name to color points by (continuous colormap).
        alpha: Marker transparency (0–1).
        marker_size: Marker size in points.
        max_points: Maximum number of points to plot. If the CSV has more,
            a random subsample is drawn.
        colorbar_label: Label for the colorbar when color_column is used.

    Returns:
        A text summary with column names, point counts, axis ranges, and
        the saved plot path, or an error message if plotting fails.
    """
    try:
        df = pd.read_csv(csv_path)

        for col in [x_column, y_column]:
            if col not in df.columns:
                return f"Error: Column \'{col}\' not found in CSV. Available columns: {list(df.columns)}"

        if color_column is not None and color_column not in df.columns:
            return f"Error: Color column \'{color_column}\' not found in CSV. Available columns: {list(df.columns)}"

        # Extract and clean
        keep_cols = [x_column, y_column]
        if color_column is not None:
            keep_cols.append(color_column)

        plot_df = df[keep_cols].copy()
        for col in keep_cols:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna()

        if len(plot_df) == 0:
            return "Error: No valid numeric rows after cleaning."

        # Apply axis limits before subsampling
        if xmin is not None:
            plot_df = plot_df[plot_df[x_column] >= xmin]
        if xmax is not None:
            plot_df = plot_df[plot_df[x_column] <= xmax]
        if ymin is not None:
            plot_df = plot_df[plot_df[y_column] >= ymin]
        if ymax is not None:
            plot_df = plot_df[plot_df[y_column] <= ymax]

        if len(plot_df) == 0:
            return "Error: No points remain after applying axis limits."

        n_total = len(plot_df)

        # Subsample if needed
        if max_points > 0 and n_total > max_points:
            plot_df = plot_df.sample(n=max_points, random_state=42)

        n_plotted = len(plot_df)

        # Build the plot
        fig, ax = plt.subplots(figsize=(8, 6))

        scatter_kwargs = dict(
            s=marker_size,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
        )

        if color_column is not None:
            sc = ax.scatter(
                plot_df[x_column],
                plot_df[y_column],
                c=plot_df[color_column],
                cmap="viridis",
                **scatter_kwargs,
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(colorbar_label if colorbar_label else color_column)
        else:
            ax.scatter(
                plot_df[x_column],
                plot_df[y_column],
                color="royalblue",
                **scatter_kwargs,
            )

        ax.set_xlabel(xlabel if xlabel else x_column)
        ax.set_ylabel(ylabel if ylabel else y_column)
        ax.set_title(title)

        if xmin is not None or xmax is not None:
            ax.set_xlim(
                left=xmin if xmin is not None else None,
                right=xmax if xmax is not None else None,
            )
        if ymin is not None or ymax is not None:
            ax.set_ylim(
                bottom=ymin if ymin is not None else None,
                top=ymax if ymax is not None else None,
            )

        fig.tight_layout()

        output_path = Path(output_name)
        if not output_path.is_absolute():
            output_path = _output_dir() / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        lines = [
            f"CSV path: {csv_path}",
            f"X column: {x_column}",
            f"Y column: {y_column}",
        ]
        if color_column:
            lines.append(f"Color column: {color_column}")
        lines.extend([
            f"Total valid points: {n_total}",
            f"Points plotted: {n_plotted}",
        ])
        if n_plotted < n_total:
            lines.append(f"(subsampled from {n_total} to {max_points})")
        lines.append(f"Saved plot: {output_path}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to plot scatter from CSV: {e}"


@mcp.tool()
def scan_cut_threshold(
    path_glob: str,
    cut_name: str,
    values: list[float],
    preset: str = "hadronic_default",
    metric: str = "count",
    max_files: int = 68,
    max_events: int = -1,
    year_group: str = "all",
    output_dir: str | None = None,
) -> str:
    """Sweep a single cut threshold across a list of values and report how the
    selection count or a physics observable changes at each step.

    This is the primary tool for studying the sensitivity of a measurement
    to a particular cut.  The cut_name must be a valid CutSpec name or
    TrackQualityCuts field (use describe_sld_preset_cuts to list them).

    Available metrics:
        "count"  — report N_selected and selection fraction at each value
        "alr"    — report N_L, N_R, A_LR, and sin2(theta_W) at each value
                   (only meaningful for hadronic presets)

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        cut_name: Name of the cut to sweep (e.g. "e_lac_ge_22", "min_pt").
        values: List of threshold values to try.
        preset: Selection preset name.
        metric: What to report at each step: "count" or "alr".
        max_files: Maximum number of parquet shards to load. Use -1 for all.
        max_events: Maximum number of events to process. Use -1 for all.
        year_group: Year group filter: "all", "1996", "1997", "1998", "1997_1998".
        output_dir: If provided, save the scan results as a CSV file to
            {output_dir}/scan_{cut_name}.csv (the directory is created if it
            doesn't already exist).

    Returns:
        A markdown table with one row per threshold value showing the
        selected counts and (if metric="alr") the asymmetry and
        sin2(theta_W), or an error message if processing fails. If
        output_dir was provided, the path to the saved CSV is appended.
    """
    try:
        files, files_to_load, data = _load_sld_data(
            path_glob, max_files=max_files, max_events=max_events,
        )
        data = data[_year_mask(data, year_group)]
        particles = build_particles(data)

        n_total = len(data)

        if metric == "alr":
            phbm = data["PHBM"]
            beam_pol = ak.to_numpy(ak.firsts(phbm.pol))

        rows = []
        for val in values:
            overrides = {cut_name: val}
            cuts, track_quality = selector_presets.PRESETS[preset]()
            cuts, track_quality = _apply_cut_overrides(cuts, track_quality, overrides)
            selector = EventSelector(data, particles, cuts, track_quality=track_quality)
            mask = selector.mask()
            n_sel = int(mask.sum())
            frac = 100.0 * n_sel / n_total if n_total > 0 else 0.0

            row = {"value": val, "n_selected": n_sel, "fraction_pct": round(frac, 2)}

            if metric == "alr":
                sel_pol = beam_pol[ak.to_numpy(mask)]
                n_left = int(np.sum(sel_pol < 0))
                n_right = int(np.sum(sel_pol > 0))
                mean_pol = float(np.mean(np.abs(sel_pol))) if len(sel_pol) > 0 else 0.0

                if n_left + n_right > 0 and mean_pol > 0:
                    a_raw = (n_left - n_right) / (n_left + n_right)
                    a_lr = a_raw / mean_pol
                    ve = 1.0 - 4.0 * 0.23150  # initial guess
                    # Solve A_e = 2*ve/(1+ve^2) = a_lr for sin2tw
                    # sin2tw = (1 - ve) / 4 where ve = (1 - sqrt(1 - a_lr^2)) / a_lr
                    if abs(a_lr) < 1.0:
                        ve_solved = (1.0 - np.sqrt(1.0 - a_lr**2)) / a_lr
                        sin2tw = (1.0 - ve_solved) / 4.0
                    else:
                        sin2tw = float("nan")
                else:
                    n_left = n_right = 0
                    a_lr = float("nan")
                    sin2tw = float("nan")

                row.update({
                    "n_left": n_left,
                    "n_right": n_right,
                    "a_lr": round(a_lr, 6) if not np.isnan(a_lr) else "nan",
                    "sin2tw": round(sin2tw, 5) if not np.isnan(sin2tw) else "nan",
                })

            rows.append(row)

        # Build markdown table
        if metric == "alr":
            header = f"| {cut_name} | N_selected | frac(%) | N_L | N_R | A_LR | sin²θ_W |"
            sep =    "|---|---|---|---|---|---|---|"
            lines = [
                f"Scan: {cut_name} over {values}",
                f"Preset: {preset}  |  Year group: {year_group}  |  Events: {n_total}",
                "",
                header,
                sep,
            ]
            for r in rows:
                lines.append(
                    f"| {r['value']} | {r['n_selected']} | {r['fraction_pct']} "
                    f"| {r['n_left']} | {r['n_right']} | {r['a_lr']} | {r['sin2tw']} |"
                )
        else:
            header = f"| {cut_name} | N_selected | frac(%) |"
            sep =    "|---|---|---|"
            lines = [
                f"Scan: {cut_name} over {values}",
                f"Preset: {preset}  |  Year group: {year_group}  |  Events: {n_total}",
                "",
                header,
                sep,
            ]
            for r in rows:
                lines.append(f"| {r['value']} | {r['n_selected']} | {r['fraction_pct']} |")

        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / f"scan_{cut_name}.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            lines.append("")
            lines.append(f"Saved CSV: {csv_path}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to scan cut threshold: {e}"


@mcp.tool()
def compare_cuts(
    path_glob: str,
    preset: str = "hadronic_default",
    overrides: dict | None = None,
    max_files: int = 68,
    max_events: int = -1,
    year_group: str = "all",
) -> str:
    """Run a selection preset twice — once with default cuts and once with
    overrides — and return a side-by-side comparison showing the impact.

    Reports event counts, selection fractions, and (for hadronic presets)
    A_LR and sin2(theta_W) for both configurations, plus the absolute
    and percentage differences.

    Args:
        path_glob: Glob pattern for SLD parquet shards.
        preset: Selection preset name.
        overrides: Dict of cut name to new threshold value for the modified run.
        max_files: Maximum number of parquet shards to load. Use -1 for all.
        max_events: Maximum number of events to process. Use -1 for all.
        year_group: Year group filter: "all", "1996", "1997", "1998", "1997_1998".

    Returns:
        A markdown table comparing baseline vs. modified selection, with
        absolute and percentage deltas, or an error message if processing fails.
    """
    try:
        if not overrides:
            return "Error: overrides dict is required for compare_cuts."

        files, files_to_load, data = _load_sld_data(
            path_glob, max_files=max_files, max_events=max_events,
        )
        data = data[_year_mask(data, year_group)]
        particles = build_particles(data)
        n_total = len(data)

        phbm = data["PHBM"]
        beam_pol = ak.to_numpy(ak.firsts(phbm.pol))

        def _run_one(override_dict):
            cuts, tq = selector_presets.PRESETS[preset]()
            if override_dict:
                cuts, tq = _apply_cut_overrides(cuts, tq, override_dict)
            selector = EventSelector(data, particles, cuts, track_quality=tq)
            mask = selector.mask()
            n_sel = int(mask.sum())

            sel_pol = beam_pol[ak.to_numpy(mask)]
            n_l = int(np.sum(sel_pol < 0))
            n_r = int(np.sum(sel_pol > 0))
            mean_p = float(np.mean(np.abs(sel_pol))) if len(sel_pol) > 0 else 0.0

            if n_l + n_r > 0 and mean_p > 0:
                a_raw = (n_l - n_r) / (n_l + n_r)
                a_lr = a_raw / mean_p
                if abs(a_lr) < 1.0:
                    ve = (1.0 - np.sqrt(1.0 - a_lr**2)) / a_lr
                    s2tw = (1.0 - ve) / 4.0
                else:
                    s2tw = float("nan")
            else:
                a_lr = float("nan")
                s2tw = float("nan")

            return {
                "n_selected": n_sel,
                "frac": 100.0 * n_sel / n_total if n_total > 0 else 0.0,
                "n_l": n_l,
                "n_r": n_r,
                "a_lr": a_lr,
                "sin2tw": s2tw,
            }

        base = _run_one(None)
        mod = _run_one(overrides)

        def _delta(a, b):
            if np.isnan(a) or np.isnan(b):
                return "—", "—"
            d = b - a
            pct = 100.0 * d / abs(a) if a != 0 else float("inf")
            return f"{d:+.4f}" if abs(d) < 10 else f"{d:+.0f}", f"{pct:+.1f}%"

        rows = [
            ("N_selected", base["n_selected"], mod["n_selected"]),
            ("Fraction (%)", round(base["frac"], 2), round(mod["frac"], 2)),
            ("N_L", base["n_l"], mod["n_l"]),
            ("N_R", base["n_r"], mod["n_r"]),
            ("A_LR", round(base["a_lr"], 6) if not np.isnan(base["a_lr"]) else "nan",
                     round(mod["a_lr"], 6) if not np.isnan(mod["a_lr"]) else "nan"),
            ("sin²θ_W", round(base["sin2tw"], 5) if not np.isnan(base["sin2tw"]) else "nan",
                        round(mod["sin2tw"], 5) if not np.isnan(mod["sin2tw"]) else "nan"),
        ]

        lines = [
            f"Preset: {preset}",
            f"Year group: {year_group}",
            f"Events processed: {n_total}",
            f"Overrides: {overrides}",
            "",
            "| Quantity | Baseline | Modified | Δ | Δ(%) |",
            "|---|---|---|---|---|",
        ]

        for label, bval, mval in rows:
            try:
                d_abs, d_pct = _delta(float(bval), float(mval))
            except (ValueError, TypeError):
                d_abs, d_pct = "—", "—"
            lines.append(f"| {label} | {bval} | {mval} | {d_abs} | {d_pct} |")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: Failed to compare cuts: {e}"

if __name__ == "__main__":
    main()