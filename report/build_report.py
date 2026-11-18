"""Automated Technical Report PDF Compiler & Architecture Diagram Generator (Member 4).

Compiles all markdown sections in `report/sections/` (Sections 1 through 7) and any
referenced PNG figures in `report/figures/` into a clean, publication-ready A4 PDF
(`report/Technical_Report.pdf`) using pure Matplotlib PdfPages (Python 3.9+ compatible,
zero OS-level native dependencies).

Usage:
    python report/build_report.py
"""

from __future__ import annotations

import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPORT_DIR = Path(__file__).resolve().parent
SECTIONS_DIR = REPORT_DIR / "sections"
FIGURES_DIR = REPORT_DIR / "figures"
OUTPUT_PDF = REPORT_DIR / "Technical_Report.pdf"

# A4 dimensions in inches
A4_W, A4_H = 8.27, 11.69


def generate_architecture_diagram(output_path: Path) -> Path:
    """Render a clean publication-quality PNG diagram of the two-laptop LAN architecture."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 5.2), dpi=220)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(
        5.5,
        4.92,
        "Two-Laptop Live Evaluation Architecture (LAN Client-Server Deployment)",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#0F172A",
    )

    # Laptop 1 Container
    l1_box = mpatches.FancyBboxPatch(
        (0.3, 0.35),
        5.7,
        4.25,
        boxstyle="round,pad=0.1,rounding_size=0.18",
        linewidth=1.8,
        edgecolor="#2563EB",
        facecolor="#EFF6FF",
    )
    ax.add_patch(l1_box)
    ax.text(
        3.15,
        4.32,
        "LAPTOP 1: FastAPI Inference Server (0.0.0.0:8000)",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#1D4ED8",
    )

    stages = [
        (0.55, 3.35, 2.45, 0.68, "1. Data & Graph (P1)\n.nodes/.arcs + NetworkX DiGraph", "#DBEAFE", "#1E40AF"),
        (3.30, 3.35, 2.45, 0.68, "2. Candidate Gen (P1)\nKDTree / Spatial Top-K", "#DBEAFE", "#1E40AF"),
        (0.55, 2.30, 2.45, 0.68, "3. Feature + Model (P2)\nbest_model.pkl Emissions", "#EDE9FE", "#5B21B6"),
        (3.30, 2.30, 2.45, 0.68, "4. HMM Viterbi (P3)\nreconstruct_route()", "#DCFCE7", "#166534"),
        (0.55, 1.20, 5.20, 0.72, "5. Server-Side Evaluator & Viva Logger (P4)\nIsolated Ground-Truth .route | Top-1, Continuity, F1, Stage ms", "#FEF3C7", "#92400E"),
    ]
    for x, y, w, h, label, fc, ec in stages:
        patch = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            linewidth=1.2,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=7.6,
            color="#0F172A",
        )

    arrow_kw = dict(arrowstyle="-|>", color="#334155", lw=1.3, mutation_scale=11)
    ax.annotate("", xy=(3.28, 3.69), xytext=(3.02, 3.69), arrowprops=arrow_kw)
    ax.annotate(
        "",
        xy=(1.78, 3.00),
        xytext=(4.52, 3.33),
        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.1, ls="--", mutation_scale=10),
    )
    ax.annotate("", xy=(3.28, 2.64), xytext=(3.02, 2.64), arrowprops=arrow_kw)
    ax.annotate("", xy=(4.52, 1.94), xytext=(4.52, 2.28), arrowprops=arrow_kw)

    ax.text(
        3.15,
        0.72,
        "Endpoints: GET /health  |  GET /trips  |  POST /predict_trip  |  POST /predict_batch",
        ha="center",
        va="center",
        fontsize=7.6,
        family="monospace",
        color="#1E293B",
    )

    # Laptop 2 Container
    l2_box = mpatches.FancyBboxPatch(
        (7.15, 0.35),
        3.55,
        4.25,
        boxstyle="round,pad=0.1,rounding_size=0.18",
        linewidth=1.8,
        edgecolor="#059669",
        facecolor="#ECFDF5",
    )
    ax.add_patch(l2_box)
    ax.text(
        8.92,
        4.32,
        "LAPTOP 2: Evaluator UI",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#047857",
    )

    l2_blocks = [
        (7.40, 3.25, 3.05, 0.72, "Streamlit Dashboard (ui/app.py)\nIP Config + 10-Trip Dropdown\nKPI Cards & Batch Benchmark", "#FFFFFF", "#059669"),
        (7.40, 1.95, 3.05, 1.00, "Interactive Folium Visualizer\n(src/visualizer.py)\nL1: Raw GPS (Blue)\nL2: Road Network (Gray)\nL3: Ground Truth (Green)\nL4: Predicted HMM (Purple)", "#FFFFFF", "#7C3AED"),
        (7.40, 0.65, 3.05, 0.95, "Exports & Live Diagnostics\n- Standalone HTML Map Export\n- 10-Trip Benchmark CSV\n- LAN RTT & Stage Breakdown", "#FFFFFF", "#0F766E"),
    ]
    for x, y, w, h, label, fc, ec in l2_blocks:
        patch = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            linewidth=1.2,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7.5, color="#0F172A")

    # Bidirectional LAN Arrows
    ax.annotate(
        "",
        xy=(6.12, 3.25),
        xytext=(7.05, 3.25),
        arrowprops=dict(arrowstyle="-|>", color="#DC2626", lw=2.0, mutation_scale=13),
    )
    ax.text(
        6.58,
        3.56,
        'HTTP POST\n{"trip_id"}',
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#B91C1C",
    )

    ax.annotate(
        "",
        xy=(7.05, 2.25),
        xytext=(6.12, 2.25),
        arrowprops=dict(arrowstyle="-|>", color="#16A34A", lw=2.0, mutation_scale=13),
    )
    ax.text(
        6.58,
        1.82,
        "JSON Response\nMetrics &\nGeometries",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color="#15803D",
    )
    ax.text(6.58, 2.75, "Wi-Fi / LAN", ha="center", va="center", fontsize=7.6, style="italic", color="#475569")

    fig.tight_layout(pad=0.3)
    fig.savefig(str(output_path), dpi=220)
    plt.close(fig)
    return output_path


def _clean_md(text: str) -> str:
    """Strip inline markdown & LaTeX formatting cleanly for crisp vector PDF rendering."""
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # Clean common LaTeX tokens from teammate sections
    cleaned = cleaned.replace("$$", "")
    cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", cleaned)
    cleaned = cleaned.replace(r"\rightarrow", "->").replace(r"\approx", "~").replace(r"\times", "x")
    cleaned = cleaned.replace(r"\ge", ">=").replace(r"\le", "<=").replace(r"\in", "in")
    cleaned = cleaned.replace(r"\Delta", "d").replace(r"\phi", "phi").replace(r"\lambda", "lon")
    cleaned = cleaned.replace(r"\theta", "theta").replace(r"\sigma", "sigma").replace(r"\perp", "_perp")
    cleaned = cleaned.replace(r"\quad", "  ").replace(r"\cdot", "*").replace(r"\%", "%")
    cleaned = cleaned.replace(r"^\circ", " deg").replace("$", "")
    cleaned = re.sub(r"\\([a-zA-Z]+)", r"\1", cleaned)
    return cleaned


def _parse_markdown_blocks(md_text: str, base_dir: Path) -> list[dict[str, Any]]:
    """Parse markdown text into structured layout blocks (handles fenced code & HR too)."""
    blocks: list[dict[str, Any]] = []
    lines = md_text.splitlines()
    idx = 0

    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped or stripped in ("---", "***"):
            idx += 1
            continue

        # Fenced code block
        if stripped.startswith("```"):
            idx += 1
            code_lines: list[str] = []
            while idx < len(lines) and not lines[idx].strip().startswith("```"):
                code_lines.append(lines[idx].rstrip())
                idx += 1
            if idx < len(lines):
                idx += 1
            if code_lines:
                blocks.append({"type": "code", "text": "\n".join(code_lines)})
            continue

        img_match = re.match(r"^!\[(.*?)\]\((.*?)\)$", stripped)
        if img_match:
            caption, rel_path = img_match.group(1), img_match.group(2)
            img_path = (base_dir / rel_path).resolve()
            if img_path.exists():
                blocks.append({"type": "image", "path": img_path, "caption": _clean_md(caption)})
            idx += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            rows: list[list[str]] = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                cells = [_clean_md(c.strip()) for c in lines[idx].strip().strip("|").split("|")]
                if not all(re.match(r"^:?-{2,}:?$", c) for c in cells if c):
                    rows.append(cells)
                idx += 1
            if rows:
                blocks.append({"type": "table", "rows": rows})
            continue

        if stripped.startswith("# "):
            blocks.append({"type": "h1", "text": _clean_md(stripped[2:])})
            idx += 1
            continue
        if stripped.startswith("## "):
            blocks.append({"type": "h2", "text": _clean_md(stripped[3:])})
            idx += 1
            continue
        if stripped.startswith("### ") or stripped.startswith("#### "):
            blocks.append({"type": "h3", "text": _clean_md(stripped.lstrip("# "))})
            idx += 1
            continue

        bullet_match = re.match(r"^([-*]|\d+\.)\s+(.*)$", stripped)
        if bullet_match:
            marker, body = bullet_match.group(1), bullet_match.group(2)
            prefix = "•" if marker in ("-", "*") else marker
            blocks.append({"type": "bullet", "prefix": prefix, "text": _clean_md(body)})
            idx += 1
            continue

        para_parts = [stripped]
        idx += 1
        while idx < len(lines):
            nxt = lines[idx].strip()
            if (
                not nxt
                or nxt in ("---", "***")
                or nxt.startswith("#")
                or nxt.startswith("|")
                or nxt.startswith("![")
                or nxt.startswith("```")
                or re.match(r"^([-*]|\d+\.)\s+", nxt)
            ):
                break
            para_parts.append(nxt)
            idx += 1
        blocks.append({"type": "para", "text": _clean_md(" ".join(para_parts))})

    return blocks


class _A4PdfWriter:
    """Simple, crisp A4 multi-page vector PDF paginator built on Matplotlib PdfPages."""

    def __init__(self, pdf: PdfPages) -> None:
        self.pdf = pdf
        self.page_num = 0
        self.fig = None
        self.ax = None
        self.cursor_y = 0.0
        self.left_x = 0.09
        self.right_x = 0.91
        self.top_y = 0.93
        self.bottom_y = 0.075

    def _finish_page(self) -> None:
        if self.fig is None:
            return
        self.ax.plot(
            [self.left_x, self.right_x],
            [0.055, 0.055],
            color="#CBD5E1",
            lw=0.6,
            transform=self.ax.transAxes,
        )
        self.ax.text(
            self.left_x,
            0.038,
            "GPS Trajectory Map-Matching & HMM Viterbi Route Reconstruction",
            fontsize=7.5,
            color="#64748B",
            va="center",
        )
        self.ax.text(
            self.right_x,
            0.038,
            f"Page {self.page_num}",
            fontsize=7.5,
            color="#64748B",
            ha="right",
            va="center",
        )
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None
        self.ax = None

    def new_page(self) -> None:
        self._finish_page()
        self.page_num += 1
        self.fig = plt.figure(figsize=(A4_W, A4_H), dpi=200)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0])
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.axis("off")
        self.cursor_y = self.top_y

    def ensure_space(self, needed_h: float) -> None:
        if self.fig is None or (self.cursor_y - needed_h) < self.bottom_y:
            self.new_page()

    def draw_cover(self) -> None:
        self.new_page()
        ax = self.ax
        # Accent top bar
        ax.add_patch(mpatches.Rectangle((0.0, 0.965), 1.0, 0.035, facecolor="#1E3A8A"))

        ax.text(
            0.5,
            0.80,
            "GPS Trajectory Map-Matching &\nHMM Viterbi Route Reconstruction",
            ha="center",
            va="center",
            fontsize=21,
            fontweight="bold",
            color="#0F172A",
            linespacing=1.35,
        )
        ax.text(
            0.5,
            0.70,
            "Technical Report: Spatial Indexing, Multi-Model Benchmarking,\n"
            "Bayesian Optimization, Hidden Markov Route Decoding &\n"
            "Two-Laptop LAN Deployment",
            ha="center",
            va="center",
            fontsize=11,
            color="#475569",
            linespacing=1.45,
        )
        ax.plot([0.18, 0.82], [0.635, 0.635], color="#2563EB", lw=2.0)

        # Team ownership table on cover
        rows = [
            ["Role", "Ownership & Core Deliverables", "Report Sections"],
            ["Member 1", "Data Engine, Spatial Indexing (KDTree/STRtree) & EDA", "Section 2.2"],
            ["Member 2", "Feature Engineering, 5-Model GroupKFold & Optuna Tuning", "Sections 3.1-3.3, 4"],
            ["Member 3", "Hidden Markov Model & Log-Space Viterbi Decoder", "Section 5"],
            ["Member 4", "FastAPI Backend (L1), Streamlit UI (L2), Folium Maps & PDF", "Sections 6, 7 & Build"],
        ]
        self.cursor_y = 0.59
        self.draw_table(rows, col_fracs=[0.16, 0.60, 0.24])

        ax.text(
            0.5,
            0.16,
            f"Compiled automatically on {datetime.now().strftime('%Y-%m-%d %H:%M')} via report/build_report.py",
            ha="center",
            va="center",
            fontsize=8.5,
            style="italic",
            color="#64748B",
        )
        self.new_page()

    def draw_h1(self, text: str) -> None:
        wrapped = textwrap.wrap(text, width=58) or [text]
        line_h = 0.024
        self.ensure_space(len(wrapped) * line_h + 0.035)
        self.cursor_y -= 0.012
        for line in wrapped:
            self.ax.text(
                self.left_x,
                self.cursor_y,
                line,
                fontsize=13.2,
                fontweight="bold",
                color="#1E3A8A",
                va="top",
            )
            self.cursor_y -= line_h
        self.cursor_y -= 0.003
        self.ax.plot([self.left_x, self.right_x], [self.cursor_y, self.cursor_y], color="#CBD5E1", lw=0.9)
        self.cursor_y -= 0.015

    def draw_h2(self, text: str) -> None:
        self.ensure_space(0.045)
        self.cursor_y -= 0.010
        self.ax.text(
            self.left_x,
            self.cursor_y,
            text,
            fontsize=11.0,
            fontweight="bold",
            color="#0F172A",
            va="top",
        )
        self.cursor_y -= 0.026

    def draw_h3(self, text: str) -> None:
        self.ensure_space(0.035)
        self.ax.text(
            self.left_x,
            self.cursor_y,
            text,
            fontsize=9.8,
            fontweight="bold",
            color="#334155",
            va="top",
        )
        self.cursor_y -= 0.022

    def draw_paragraph(self, text: str, indent: float = 0.0, prefix: str | None = None) -> None:
        wrap_width = 92 if indent == 0.0 else 86
        wrapped = textwrap.wrap(text, width=wrap_width) or [""]
        line_h = 0.0175
        total_h = len(wrapped) * line_h + 0.010
        self.ensure_space(total_h)

        for i, line in enumerate(wrapped):
            if i == 0 and prefix is not None:
                self.ax.text(
                    self.left_x + 0.008,
                    self.cursor_y,
                    prefix,
                    fontsize=9.0,
                    fontweight="bold",
                    color="#1E3A8A",
                    va="top",
                )
            self.ax.text(
                self.left_x + indent,
                self.cursor_y,
                line,
                fontsize=9.0,
                color="#1E293B",
                va="top",
            )
            self.cursor_y -= line_h
        self.cursor_y -= 0.009

    def draw_table(self, rows: list[list[str]], col_fracs: list[float] | None = None) -> None:
        if not rows:
            return
        n_cols = max(len(r) for r in rows)
        if col_fracs is None:
            if n_cols == 4:
                col_fracs = [0.18, 0.11, 0.24, 0.47]
            else:
                col_fracs = [1.0 / n_cols] * n_cols

        total_w = self.right_x - self.left_x
        col_widths = [total_w * f for f in col_fracs]

        # Pre-wrap every cell to compute exact row heights
        wrapped_rows: list[list[list[str]]] = []
        row_heights: list[float] = []
        for r_idx, row in enumerate(rows):
            padded = row + [""] * (n_cols - len(row))
            w_row: list[list[str]] = []
            max_lines = 1
            for c_idx, cell in enumerate(padded):
                chars = max(10, int(col_widths[c_idx] * 112))
                w_lines = textwrap.wrap(cell, width=chars) or [""]
                w_row.append(w_lines)
                max_lines = max(max_lines, len(w_lines))
            wrapped_rows.append(w_row)
            row_heights.append(max_lines * 0.0155 + 0.012)

        total_h = sum(row_heights) + 0.015
        self.ensure_space(total_h)

        y_top = self.cursor_y
        for r_idx, (w_row, rh) in enumerate(zip(wrapped_rows, row_heights)):
            x_left = self.left_x
            is_header = r_idx == 0
            bg = "#1E3A8A" if is_header else ("#FFFFFF" if r_idx % 2 == 1 else "#F8FAFC")
            fg = "#FFFFFF" if is_header else "#0F172A"

            for c_idx, cell_lines in enumerate(w_row):
                cw = col_widths[c_idx]
                self.ax.add_patch(
                    mpatches.Rectangle(
                        (x_left, y_top - rh),
                        cw,
                        rh,
                        facecolor=bg,
                        edgecolor="#CBD5E1",
                        linewidth=0.6,
                    )
                )
                text_y = y_top - 0.006
                for line in cell_lines:
                    self.ax.text(
                        x_left + 0.007,
                        text_y,
                        line,
                        fontsize=7.9,
                        fontweight="bold" if is_header else "normal",
                        color=fg,
                        va="top",
                    )
                    text_y -= 0.0155
                x_left += cw
            y_top -= rh

        self.cursor_y = y_top - 0.016

    def draw_image(self, img_path: Path, caption: str) -> None:
        img_arr = mpimg.imread(str(img_path))
        img_h_px, img_w_px = img_arr.shape[0], img_arr.shape[1]
        aspect = img_h_px / max(1, img_w_px)

        box_w_in = (self.right_x - self.left_x) * A4_W
        box_h_in = min(box_w_in * aspect, 4.0)
        box_h_norm = box_h_in / A4_H

        needed_h = box_h_norm + 0.042
        self.ensure_space(needed_h)

        img_bottom = self.cursor_y - box_h_norm
        img_ax = self.fig.add_axes([self.left_x, img_bottom, self.right_x - self.left_x, box_h_norm])
        img_ax.imshow(img_arr)
        img_ax.axis("off")

        self.cursor_y = img_bottom - 0.010
        if caption:
            self.ax.text(
                0.5,
                self.cursor_y,
                f"Figure: {caption}",
                ha="center",
                va="top",
                fontsize=8.2,
                style="italic",
                color="#475569",
            )
            self.cursor_y -= 0.024

    def draw_code(self, text: str) -> None:
        lines = text.splitlines()
        line_h = 0.0145
        box_h = len(lines) * line_h + 0.016
        self.ensure_space(box_h + 0.012)
        self.ax.add_patch(
            mpatches.Rectangle(
                (self.left_x, self.cursor_y - box_h),
                self.right_x - self.left_x,
                box_h,
                facecolor="#F1F5F9",
                edgecolor="#CBD5E1",
                linewidth=0.6,
            )
        )
        y = self.cursor_y - 0.008
        for line in lines:
            self.ax.text(
                self.left_x + 0.010,
                y,
                line[:96],
                fontsize=7.4,
                family="monospace",
                color="#0F172A",
                va="top",
            )
            y -= line_h
        self.cursor_y -= box_h + 0.014

    def close(self) -> None:
        self._finish_page()


def build_technical_report(output_pdf: Path = OUTPUT_PDF) -> Path:
    """Generate architecture diagram and compile all markdown sections into Technical_Report.pdf."""
    generate_architecture_diagram(FIGURES_DIR / "system_architecture.png")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    section_files = sorted(SECTIONS_DIR.glob("section_*.md"))

    # Automatically substitute Member 1's merged `report/section_2_2_eda.md` for Section 2!
    m1_eda = REPORT_DIR / "section_2_2_eda.md"
    resolved_sections: list[Path] = []
    for sec in section_files:
        if sec.name == "section_02_eda.md" and m1_eda.exists():
            resolved_sections.append(m1_eda)
        else:
            resolved_sections.append(sec)

    with PdfPages(str(output_pdf)) as pdf:
        writer = _A4PdfWriter(pdf)
        writer.draw_cover()

        for sec_path in resolved_sections:
            blocks = _parse_markdown_blocks(sec_path.read_text(encoding="utf-8"), base_dir=REPORT_DIR)
            for block in blocks:
                btype = block["type"]
                if btype == "h1":
                    writer.draw_h1(block["text"])
                elif btype == "h2":
                    writer.draw_h2(block["text"])
                elif btype == "h3":
                    writer.draw_h3(block["text"])
                elif btype == "bullet":
                    writer.draw_paragraph(block["text"], indent=0.026, prefix=block["prefix"])
                elif btype == "code":
                    writer.draw_code(block["text"])
                elif btype == "table":
                    writer.draw_table(block["rows"])
                elif btype == "image":
                    writer.draw_image(block["path"], block["caption"])
                else:
                    writer.draw_paragraph(block["text"])

        writer.close()

        d = pdf.infodict()
        d["Title"] = "GPS Trajectory Map-Matching & HMM Viterbi Route Reconstruction"
        d["Author"] = "ML Hackathon Team (Compiled by Member 4)"
        d["Subject"] = "Technical Report"

    return output_pdf


if __name__ == "__main__":
    pdf_path = build_technical_report()
    print(f"Compiled Technical Report PDF -> {pdf_path} ({pdf_path.stat().st_size:,} bytes)")
