#!/usr/bin/env python3
"""
generate_report.py — build a static HTML report from eval_detailed output.

Reads:
    <EVAL_DIR>/summary.json
    <EVAL_DIR>/*.png
    <EVAL_DIR>/attribute_analysis/*.png

Writes:
    <EVAL_DIR>/report.html    (or wherever OUTPUT_HTML points)

When EMBED_IMAGES is True (default) PNGs are inlined as data: URIs so the
report.html is a single portable file that survives being copied around.
Set False to keep them as relative paths (smaller HTML, folder must travel
together).

Usage
----

    import and call directly:
        from generate_report import generate_report
        generate_report("eval_results/", title="BuzzSet — Test split")
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path


# Human-readable labels for attribute values. Same scheme as eval_detailed.py.
# Keys are stringified because they're read from JSON.
ATTR_LABELS = {
    "blur":      {"0": "Sharp", "1": "Blurry"},
    "occluded":  {"1": "full vis", "2": "self occ.", "3": "obj. occ."},
}


# HELPERS

def fmt(v, nd: int = 4) -> str:
    """Format a value for table display. NaN / None → em-dash."""
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:                        # NaN
            return "—"
        if v == int(v) and abs(v) < 1e9:  # ints stored as float
            return f"{int(v)}"
        return f"{v:.{nd}f}"
    return str(v)


def png_src(path: Path, embed: bool, rel_root: Path):
    """Return either a data: URI or a relative path. None if file is missing."""
    if not path.exists():
        return None
    if embed:
        b = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b}"
    return str(path.relative_to(rel_root)).replace("\\", "/")


def maybe_figure(src, caption: str) -> str:
    if not src:
        return ""
    return (
        '<figure class="plot">'
        f'<img src="{src}" alt="{html.escape(caption)}">'
        f'<figcaption class="plot-caption">{html.escape(caption)}</figcaption>'
        '</figure>'
    )


# SECTION RENDERERS

def render_header(summary: dict, title: str) -> str:
    cfg = summary.get("config", {})
    bits = [
        ("GT",          Path(cfg.get("gt_json", "—")).name),
        ("PRED",        Path(cfg.get("pred_json", "—")).name),
        ("CONF",        fmt(cfg.get("confidence_threshold"), 2)),
        ("IOU",         fmt(cfg.get("iou_threshold"), 2)),
        ("IMAGES",      fmt(summary.get("n_images"))),
        ("ANNOTATIONS", fmt(summary.get("n_gt"))),
        ("PREDICTIONS", fmt(summary.get("n_predictions_total"))),
    ]
    meta = "".join(
        f'<span><strong>{html.escape(k)}</strong>{html.escape(str(v))}</span>'
        for k, v in bits
    )
    return f"""\
<header class="banner">
  <div class="eyebrow">Evaluation Report</div>
  <h1 class="title">{html.escape(title)}</h1>
  <div class="meta">{meta}</div>
</header>"""


def render_kpis(summary: dict) -> str:
    cfg = summary.get("config", {})
    conf_t = cfg.get("confidence_threshold", 0.5)
    coco = summary.get("coco", {})

    cards = [
        ("mAP @.50:.95", fmt(coco.get("mAP@50:95"), 3), "COCO"),
        ("mAP @.50",     fmt(coco.get("mAP@50"), 3),    "COCO"),
        ("Precision",    fmt(summary.get(f"precision@{conf_t}"), 3), f"conf ≥ {conf_t}"),
        ("Recall",       fmt(summary.get(f"recall@{conf_t}"), 3),    f"conf ≥ {conf_t}"),
        ("F1",           fmt(summary.get(f"f1@{conf_t}"), 3),        f"conf ≥ {conf_t}"),
    ]
    items = "".join(
        f'<div class="kpi">'
        f'<div class="kpi-label">{html.escape(lab)}</div>'
        f'<div class="kpi-value">{html.escape(val)}</div>'
        f'<div class="kpi-sub">{html.escape(sub)}</div>'
        '</div>'
        for lab, val, sub in cards
    )
    return f'<div class="kpis">{items}</div>'


def render_coco_section(summary: dict) -> str:
    coco = summary.get("coco", {})
    left = [
        ("mAP @.50:.95", coco.get("mAP@50:95")),
        ("mAP @.50",     coco.get("mAP@50")),
        ("mAP @.75",     coco.get("mAP@75")),
        ("mAP small",    coco.get("mAP_small")),
        ("mAP medium",   coco.get("mAP_medium")),
        ("mAP large",    coco.get("mAP_large")),
    ]
    right = [
        ("AR @1",        coco.get("AR@1")),
        ("AR @10",       coco.get("AR@10")),
        ("AR @100",      coco.get("AR@100")),
        ("AR small",     coco.get("AR_small")),
        ("AR medium",    coco.get("AR_medium")),
        ("AR large",     coco.get("AR_large")),
    ]

    def rows(items):
        return "".join(
            f"<tr><td>{html.escape(k)}</td><td class='num'>{fmt(v, 4)}</td></tr>"
            for k, v in items
        )

    return f"""\
<section>
  <h2 class="section-title">Official COCO metrics</h2>
  <p class="section-lede">Computed by pycocotools over the full prediction set,
    independent of the confidence threshold.</p>
  <div class="two-col">
    <table>
      <thead><tr><th>Precision</th><th class="num">Value</th></tr></thead>
      <tbody>{rows(left)}</tbody>
    </table>
    <table>
      <thead><tr><th>Recall</th><th class="num">Value</th></tr></thead>
      <tbody>{rows(right)}</tbody>
    </table>
  </div>
</section>"""


def render_per_class(summary: dict, eval_dir: Path, embed: bool) -> str:
    pcap = summary.get("per_class_AP50", {}) or {}
    pcgt = summary.get("per_class_n_gt", {}) or {}

    body = "".join(
        f"<tr><td>{html.escape(cls)}</td>"
        f"<td class='num'>{fmt(pcap.get(cls), 4)}</td>"
        f"<td class='num'>{fmt(pcgt.get(cls, 0))}</td></tr>"
        for cls in pcap
    )
    img = maybe_figure(
        png_src(eval_dir / "per_class_ap50.png", embed, eval_dir),
        "per_class_ap50.png"
    )
    return f"""\
<section>
  <h2 class="section-title">Per-class performance</h2>
  <p class="section-lede">AP@.50 from the script's own greedy matching at the
    configured IoU threshold.</p>
  <div class="two-col">
    <table>
      <thead><tr>
        <th>Class</th><th class="num">AP@.50</th><th class="num">n_gt</th>
      </tr></thead>
      <tbody>{body}</tbody>
    </table>
    {img}
  </div>
</section>"""


def render_diagnostics(summary: dict, eval_dir: Path, embed: bool) -> str:
    cfg = summary.get("config", {})
    conf_t = cfg.get("confidence_threshold", 0.5)
    iou_t = cfg.get("iou_threshold", 0.5)

    tp = summary.get(f"TP@{conf_t}", 0)
    fp = summary.get(f"FP@{conf_t}", 0)
    fn = summary.get(f"FN@{conf_t}", 0)
    p  = summary.get(f"precision@{conf_t}")
    r  = summary.get(f"recall@{conf_t}")
    f1 = summary.get(f"f1@{conf_t}")

    counts = f"""\
<table class="tight">
  <thead><tr>
    <th class="num">TP</th>
    <th class="num">FP</th>
    <th class="num">FN</th>
    <th class="num">Precision</th>
    <th class="num">Recall</th>
    <th class="num">F1</th>
  </tr></thead>
  <tbody><tr>
    <td class="num">{fmt(tp)}</td>
    <td class="num">{fmt(fp)}</td>
    <td class="num">{fmt(fn)}</td>
    <td class="num">{fmt(p, 4)}</td>
    <td class="num">{fmt(r, 4)}</td>
    <td class="num">{fmt(f1, 4)}</td>
  </tr></tbody>
</table>"""

    plots = [
        ("pr_curves.png",         eval_dir / "pr_curves.png"),
        ("confusion_matrix.png",  eval_dir / "confusion_matrix.png"),
        ("score_histogram.png",   eval_dir / "score_histogram.png"),
        ("size_distribution.png", eval_dir / "size_distribution.png"),
    ]
    plot_html = "".join(
        maybe_figure(png_src(p, embed, eval_dir), cap)
        for cap, p in plots
    )

    return f"""\
<section>
  <h2 class="section-title">Match-level diagnostics</h2>
  <p class="section-lede">Greedy IoU matching at threshold {fmt(iou_t, 2)},
    confidence threshold {fmt(conf_t, 2)}. The matrix is class-agnostic so
    inter-class confusion lands off-diagonal.</p>
  {counts}
  <div class="plot-grid">{plot_html}</div>
</section>"""


def render_attributes(summary: dict, eval_dir: Path, embed: bool) -> str:
    attr_metrics = summary.get("attribute_metrics", {}) or {}
    if not attr_metrics:
        return ""

    overview = maybe_figure(
        png_src(eval_dir / "attribute_analysis" / "attribute_overview.png",
                embed, eval_dir),
        "attribute_overview.png"
    )

    blocks = []
    for attr_name, buckets in attr_metrics.items():
        label_map = ATTR_LABELS.get(attr_name, {})
        # Sort buckets by their integer-cast key if possible.
        def sort_key(item):
            try:    return (0, int(item[0]))
            except: return (1, str(item[0]))

        rows = []
        for val_key, d in sorted(buckets.items(), key=sort_key):
            label = label_map.get(str(val_key), str(val_key))
            rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td class='num'>{fmt(d.get('n_gt'))}</td>"
                f"<td class='num'>{fmt(d.get('n_tp'))}</td>"
                f"<td class='num'>{fmt(d.get('n_fn'))}</td>"
                f"<td class='num'>{fmt(d.get('recall'), 4)}</td>"
                f"<td class='num'>{fmt(d.get('mean_conf'), 3)}</td>"
                "</tr>"
            )

        recall_img = png_src(
            eval_dir / "attribute_analysis" / f"{attr_name}_recall.png",
            embed, eval_dir)
        conf_img = png_src(
            eval_dir / "attribute_analysis" / f"{attr_name}_confidence.png",
            embed, eval_dir)

        blocks.append(f"""\
<div class="attr-block">
  <h3 class="subsection-title">{html.escape(attr_name)}</h3>
  <table>
    <thead><tr>
      <th>Level</th>
      <th class="num">n_gt</th>
      <th class="num">n_tp</th>
      <th class="num">n_fn</th>
      <th class="num">Recall</th>
      <th class="num">Mean TP conf</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <div class="plot-grid">
    {maybe_figure(recall_img, f'{attr_name}_recall.png')}
    {maybe_figure(conf_img,   f'{attr_name}_confidence.png')}
  </div>
</div>""")

    return f"""\
<section>
  <h2 class="section-title">Attribute-stratified recall</h2>
  <p class="section-lede">How the model performs on subsets of the ground
    truth defined by per-annotation attributes.</p>
  {overview}
  {''.join(blocks)}
</section>"""


def render_footer(eval_dir: Path) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""\
<footer>
  <span>generated {ts}</span>
  <span>·</span>
  <span>source <code>{html.escape(str(eval_dir))}</code></span>
  <span>·</span>
  <span>raw <a href="summary.json">summary.json</a></span>
</footer>"""


# CSS (kept inline so the report is a single file)

CSS = """
:root {
  --bg: #FAF7F2;
  --surface: #F2EDE2;
  --border: #E7E0D3;
  --text: #1C1917;
  --muted: #57534E;
  --accent: #B45309;
}

* { box-sizing: border-box; }

html, body {
  margin: 0; padding: 0;
  background: var(--bg);
  color: var(--text);
  font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}

.container { max-width: 1100px; margin: 0 auto; padding: 56px 32px 80px; }

.banner {
  border-top: 1px solid var(--text);
  border-bottom: 1px solid var(--text);
  padding: 26px 0 28px;
  margin-bottom: 56px;
}

.eyebrow {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 12px;
}

.title {
  font-family: "Fraunces", Georgia, serif;
  font-size: 48px;
  line-height: 1.02;
  font-weight: 500;
  font-variation-settings: "opsz" 144, "SOFT" 50;
  letter-spacing: -0.018em;
  margin: 0 0 18px;
}

.meta {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11.5px;
  color: var(--muted);
  display: flex;
  gap: 22px;
  flex-wrap: wrap;
}

.meta span strong {
  color: var(--text);
  font-weight: 500;
  margin-right: 6px;
}

.kpis {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  margin-bottom: 64px;
}

.kpi { background: var(--bg); padding: 22px 24px; }

.kpi-label {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 10.5px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
}

.kpi-value {
  font-family: "Fraunces", Georgia, serif;
  font-size: 40px;
  font-weight: 500;
  font-variation-settings: "opsz" 144;
  line-height: 1;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}

.kpi-sub {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px;
  color: var(--muted);
  margin-top: 8px;
}

section { margin-bottom: 64px; }

.section-title {
  font-family: "Fraunces", Georgia, serif;
  font-size: 28px;
  font-weight: 500;
  font-variation-settings: "opsz" 36;
  letter-spacing: -0.01em;
  margin: 0 0 6px;
}

.subsection-title {
  font-family: "Fraunces", Georgia, serif;
  font-size: 18px;
  font-weight: 500;
  font-variation-settings: "opsz" 24;
  text-transform: capitalize;
  margin: 32px 0 14px;
  color: var(--accent);
}

.section-lede {
  color: var(--muted);
  font-size: 14px;
  margin: 0 0 26px;
  max-width: 64ch;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 12.5px;
}

th, td {
  text-align: left;
  padding: 11px 14px;
  border-bottom: 1px solid var(--border);
}

th {
  font-weight: 500;
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  background: var(--surface);
}

td.num, th.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

tbody tr:last-child td { border-bottom: 1px solid var(--text); }

table.tight { margin-bottom: 26px; }

.two-col, .plot-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  align-items: start;
}

.plot {
  margin: 0;
  border: 1px solid var(--border);
  background: white;
  overflow: hidden;
}

.plot img {
  display: block;
  width: 100%;
  height: auto;
}

.plot-caption {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 10.5px;
  color: var(--muted);
  padding: 9px 14px;
  border-top: 1px solid var(--border);
  background: var(--surface);
  letter-spacing: 0.04em;
}

.attr-block { margin-top: 32px; }

footer {
  margin-top: 96px;
  padding-top: 22px;
  border-top: 1px solid var(--border);
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px;
  color: var(--muted);
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

footer a, footer code { color: var(--accent); }
footer a { text-decoration: none; }
footer a:hover { text-decoration: underline; }
footer code { font-family: inherit; }

@media (max-width: 760px) {
  .container { padding: 32px 20px 64px; }
  .title { font-size: 34px; }
  .two-col, .plot-grid { grid-template-columns: 1fr; }
}
"""


# ASSEMBLY

FONT_LINK = (
  '<link rel="preconnect" href="https://fonts.googleapis.com">'
  '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
  '<link href="https://fonts.googleapis.com/css2?'
  'family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600'
  '&family=IBM+Plex+Mono:wght@400;500'
  '&family=IBM+Plex+Sans:wght@400;500;600'
  '&display=swap" rel="stylesheet">'
)


def build_html(summary: dict, eval_dir: Path, embed: bool, title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
{FONT_LINK}
<style>{CSS}</style>
</head>
<body>
<div class="container">
{render_header(summary, title)}
{render_kpis(summary)}
{render_coco_section(summary)}
{render_per_class(summary, eval_dir, embed)}
{render_diagnostics(summary, eval_dir, embed)}
{render_attributes(summary, eval_dir, embed)}
{render_footer(eval_dir)}
</div>
</body>
</html>"""


def generate_report(eval_dir, output_html=None,
                    embed_images: bool = True,
                    title: str = "Evaluation Report") -> Path:
    """Build report.html from summary.json + PNGs in `eval_dir`."""

    eval_dir = Path(eval_dir)
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out = Path(output_html) if output_html else (eval_dir / "report.html")
    html_str = build_html(summary, eval_dir, embed_images, title)
    out.write_text(html_str, encoding="utf-8")

    size_kb = out.stat().st_size / 1024
    mode = "embedded" if embed_images else "linked"
    print(f"Wrote {out.resolve()}")
    print(f"  size : {size_kb:,.1f} KB  ({mode} images)")
    
    return out


if __name__ == "__main__":
    generate_report("/path/to/EVAL_DIR")
