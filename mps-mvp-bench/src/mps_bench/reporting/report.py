"""summary.csv + self-contained report.html (no Prometheus/Grafana required).

Charts are inline SVG generated from the collected timeseries: GPU util, SM
active, DRAM active, memory, business throughput and latency, all on a shared
time axis, annotated with MPS mode / load steps / fault injection / stop /
rebuild / recovery markers. Unavailable series are drawn as explicit gaps with a
visible "unavailable" label rather than zeros.
"""

from __future__ import annotations

import csv
import html
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SUMMARY_COLUMNS = [
    "run_id", "case_id", "case_family", "mode", "round_index", "worker_id", "role",
    "workload", "unit", "mps_active_thread_pct", "mps_priority", "mps_mem_limit_bytes",
    "static_partition", "batch_size", "arrival", "target_qps",
    "window_s", "cohort_size", "success", "errors", "timeouts", "rejected", "not_sent",
    "incomplete", "output_errors",
    "throughput_req_s", "throughput_units_s", "slo_throughput_req_s",
    "success_rate", "error_rate", "timeout_rate", "reject_rate",
    "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_p999_ms",
    "latency_p999_status", "latency_samples",
    "gpu_ms_p50", "gpu_ms_p99", "queue_p99_ms", "send_delay_p99_ms",
    "sm_active_mean", "sm_active_available", "gpu_util_mean", "dram_active_mean",
    "memory_used_mib_max", "memory_free_mib_min",
    "throughput_delta_vs_B0", "throughput_delta_vs_B2",
    "latency_delta_vs_B0", "latency_delta_vs_B2",
    "status", "status_reason", "synthetic",
]


def write_summary_csv(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


# --------------------------------------------------------------------------- #
# inline SVG line chart
# --------------------------------------------------------------------------- #

def _svg_line_chart(title: str, series: Sequence[Dict[str, Any]],
                    markers: Sequence[Dict[str, Any]] = (),
                    width: int = 900, height: int = 220,
                    y_label: str = "") -> str:
    """series item: {name, points:[(x,y|None)], unavailable_note}"""
    pad_l, pad_r, pad_t, pad_b = 60, 150, 28, 30
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    xs = [x for s in series for x, y in s["points"] if y is not None]
    ys = [y for s in series for x, y in s["points"] if y is not None]
    if not xs or not ys:
        note = "; ".join(s.get("unavailable_note") or s["name"] for s in series)
        return (f'<div class="chart"><h4>{html.escape(title)}</h4>'
                f'<p class="unavailable">指标不可用，无数据点（不以 0 填充、不以其他指标代替）：'
                f'{html.escape(note)}</p></div>')
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys + [0.0]), max(ys)
    if y_max == y_min:
        y_max = y_min + 1.0
    x_span = (x_max - x_min) or 1.0

    def sx(x: float) -> float:
        return pad_l + (x - x_min) / x_span * plot_w

    def sy(y: float) -> float:
        return pad_t + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chartsvg" role="img" '
             f'aria-label="{html.escape(title)}">']
    parts.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
                 f'fill="#fbfbfd" stroke="#ccc"/>')
    for frac in (0.0, 0.5, 1.0):
        y = y_min + (y_max - y_min) * frac
        parts.append(f'<line x1="{pad_l}" y1="{sy(y):.1f}" x2="{pad_l + plot_w}" '
                     f'y2="{sy(y):.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{sy(y) + 4:.1f}" text-anchor="end" '
                     f'class="tick">{y:.3g}</text>')
    parts.append(f'<text x="{pad_l}" y="{height - 8}" class="tick">{x_min:.0f}s</text>')
    parts.append(f'<text x="{pad_l + plot_w}" y="{height - 8}" text-anchor="end" '
                 f'class="tick">{x_max:.0f}s</text>')
    if y_label:
        parts.append(f'<text x="6" y="{pad_t + 10}" class="tick">{html.escape(y_label)}</text>')

    for i, s in enumerate(series):
        color = palette[i % len(palette)]
        # split into segments so gaps (None) are visible, not interpolated
        segment: List[str] = []

        def flush(seg: List[str]) -> None:
            if len(seg) > 1:
                parts.append(f'<polyline points="{" ".join(seg)}" fill="none" '
                             f'stroke="{color}" stroke-width="1.5"/>')
            elif len(seg) == 1:
                # A single sample surrounded by gaps still has to be visible; a
                # one-point polyline draws nothing, so render it as a dot rather
                # than silently dropping a real measurement.
                cx, cy = seg[0].split(",")
                parts.append(f'<circle cx="{cx}" cy="{cy}" r="2" fill="{color}"/>')

        for x, y in s["points"]:
            if y is None:
                flush(segment)
                segment = []
                continue
            segment.append(f"{sx(x):.1f},{sy(y):.1f}")
        flush(segment)
        ly = pad_t + 14 * i + 10
        parts.append(f'<line x1="{pad_l + plot_w + 10}" y1="{ly}" x2="{pad_l + plot_w + 30}" '
                     f'y2="{ly}" stroke="{color}" stroke-width="2"/>')
        label = s["name"] + (f' ({s["unavailable_note"]})' if s.get("unavailable_note") else "")
        parts.append(f'<text x="{pad_l + plot_w + 34}" y="{ly + 4}" class="legend">'
                     f'{html.escape(label)}</text>')

    for mark in markers:
        x = mark.get("t")
        if x is None or not (x_min <= x <= x_max):
            continue
        parts.append(f'<line x1="{sx(x):.1f}" y1="{pad_t}" x2="{sx(x):.1f}" '
                     f'y2="{pad_t + plot_h}" stroke="#888" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{sx(x) + 3:.1f}" y="{pad_t + 12}" class="marker">'
                     f'{html.escape(str(mark.get("label", "")))}</text>')
    parts.append("</svg>")
    return f'<div class="chart"><h4>{html.escape(title)}</h4>{"".join(parts)}</div>'


def _read_metrics(csv_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(row: Dict[str, Any], key: str) -> Optional[float]:
    raw = row.get(key)
    if raw in (None, "", "None"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _series_from_metrics(rows: List[Dict[str, Any]], key: str, name: str,
                         scale: float = 1.0) -> Dict[str, Any]:
    points: List[Tuple[float, Optional[float]]] = []
    missing = 0
    for row in rows:
        t = _num(row, "monotonic_s")
        v = _num(row, key)
        if t is None:
            continue
        points.append((t, None if v is None else v * scale))
        if v is None:
            missing += 1
    note = ""
    if points and missing == len(points):
        note = "unavailable"
    elif missing:
        note = f"{missing}/{len(points)} 采样缺失"
    return {"name": name, "points": points, "unavailable_note": note}


_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#222;
line-height:1.55}
h1{font-size:22px}h2{font-size:18px;margin-top:28px;border-bottom:1px solid #eee;padding-bottom:4px}
h3{font-size:15px}h4{font-size:14px;margin:8px 0}
table{border-collapse:collapse;font-size:12px;margin:8px 0;width:100%}
th,td{border:1px solid #ddd;padding:4px 6px;text-align:right}
th{background:#f5f6f8;text-align:center}
td.l,th.l{text-align:left}
.tick,.legend,.marker{font-size:10px;fill:#555}
.chart{margin:12px 0}
.chartsvg{width:100%;height:auto;border:1px solid #eee;background:#fff}
.unavailable{color:#b00;background:#fff4f4;border-left:3px solid #b00;padding:6px 10px;
font-size:13px}
.warn{background:#fffaf0;border-left:3px solid #e69500;padding:8px 12px;font-size:13px}
.ok{color:#1a7f37}.fail{color:#b00}.skip{color:#666}.inconclusive{color:#a15c00}
code,pre{font-family:SFMono-Regular,Consolas,monospace;font-size:12px}
pre{background:#f7f7f9;padding:10px;overflow:auto}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;border:1px solid #ccc}
"""

_STATUS_CLASS = {"PASS": "ok", "FAIL": "fail", "SKIP": "skip",
                 "INCONCLUSIVE": "inconclusive", "NOT_RUN": "skip"}


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
           left_cols: Sequence[int] = ()) -> str:
    def cell(i: int, v: Any, tag: str) -> str:
        cls = ' class="l"' if i in left_cols else ""
        text = "n/a" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
        return f"<{tag}{cls}>{html.escape(text)}</{tag}>"
    head = "".join(cell(i, h, "th") for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(cell(i, v, "td") for i, v in enumerate(r)) + "</tr>"
                   for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(run_dir: str,
                run_id: str,
                summary_rows: Sequence[Dict[str, Any]],
                mvp_status: Sequence[Dict[str, Any]],
                events: Sequence[Dict[str, Any]],
                capability: Dict[str, Any],
                environment: Dict[str, Any],
                telemetry_status: Dict[str, Any],
                findings: Dict[str, Any],
                fault_summary: Sequence[Dict[str, Any]] = (),
                memory_summary: Sequence[Dict[str, Any]] = (),
                not_verified: Sequence[str] = ()) -> str:
    metrics = _read_metrics(os.path.join(run_dir, "gpu_metrics.csv"))
    markers = [{"t": e.get("monotonic_s"), "label": e.get("label") or e.get("event")}
               for e in events if e.get("monotonic_s") is not None]

    charts = [
        _svg_line_chart("GPU util (device busy) vs SM active vs DRAM active", [
            _series_from_metrics(metrics, "gpu_util_device_busy_pct", "GPU util (device busy) %"),
            _series_from_metrics(metrics, "sm_active", "DCGM SM active %", scale=100.0),
            _series_from_metrics(metrics, "sm_occupancy", "DCGM SM occupancy %", scale=100.0),
            _series_from_metrics(metrics, "dram_active", "DCGM DRAM active %", scale=100.0),
        ], markers=markers, y_label="%"),
        _svg_line_chart("显存 (MiB)", [
            _series_from_metrics(metrics, "memory_used_mib", "used MiB"),
            _series_from_metrics(metrics, "memory_free_mib", "free MiB"),
        ], markers=markers, y_label="MiB"),
        _svg_line_chart("功耗 / 温度 / SM 时钟", [
            _series_from_metrics(metrics, "power_w", "power W"),
            _series_from_metrics(metrics, "temperature_c", "temp C"),
            _series_from_metrics(metrics, "sm_clock_mhz", "SM clock MHz"),
        ], markers=markers),
    ]

    ts_series = findings.get("business_timeseries") or []
    if ts_series:
        charts.append(_svg_line_chart(
            "业务每秒成功吞吐", [{"name": s["name"], "points": s["points"]} for s in ts_series],
            markers=markers, y_label="req/s"))
    lat_series = findings.get("latency_timeseries") or []
    if lat_series:
        charts.append(_svg_line_chart(
            "业务秒级延迟（样本不足的秒级分位数仅作参考，不作可靠分位数）",
            [{"name": s["name"], "points": s["points"]} for s in lat_series],
            markers=markers, y_label="ms"))

    degraded_banner = ""
    if not telemetry_status.get("sm_active_available"):
        degraded_banner = (
            '<p class="unavailable"><strong>可观测验收缺项</strong>：'
            'DCGM_FI_PROF_SM_ACTIVE 不可用（' +
            html.escape(str(telemetry_status.get("dcgm_reason") or "原因未记录")) +
            '）。本报告不得宣称已完成 SM 可观测验收；相关曲线以缺失显示，未用 GPU util 代替。</p>')

    synthetic = any(r.get("synthetic") for r in summary_rows)
    synth_banner = ('<p class="warn">本次结果使用 <strong>synthetic</strong> 权重/输入：'
                    '仅验证系统行为与相对性能，<strong>不代表真实模型精度或生产收益</strong>。</p>'
                    if synthetic else "")

    mvp_rows = [(m.get("mvp_item"), m.get("case_id"), m.get("config_summary"),
                 f'<span class="badge {_STATUS_CLASS.get(m.get("status"), "")}">'
                 f'{html.escape(str(m.get("status")))}</span>',
                 m.get("reason"), m.get("evidence")) for m in mvp_status]
    mvp_table = ("<table><thead><tr><th class='l'>MVP 项</th><th class='l'>用例</th>"
                 "<th class='l'>实际配置</th><th>状态</th><th class='l'>原因</th>"
                 "<th class='l'>证据</th></tr></thead><tbody>" +
                 "".join("<tr>" + "".join(
                     f"<td class='l'>{v if i == 3 else html.escape(str(v))}</td>"
                     for i, v in enumerate(row)) + "</tr>" for row in mvp_rows) +
                 "</tbody></table>")

    perf_headers = ["case", "worker", "role", "unit", "ATP%", "prio", "Q req/s",
                    "units/s", "P99 ms", "Δtp vs B0", "Δtp vs B2", "ΔP99 vs B0",
                    "ΔP99 vs B2", "err", "SM active"]
    perf_rows = [[r.get("case_id"), r.get("worker_id"), r.get("role"), r.get("unit"),
                  r.get("mps_active_thread_pct"), r.get("mps_priority"),
                  r.get("throughput_req_s"), r.get("throughput_units_s"),
                  r.get("latency_p99_ms"), r.get("throughput_delta_vs_B0"),
                  r.get("throughput_delta_vs_B2"), r.get("latency_delta_vs_B0"),
                  r.get("latency_delta_vs_B2"), r.get("error_rate"),
                  r.get("sm_active_mean")] for r in summary_rows]

    fault_table = ""
    if fault_summary:
        fault_table = _table(
            ["用例", "injection_status", "peer_outcome", "传播次数/总次数",
             "检测(s)", "首个成功请求(s)", "稳定恢复(s)", "失败请求", "超时请求", "恢复层级"],
            [[f.get("case_id"), f.get("injection_status"), f.get("peer_outcome"),
              f"{f.get('propagations', 0)}/{f.get('injections', 0)}",
              f.get("detect_s"), f.get("first_success_s"), f.get("stable_recovery_s"),
              f.get("failed_requests"), f.get("timeout_requests"),
              f.get("recovery_level")] for f in fault_summary], left_cols=(0, 1, 2, 9))

    mem_table = ""
    if memory_summary:
        mem_table = _table(
            ["用例", "client", "配额 bytes", "逻辑分配 MiB", "框架 allocated MiB",
             "框架 reserved MiB", "整卡 used MiB", "整卡 free MiB", "配额 OOM", "意外 OOM",
             "CUDA 错误码", "释放后可继续计算", "对端 QoS"],
            [[m.get("case_id"), m.get("client"), m.get("quota_bytes"),
              m.get("logical_allocated_mib"), m.get("framework_allocated_mib"),
              m.get("framework_reserved_mib"), m.get("device_used_mib"),
              m.get("device_free_mib"), m.get("expected_quota_oom"),
              m.get("unexpected_oom"), m.get("cuda_error"),
              m.get("recovered_compute"), m.get("peer_qos")] for m in memory_summary],
            left_cols=(0, 1, 10, 12))

    def _list(items: Iterable[str]) -> str:
        items = list(items)
        if not items:
            return "<p>无</p>"
        return "<ul>" + "".join(f"<li>{html.escape(str(i))}</li>" for i in items) + "</ul>"

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MPS MVP 验证报告 {html.escape(run_id)}</title><style>{_CSS}</style></head><body>
<h1>MPS MVP 验证报告 <code>{html.escape(run_id)}</code></h1>
{synth_banner}{degraded_banner}

<h2>1. MVP 项状态</h2>
{mvp_table}

<h2>2. 性能对照（同时给出相对 B0 独占与 B2 同卡非 MPS）</h2>
<p>Δtp = Q_test/Q_ref - 1；ΔP99 = P99_test/P99_ref - 1。
受限单跑（B1）用于解释限额成本，不等同完整 GPU 独占能力。</p>
{_table(perf_headers, perf_rows, left_cols=(0, 1, 2, 3, 5))}

<h2>3. 正例 / 负例</h2>
<h3>正例</h3>{_list(findings.get("positive", []))}
<h3>负例</h3>{_list(findings.get("negative", []))}
<h3>本配置下未观察到</h3>{_list(findings.get("not_reproduced", []))}
<h3>扫描记录</h3><pre>{html.escape(json.dumps(findings.get("sweeps", []), ensure_ascii=False, indent=2))}</pre>

<h2>4. 显存配额专项</h2>
{mem_table or "<p>本次未执行显存专项</p>"}

<h2>5. 故障传播与恢复</h2>
<p>容器 RUNNING、MPS ACTIVE、业务恢复是不同事件，分别记录。未观察到传播只表述为
&quot;0/n 次&quot;，不表述为完全隔离。</p>
{fault_table or "<p>本次未执行故障用例</p>"}

<h2>6. 时间序列</h2>
<p>GPU util 是设备 busy 指标；SM active 是 DCGM_FI_PROF_SM_ACTIVE；SM occupancy 是
DCGM_FI_PROF_SM_OCCUPANCY。三者语义不同，SM active 高不等于算术单元满载，也不证明吞吐提升。</p>
{"".join(charts)}

<h2>7. 尚未验证的能力</h2>
{_list(not_verified)}

<h2>8. 环境与能力</h2>
<pre>{html.escape(json.dumps(environment, ensure_ascii=False, indent=2))}</pre>
<pre>{html.escape(json.dumps(capability.get("summary", capability), ensure_ascii=False, indent=2))}</pre>

<h2>9. 事件时间线</h2>
<pre>{html.escape(json.dumps(list(events)[:400], ensure_ascii=False, indent=2))}</pre>
</body></html>"""
    path = os.path.join(run_dir, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
    return path
