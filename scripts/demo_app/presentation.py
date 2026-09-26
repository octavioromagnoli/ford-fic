"""Presentation only: cockpit styles, the brand lockup, the fleet calendar, the guided tour, messages, tables and
exact-value metric animation."""
from base64 import b64encode
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

HERE = Path(__file__).parent
PANEL = "#0c1a2f"   # --panel in tokens.css: Vega-Lite needs the literal value


def inject_theme():
    css = (HERE / "theme.css").read_text()
    tokens = (HERE / "tokens.css").read_text()
    # theme.css first: it opens with the font @import, which the browser ignores anywhere but at the top.
    st.markdown(f"<style>{css}\n{tokens}</style>", unsafe_allow_html=True)


@lru_cache(maxsize=1)
def _ford_script() -> str:
    # The Ford script, extracted from Wikimedia Commons' "Ford logo flat.svg" (public domain file; Ford trademark).
    return "data:image/svg+xml;base64," + b64encode((HERE / "assets" / "ford-script.svg").read_bytes()).decode()


def brand_html(product: str, extra_class: str = "") -> str:
    """`Ford DPF`: the Ford script for the first word, a hairline, the rest as type."""
    first, _, rest = product.partition(" ")
    mark = f'<img src="{_ford_script()}" alt="Ford">' if first == "Ford" else escape(first)
    classes = f"brand {extra_class}".strip()
    return f'<div class="{classes}">{mark}<span>{escape(rest)}</span></div>'


def brand_block(product: str) -> str:
    """The lockup plus what the product is and where it comes from (sidebar and login)."""
    return (f'<div class="brand-block">{brand_html(product)}'
            '<p class="brand-tag">Posventa predictiva del filtro de partículas diésel</p>'
            '<p class="brand-event">Ford Innovation Challenge III</p></div>')


@lru_cache(maxsize=None)
def _component(name: str):
    # Registered once per process: registering again on every rerun logs a warning. Not isolated, so the styles
    # come from theme.css and the tour can find its targets inside the calendar.
    return st.components.v2.component(f"fdpf_{name}", js=(HERE / f"{name}.js").read_text(), isolate_styles=False)


def timeline(data: dict, on_week) -> None:
    """The fleet calendar (timeline.js): `on_week` runs when a week is chosen, with the Monday in
    `st.session_state["timeline"]["week"]`."""
    _component("timeline")(data=data, key="timeline", on_week_change=on_week)


def tour(steps: list[dict]) -> None:
    """The «Cómo usar» button and the guided tour it opens (tour.js). Runs in the browser, without a rerun."""
    _component("tour")(data={"steps": steps, "label": "Cómo usar"}, key="tour_button", width="content")


def heading(title, subtitle=""):
    st.markdown(f'<div class="page-heading"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def message(text, subject=False, note=None):
    """The message as sent, word for word, with its structure made visible: the subject (driver messages),
    the labels that open a list ("Qué podés hacer:"), list items with a hanging dash, and the closing note."""
    paragraphs = text.split("\n\n")
    html = []
    for i, para in enumerate(paragraphs):
        if subject and i == 0:
            html.append(f'<p class="msg-subject">{escape(para)}</p>')
            continue
        if note and i == len(paragraphs) - 1 and para.strip() == note.strip():
            html.append(f'<p class="msg-note">{escape(para)}</p>')
            continue
        lines = para.split("\n")
        spans = []
        for j, line in enumerate(lines):
            if line.startswith("- "):
                spans.append(f'<span class="msg-li">{escape(line)}</span>')
            elif j == 0 and len(lines) > 1 and line.rstrip().endswith(":"):
                spans.append(f'<span class="msg-label">{escape(line)}</span>')
            else:
                spans.append(f"<span>{escape(line)}</span>")
        html.append("<p>" + "".join(spans) + "</p>")
    st.html(f'<div class="message-body">{"".join(html)}</div>')


def agent_trace(steps, lead="", note=""):
    """The agent's trace as numbered steps: a title, a detail, the result (`ok`, `fail` or the policy's
    `policy` answer, each with its icon from theme.css) and a quote. Every string is escaped."""
    items = []
    for i, s in enumerate(steps, start=1):
        parts = [f'<p class="trace-title">{escape(s["title"])}</p>']
        if s.get("detail"):
            parts.append(f'<p class="trace-detail">{escape(s["detail"])}</p>')
        if s.get("result"):
            kind, text = s["result"]
            parts.append(f'<p class="trace-result is-{kind}"><span>{escape(text)}</span></p>')
        if s.get("quote"):
            parts.append(f'<p class="trace-quote">«{escape(s["quote"])}»</p>')
        items.append(f'<li><span class="trace-n" aria-hidden="true">{i}</span><div>{"".join(parts)}</div></li>')
    body = (f'<p class="trace-lead">{escape(lead)}</p>' if lead else "") + f'<ol class="trace">{"".join(items)}</ol>' \
        + (f'<p class="trace-note">{escape(note)}</p>' if note else "")
    st.html(f'<div class="agent-trace">{body}</div>')


def table(columns, rows, first_is_header=True, nowrap_values=False, stack=False):
    """A static HTML table: it wraps its text instead of scrolling sideways like st.dataframe on a phone.

    `nowrap_values` keeps short values ("17,0% ± 3,8") on one line and lets the row labels wrap instead;
    `stack` turns each row into label/value pairs when the table is narrow (long cells, like the agent trace)."""
    head = "".join(f'<th scope="col">{escape(str(c))}</th>' for c in columns)
    body = "".join("<tr>" + "".join(
        f'<th scope="row">{escape(str(v))}</th>' if i == 0 and first_is_header
        else f'<td data-label="{escape(str(columns[i]))}">{escape(str(v))}</td>'
        for i, v in enumerate(row)) + "</tr>" for row in rows)
    cls = "data-table" + (" nowrap-values" if nowrap_values else "") + (" stack" if stack else "")
    st.html(f'<div class="{cls}"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


def metrics(items):
    """Animate integer counts only; the final string is always the supplied value.

    The tiles mirror st.metric in theme.css (surface, label, value) so the Bandeja reads like the other pages."""
    cards = "".join(f'<article><label>{escape(label)}</label><strong data-value="{escape(str(value))}">{escape(str(value))}</strong>'
                    f'<span>{escape(note)}</span></article>' for label, value, note in items)
    tokens = (HERE / "tokens.css").read_text()
    components.html('''<html lang="es"><head><style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
''' + tokens + '''
*{box-sizing:border-box}html,body{margin:0;background:transparent;color:var(--text);font-family:Manrope,Arial,sans-serif}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--s2)}
article{display:flex;flex-direction:column;gap:6px;min-height:116px;padding:18px 20px;border:1px solid var(--line);border-radius:var(--radius);background:var(--panel)}
label{min-height:2.7em;font-size:var(--fs-label);font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);line-height:1.35}
strong{font-size:var(--fs-metric);font-weight:700;letter-spacing:-.03em;line-height:1.15;font-variant-numeric:tabular-nums}
span{color:var(--muted);font-size:12px;line-height:1.4}
@media(max-width:720px){.grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
 article{min-height:0;padding:14px 16px}strong{font-size:28px}}
</style></head><body><div class="grid">''' + cards + '''</div><script>
// The iframe has a fixed height; on narrow widths the grid is two rows, so size the frame to its content.
const grid=document.querySelector('.grid');
function fit(){try{const f=window.frameElement;if(f)f.style.height=Math.ceil(grid.getBoundingClientRect().height)+2+'px'}catch(e){}}
new ResizeObserver(fit).observe(grid);fit();
if(!matchMedia('(prefers-reduced-motion: reduce)').matches){
 document.querySelectorAll('strong').forEach(el=>{const exact=el.dataset.value;
 if(!/^[0-9]+$/.test(exact))return;
 const end=Number(exact),start=performance.now();
 function frame(now){const t=Math.min((now-start)/620,1);el.textContent=t===1?exact:String(Math.floor(end*(1-Math.pow(1-t,3))));if(t<1)requestAnimationFrame(frame)}
 requestAnimationFrame(frame);
 });
}
</script></body></html>''', height=118)


def chart_style(chart):
    return chart.properties(padding={"left": 16, "right": 24, "top": 16, "bottom": 16}, autosize={"type": "fit", "contains": "padding"}).configure(
        background=PANEL).configure_view(strokeOpacity=0).configure_title(
        color="#c6d6ec", font="Manrope", fontSize=12, fontWeight=600, anchor="start", offset=12).configure_axis(
        labelColor="#b6c8e0", titleColor="#b6c8e0", gridColor="#20324d", domainColor="#30445f",
        tickColor="#30445f", labelFont="Manrope", titleFont="Manrope", labelFontSize=12,
        titleFontSize=12, titlePadding=16).configure_legend(labelColor="#c6d6ec", titleColor="#c6d6ec",
        labelFont="Manrope", labelFontSize=12)
