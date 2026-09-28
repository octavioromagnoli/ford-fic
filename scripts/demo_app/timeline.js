// The fleet calendar: one column per replay week, its bar the alerts that arrived that week. Choosing a week
// (a column, the arrows or "Próxima con alertas") sends its Monday to Python as the `week` trigger.
//
// Streamlit calls this function again on every rerun without cleaning up, so it renders idempotently: the
// columns are built once per set of weeks and only the selection is updated afterwards. The styles live in
// theme.css (the component is not isolated) so the calendar reads as the same material as the rest of the app.

const ICONS = {
  prev: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15.41 7.41 14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg>',
  next: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 6 8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/></svg>',
  skip: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/></svg>',
};

const count = (n, one, many) => `${n} ${n === 1 ? one : many}`;

function summary(w) {
  const parts = [];
  if (w.new) parts.push(count(w.new, 'alerta nueva', 'alertas nuevas'));
  if (w.esc) parts.push(count(w.esc, 'escalamiento', 'escalamientos'));
  return parts.length ? parts.join(' · ') : 'sin alertas';
}

function build(data) {
  const weeks = data.weeks;
  const max = Math.max(1, ...weeks.map(w => w.new + w.esc));
  const root = document.createElement('div');
  // A replay of more than ~10 months packs its columns tighter (theme.css `.tl-dense`): the count per week moves to
  // the tooltip and the accessible label, and the bars keep their height.
  root.className = weeks.length > 40 ? 'tl tl-dense' : 'tl';
  root.style.setProperty('--n', weeks.length);
  root.innerHTML = `
    <div class="tl-head">
      <div class="tl-when">
        <span class="tl-eyebrow"></span>
        <div class="tl-titlerow">
          <strong class="tl-title" aria-live="polite"><span class="tl-long"></span><span class="tl-short"></span></strong>
          <span class="tl-chips"></span>
        </div>
      </div>
      <div class="tl-ctrl">
        <button type="button" class="tl-btn tl-prev" aria-label="Semana anterior" title="Semana anterior">${ICONS.prev}</button>
        <button type="button" class="tl-btn tl-next" aria-label="Semana siguiente" title="Semana siguiente">${ICONS.next}</button>
        <button type="button" class="tl-btn tl-skip">${ICONS.skip}<span>Próxima con alertas</span></button>
      </div>
    </div>
    <div class="tl-plot">
      <div class="tl-cols" role="group" aria-label="Alertas por semana del replay"></div>
      <div class="tl-axis" aria-hidden="true"></div>
      <div class="tl-tip" role="tooltip" hidden></div>
    </div>
    <div class="tl-foot">
      <span class="tl-key"><i class="tl-key-new"></i>Alerta nueva</span>
      <span class="tl-key"><i class="tl-key-esc"></i>Escalamiento</span>
      <span class="tl-hint"><span class="tl-hint-fine">Elegí una semana en el gráfico para abrirla</span><span class="tl-hint-touch">Tocá una semana para abrirla</span></span>
    </div>`;

  const cols = root.querySelector('.tl-cols');
  const axis = root.querySelector('.tl-axis');
  weeks.forEach((w, i) => {
    const total = w.new + w.esc;
    const col = document.createElement('button');
    col.type = 'button';
    col.className = 'tl-col' + (total ? ' has-events' : '');
    col.dataset.i = i;
    col.setAttribute('aria-label', `${w.range}: ${summary(w)}`);
    col.title = `${w.range}: ${summary(w)}`;
    const stack = total
      ? `<span class="tl-n">${total}</span><span class="tl-stack" style="--h:${total / max}">` +
        (w.esc ? `<i class="tl-esc" style="flex-grow:${w.esc}"></i>` : '') +
        (w.new ? `<i class="tl-new" style="flex-grow:${w.new}"></i>` : '') + '</span>'
      : '<span class="tl-empty"></span>';
    col.innerHTML = `<span class="tl-bar">${stack}</span><span class="tl-seg"></span>`;
    col.addEventListener('click', () => go(root, i));
    col.addEventListener('pointerenter', () => tip(root, i));
    col.addEventListener('pointerleave', () => tip(root, -1));
    col.addEventListener('focus', () => tip(root, i));
    col.addEventListener('blur', () => tip(root, -1));
    cols.appendChild(col);
    if (w.month) {
      const label = document.createElement('span');
      label.style.gridColumn = `${i + 1} / span 4`;
      label.innerHTML = w.month + (w.year ? ` <span class="tl-year">${w.year}</span>` : '');
      if (w.year) label.classList.add('has-year');
      axis.appendChild(label);
    }
  });
  // The months that fit depend on the width (82 weeks leave ~4 px per week on a phone): thin them out again
  // whenever the axis changes size, and once the web font has loaded.
  new ResizeObserver(() => fitAxis(axis)).observe(axis);
  document.fonts?.ready.then(() => fitAxis(axis));
  // One tab stop for the whole chart; the arrows move between weeks (roving tabindex).
  cols.addEventListener('keydown', (e) => {
    const i = Number(document.activeElement?.dataset?.i ?? root._i);
    const to = { ArrowLeft: i - 1, ArrowRight: i + 1, Home: 0, End: weeks.length - 1 }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    const target = cols.children[Math.max(0, Math.min(weeks.length - 1, to))];
    target.focus();
  });
  root.querySelector('.tl-prev').addEventListener('click', () => go(root, root._i - 1));
  root.querySelector('.tl-next').addEventListener('click', () => go(root, root._i + 1));
  root.querySelector('.tl-skip').addEventListener('click', () => go(root, root._nextEvent));
  return root;
}

// Hides the month labels that would run into the one before them, left to right. A label with the year (the
// first month and every January) always stays: the months before it give way. Nothing crosses the right edge.
function fitAxis(axis) {
  const labels = [...axis.children];
  labels.forEach(l => l.classList.remove('is-hidden'));
  const edge = axis.getBoundingClientRect().right;
  const boxes = labels.map(l => {
    const range = document.createRange();
    range.selectNodeContents(l);
    return range.getBoundingClientRect();
  });
  const gap = 8;
  const kept = [];
  const fits = i => !kept.length || boxes[i].left >= boxes[kept[kept.length - 1]].right + gap;
  labels.forEach((l, i) => {
    if (l.classList.contains('has-year')) {
      while (!fits(i) && !labels[kept[kept.length - 1]].classList.contains('has-year')) {
        labels[kept.pop()].classList.add('is-hidden');
      }
      kept.push(i);
    } else if (fits(i) && boxes[i].right <= edge + 1) {
      kept.push(i);
    } else {
      l.classList.add('is-hidden');
    }
  });
}

function tip(root, i) {
  const box = root.querySelector('.tl-tip');
  if (i < 0) { box.hidden = true; return; }
  const w = root._weeks[i];
  const col = root.querySelector('.tl-cols').children[i];
  box.innerHTML = `<strong>${w.range}</strong><span>${summary(w)}</span>`;
  box.hidden = false;
  const plot = root.querySelector('.tl-plot');
  const center = col.offsetLeft + col.offsetWidth / 2;
  const half = box.offsetWidth / 2;
  box.style.left = `${Math.max(half, Math.min(plot.clientWidth - half, center))}px`;
}

function select(root, i) {
  const weeks = root._weeks;
  const w = weeks[i];
  root._i = i;
  [...root.querySelector('.tl-cols').children].forEach((col, j) => {
    col.classList.toggle('is-current', j === i);
    col.classList.toggle('is-past', j < i);
    col.tabIndex = j === i ? 0 : -1;
    if (j === i) col.setAttribute('aria-current', 'date');
    else col.removeAttribute('aria-current');
  });
  root.querySelector('.tl-eyebrow').textContent = `Semana ${i + 1} de ${weeks.length} del replay`;
  root.querySelector('.tl-long').textContent = w.title;
  root.querySelector('.tl-short').textContent = w.range;
  const chips = [];
  if (w.new) chips.push(`<span class="tl-chip tl-chip-new">${count(w.new, 'alerta nueva', 'alertas nuevas')}</span>`);
  if (w.esc) chips.push(`<span class="tl-chip tl-chip-esc">${count(w.esc, 'escalamiento', 'escalamientos')}</span>`);
  if (!chips.length) chips.push('<span class="tl-chip">Sin alertas</span>');
  root.querySelector('.tl-chips').innerHTML = chips.join('');
  root._nextEvent = weeks.findIndex((x, j) => j > i && x.new + x.esc > 0);
  root.querySelector('.tl-prev').disabled = i === 0;
  root.querySelector('.tl-next').disabled = i === weeks.length - 1;
  root.querySelector('.tl-skip').disabled = root._nextEvent < 0;
}

function go(root, i) {
  if (i < 0 || i >= root._weeks.length || i === root._i) return;
  select(root, i);            // answer at once; the rerun confirms it
  root._send('week', root._weeks[i].iso);
}

export default function ({ parentElement, data, setTriggerValue }) {
  const signature = data.weeks.map(w => `${w.iso}:${w.new}:${w.esc}`).join('|');
  let root = parentElement.querySelector('.tl');
  if (!root || root.dataset.signature !== signature) {
    root?.remove();
    root = build(data);
    root.dataset.signature = signature;
    root._weeks = data.weeks;
    parentElement.appendChild(root);
  }
  root._send = setTriggerValue;
  const i = data.weeks.findIndex(w => w.iso === data.current);
  if (i >= 0 && i !== root._i) select(root, i);
}
