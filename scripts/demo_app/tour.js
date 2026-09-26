// "Cómo usar": a guided tour that dims the app, rings one part at a time and explains it in a card.
//
// The steps come from Python (guide.py) for the page being shown; each names CSS selectors for its target and
// the first one visible wins, so the same step points at the sidebar on a notebook and at the dock on a phone.
// A step whose target is not on the page (a week without alerts has no card) is skipped. The tour lives on
// `window`, outside React: Streamlit calls this function again on every rerun, and that only refreshes the steps.
// Everything runs in the browser, with no rerun. The styles live in theme.css.

const VERSION = 1;
const SEEN = 'fdpf-tour-seen';

const ICONS = {
  help: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M11 18h2v-2h-2v2zm1-16C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-14c-2.21 0-4 1.79-4 4h2c0-1.1.9-2 2-2s2 .9 2 2c0 2-3 1.75-3 5h2c0-2.25 3-2.5 3-5 0-2.21-1.79-4-4-4z"/></svg>',
  close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>',
};

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

// Seen, or reachable by scrolling. The collapsed sidebar keeps its size off-screen to the left, so a target also has
// to fall within the width of the window.
function visible(el) {
  const r = el.getBoundingClientRect();
  if (r.width < 2 || r.height < 2 || r.right <= 0 || r.left >= window.innerWidth) return false;
  const style = getComputedStyle(el);
  return style.visibility !== 'hidden' && style.display !== 'none';
}

function createTour() {
  const tour = { steps: [], list: [], i: 0, active: false, version: VERSION };
  let layer, spot, card, frame = 0, origin = null;

  function target(step) {
    for (const selector of step.targets || []) {
      for (const el of document.querySelectorAll(selector)) if (visible(el)) return el;
    }
    return null;
  }

  function mount() {
    layer = document.createElement('div');
    layer.className = 'tour-layer';
    layer.innerHTML = `
      <div class="tour-block"></div>
      <div class="tour-spot"></div>
      <div class="tour-card" role="dialog" aria-modal="true" aria-labelledby="tour-title" aria-describedby="tour-body">
        <div class="tour-top">
          <span class="tour-count"></span>
          <button type="button" class="tour-close" aria-label="Cerrar el recorrido">${ICONS.close}</button>
        </div>
        <h2 id="tour-title" class="tour-title"></h2>
        <div id="tour-body" class="tour-body"></div>
        <div class="tour-progress"><i></i></div>
        <div class="tour-actions">
          <button type="button" class="tour-prev">Anterior</button>
          <button type="button" class="tour-next">Siguiente</button>
        </div>
      </div>`;
    document.body.appendChild(layer);
    spot = layer.querySelector('.tour-spot');
    card = layer.querySelector('.tour-card');
    layer.querySelector('.tour-close').addEventListener('click', () => tour.end());
    layer.querySelector('.tour-prev').addEventListener('click', () => tour.go(tour.i - 1));
    layer.querySelector('.tour-next').addEventListener('click', () => tour.next());
  }

  function onKey(e) {
    if (!tour.active) return;
    if (e.key === 'Escape') { e.preventDefault(); tour.end(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); tour.next(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); tour.go(tour.i - 1); }
    else if (e.key === 'Tab') {
      // Keep focus inside the card while the tour is open.
      const buttons = [...card.querySelectorAll('button:not([disabled])')];
      const at = buttons.indexOf(document.activeElement);
      e.preventDefault();
      buttons[(at + (e.shiftKey ? -1 : 1) + buttons.length) % buttons.length].focus();
    }
  }

  // Where the ring and the card go. It runs every frame while the tour is open: the page scrolls, Streamlit
  // re-renders and the window resizes, and re-measuring is cheaper than tracking each of those.
  function place() {
    const step = tour.list[tour.i];
    const el = step && target(step);
    const vw = document.documentElement.clientWidth, vh = window.innerHeight, m = 12, pad = 8;
    let r = null;
    if (el) {
      const b = el.getBoundingClientRect();
      const top = Math.max(b.top - pad, 6), left = Math.max(b.left - pad, 6);
      const bottom = Math.min(b.bottom + pad, vh - 6), right = Math.min(b.right + pad, vw - 6);
      if (bottom - top > 4 && right - left > 4) r = { top, left, bottom, right, width: right - left, height: bottom - top };
    }
    layer.classList.toggle('is-center', !r);
    if (r) Object.assign(spot.style, { left: `${r.left}px`, top: `${r.top}px`, width: `${r.width}px`, height: `${r.height}px` });
    else Object.assign(spot.style, { left: `${vw / 2}px`, top: `${vh / 2}px`, width: '0px', height: '0px' });

    const sheet = vw < 640;
    card.classList.toggle('is-sheet', sheet);
    const cw = card.offsetWidth, ch = card.offsetHeight, gap = 16;
    let x, y;
    if (sheet) {
      // On a phone the card is a sheet on the side of the screen with more room.
      x = m;
      y = r && r.top > vh - r.bottom ? m : vh - ch - m;
    } else if (!r) {
      x = (vw - cw) / 2; y = (vh - ch) / 2;
    } else if (r.width < vw * 0.4 && vw - r.right >= cw + gap + m) {
      x = r.right + gap; y = clamp(r.top, m, vh - ch - m);
    } else if (vh - r.bottom >= ch + gap + m) {
      y = r.bottom + gap; x = clamp(r.left, m, vw - cw - m);
    } else if (r.top >= ch + gap + m) {
      y = r.top - gap - ch; x = clamp(r.left, m, vw - cw - m);
    } else {
      y = vh - ch - m; x = clamp(r.right - cw - m, m, vw - cw - m);
    }
    card.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  }

  function loop() {
    place();
    frame = requestAnimationFrame(loop);
  }

  tour.start = (from) => {
    origin = from;
    tour.list = tour.steps.filter(step => !step.targets || target(step));
    if (!tour.list.length) return;
    if (!layer) mount();
    tour.active = true;
    layer.classList.add('is-instant');
    tour.go(0);
    place();
    layer.classList.add('is-on');
    requestAnimationFrame(() => layer.classList.remove('is-instant'));
    document.addEventListener('keydown', onKey, true);
    cancelAnimationFrame(frame);
    loop();
  };

  tour.go = (i) => {
    if (i < 0 || i >= tour.list.length) return;
    tour.i = i;
    const step = tour.list[i], last = i === tour.list.length - 1;
    layer.querySelector('.tour-count').textContent = `${i + 1} de ${tour.list.length}`;
    layer.querySelector('.tour-title').textContent = step.title;
    layer.querySelector('.tour-body').innerHTML = step.body;   // trusted: written in guide.py
    layer.querySelector('.tour-progress i').style.width = `${(100 * (i + 1)) / tour.list.length}%`;
    layer.querySelector('.tour-prev').disabled = i === 0;
    layer.querySelector('.tour-next').textContent = last ? 'Listo' : 'Siguiente';
    const el = target(step);
    if (el) {
      const b = el.getBoundingClientRect();
      if (b.top < 8 || b.bottom > window.innerHeight - 8) {
        el.scrollIntoView({ block: b.height > window.innerHeight * 0.6 ? 'start' : 'center',
                            behavior: reducedMotion() ? 'auto' : 'smooth' });
      }
    }
    card.classList.toggle('is-welcome', !step.targets);   // the welcome card glows like the button that opened it
    card.classList.remove('is-in');
    void card.offsetWidth;   // restart the entrance
    card.classList.add('is-in');
    layer.querySelector('.tour-next').focus({ preventScroll: true });
  };

  tour.next = () => (tour.i === tour.list.length - 1 ? tour.end() : tour.go(tour.i + 1));

  tour.end = () => {
    if (!tour.active) return;
    tour.active = false;
    cancelAnimationFrame(frame);
    layer.classList.remove('is-on');
    document.removeEventListener('keydown', onKey, true);
    try { localStorage.setItem(SEEN, '1'); } catch (e) { /* storage blocked: the hint just stays */ }
    document.querySelectorAll('.tour-btn.is-new').forEach(b => b.classList.remove('is-new'));
    if (origin?.isConnected) origin.focus({ preventScroll: true });
  };

  return tour;
}

export default function ({ parentElement, data }) {
  if (window.__fdpfTour?.version !== VERSION) window.__fdpfTour = createTour();
  window.__fdpfTour.steps = data.steps;
  let button = parentElement.querySelector('.tour-btn');
  if (!button) {
    button = document.createElement('button');
    button.type = 'button';
    button.className = 'tour-btn';
    button.innerHTML = `${ICONS.help}<span>${data.label}</span>`;
    button.addEventListener('click', () => window.__fdpfTour.start(button));
    parentElement.appendChild(button);
  }
  let seen = true;
  try { seen = Boolean(localStorage.getItem(SEEN)); } catch (e) { /* storage blocked: no hint */ }
  button.classList.toggle('is-new', !seen);
  button.title = seen ? '' : 'Primera vez: recorré la demo paso a paso';
}
