"""Builds docs/RESEARCH_STORY.html (a local page; never published) from docs/story/tests.json (every test, extracted
from the records) and docs/story/story.json (the narrative). Rerun after either changes:

    python docs/story/build.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "RESEARCH_STORY.html"

CSS = r"""
:root { --paper:#f2f4f6; --card:#fff; --ink:#15212c; --ink-2:#4a5866; --muted:#7d8a96; --hair:#d5dbe2; --accent:#0f6e8c;
  --accent-soft:#e2f0f4; --good:#0a7d0a; --good-soft:#e5f4e5; --bad:#b83232; --bad-soft:#f9e6e6; --shine:#b7791f; --shine-soft:#fbf3e4;
  --run:#2a5fb0; --run-soft:#e4ecf8; --side:#e8ecf0;
  --f-display:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif; --f-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Consolas,monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --paper:#0f151b; --card:#16202a; --ink:#e8eef3; --ink-2:#b5c1cc;
  --muted:#8796a3; --hair:#2a3743; --accent:#4fb3d1; --accent-soft:#15303a; --good:#3cc23c; --good-soft:#12291a; --bad:#e66767; --bad-soft:#321a1a;
  --shine:#e0a548; --shine-soft:#2a2213; --run:#6f9ee8; --run-soft:#172438; --side:#121a22; color-scheme:dark; } }
:root[data-theme="dark"] { --paper:#0f151b; --card:#16202a; --ink:#e8eef3; --ink-2:#b5c1cc; --muted:#8796a3; --hair:#2a3743; --accent:#4fb3d1;
  --accent-soft:#15303a; --good:#3cc23c; --good-soft:#12291a; --bad:#e66767; --bad-soft:#321a1a; --shine:#e0a548; --shine-soft:#2a2213;
  --run:#6f9ee8; --run-soft:#172438; --side:#121a22; color-scheme:dark; }
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font:15px/1.6 var(--f-body); }
.app { display:grid; grid-template-columns:300px minmax(0,1fr); min-height:100vh; }
nav { background:var(--side); border-right:1px solid var(--hair); position:sticky; top:0; height:100vh; overflow-y:auto; padding:18px 14px 40px; }
nav .brand { font:700 20px/1.1 var(--f-display); margin:0 0 4px; }
nav .sub { font-size:12px; color:var(--muted); margin-bottom:12px; }
nav input { width:100%; padding:7px 10px; border:1px solid var(--hair); border-radius:8px; background:var(--card); color:var(--ink); font:13px var(--f-body); margin-bottom:10px; }
nav a { display:flex; gap:8px; align-items:baseline; color:var(--ink-2); text-decoration:none; padding:4px 8px; border-radius:6px; font-size:13px; line-height:1.35; }
nav a:hover { background:var(--card); color:var(--ink); }
nav a.on { background:var(--accent-soft); color:var(--ink); font-weight:600; }
nav a .id { font:600 11px var(--f-mono); color:var(--muted); min-width:58px; }
nav .phase { font:600 11px var(--f-mono); letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin:16px 8px 4px; }
nav .home { font-weight:600; color:var(--ink); }
.dot { width:8px; height:8px; border-radius:50%; flex:none; align-self:center; background:var(--muted); }
.v-adopted { background:var(--good); } .v-rejected { background:var(--bad); } .v-bug { background:var(--shine); } .v-progress { background:var(--run); }
main { padding:36px 44px 90px; max-width:980px; }
h1 { font:700 42px/1.05 var(--f-display); margin:0 0 10px; text-wrap:balance; }
h2 { font:700 26px/1.15 var(--f-display); margin:34px 0 8px; text-wrap:balance; }
h3 { font:600 17px/1.25 var(--f-display); margin:20px 0 6px; }
p { margin:0 0 12px; max-width:72ch; color:var(--ink-2); }
p b, li b { color:var(--ink); }
.lede { font-size:18px; color:var(--ink); }
.eyebrow { font:600 12px var(--f-mono); letter-spacing:.08em; color:var(--muted); text-transform:uppercase; margin-bottom:8px; }
.panel { background:var(--card); border:1px solid var(--hair); border-radius:12px; padding:18px 20px; margin:14px 0; }
.kv { display:grid; grid-template-columns:120px minmax(0,1fr); gap:6px 16px; font-size:14px; }
.kv dt { color:var(--muted); font-weight:600; } .kv dd { margin:0; color:var(--ink); }
ul { margin:6px 0 12px; padding-left:20px; color:var(--ink-2); max-width:76ch; } li { margin:4px 0; }
.badge { display:inline-block; font:600 12px var(--f-body); padding:3px 10px; border-radius:999px; background:var(--accent-soft); color:var(--accent); }
.b-adopted { background:var(--good-soft); color:var(--good); } .b-rejected { background:var(--bad-soft); color:var(--bad); }
.b-bug { background:var(--shine-soft); color:var(--shine); } .b-progress { background:var(--run-soft); color:var(--run); }
.results li { font-family:var(--f-mono); font-size:13px; color:var(--ink); }
.src { font:12px var(--f-mono); color:var(--muted); }
.pager { display:flex; justify-content:space-between; gap:12px; margin-top:26px; }
.pager a { color:var(--accent); text-decoration:none; font-weight:600; font-size:14px; }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 4px; }
.chips a { font:600 12px var(--f-mono); padding:3px 8px; border-radius:6px; background:var(--accent-soft); color:var(--accent); text-decoration:none; }
.tl { border-left:2px solid var(--hair); margin:10px 0 10px 6px; padding-left:16px; }
.lessons li { margin:8px 0; }
.toggle { position:fixed; top:10px; right:12px; z-index:5; font:600 12px var(--f-body); border:1px solid var(--hair); background:var(--card); color:var(--ink); border-radius:8px; padding:5px 10px; cursor:pointer; }
.menu { display:none; }
@media (max-width: 860px) {
  .app { grid-template-columns:1fr; } nav { position:fixed; inset:0 auto 0 0; width:86%; max-width:320px; z-index:6; transform:translateX(-100%); transition:transform .2s; }
  nav.open { transform:none; } main { padding:56px 16px 80px; } h1 { font-size:32px; }
  .menu { display:block; position:fixed; top:10px; left:12px; z-index:7; font:600 12px var(--f-body); border:1px solid var(--hair); background:var(--card); color:var(--ink); border-radius:8px; padding:5px 10px; }
  .kv { grid-template-columns:1fr; }
}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const byId = Object.fromEntries(D.tests.map((t, i) => [t.id, i]));
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const vclass = v => { v = (v||'').toLowerCase(); return v.startsWith('adopt') ? 'adopted' : (v.startsWith('reject') || v.startsWith('not adopt')) ? 'rejected'
  : v.includes('bug') || v.includes('finding') || v.includes('leak') ? 'bug' : v.includes('progress') || v.includes('running') || v.includes('queued') ? 'progress' : ''; };
const nav = document.getElementById('nav'), main = document.getElementById('main');
function buildNav(filter) {
  const f = (filter||'').toLowerCase(); let h = '<div class="brand">The research story</div><div class="sub">Text Research Model · Adam Barnett</div>' +
    '<div class="sub"><span class="dot v-adopted" style="display:inline-block"></span> adopted &nbsp;<span class="dot v-rejected" style="display:inline-block"></span> rejected &nbsp;<span class="dot v-bug" style="display:inline-block"></span> bug / finding<br><span class="dot v-progress" style="display:inline-block"></span> in progress &nbsp;<span class="dot" style="display:inline-block"></span> kept, mixed or inconclusive</div>' +
    '<input id="q" placeholder="Filter tests…" value="' + esc(filter||'') + '">' + '<a href="#" class="home' + (location.hash ? '' : ' on') + '">The story of the journey</a>' +
    '<a href="#lessons">Lessons learned</a>';
  let ph = null;
  D.tests.forEach(t => {
    const hay = (t.id + ' ' + t.title + ' ' + t.phase + ' ' + (t.lesson||'')).toLowerCase();
    if (f && !hay.includes(f)) return;
    if (t.phase !== ph) { ph = t.phase; h += '<div class="phase">' + esc(ph) + '</div>'; }
    h += '<a href="#test/' + encodeURIComponent(t.id) + '" class="' + (location.hash === '#test/' + encodeURIComponent(t.id) ? 'on' : '') + '"><span class="dot v-' + vclass(t.verdict) + '"></span><span class="id">' + esc(t.id) + '</span><span>' + esc(t.title) + '</span></a>';
  });
  nav.innerHTML = h;
  const q = document.getElementById('q'); q.oninput = () => { buildNav(q.value); const n = document.getElementById('q'); n.focus(); n.setSelectionRange(n.value.length, n.value.length); };
}
function chips(ids) { return '<div class="chips">' + ids.filter(i => i in byId).map(i => '<a href="#test/' + encodeURIComponent(i) + '">' + esc(i) + '</a>').join('') + '</div>'; }
function story() {
  const S = D.story; let h = '<div class="eyebrow">' + esc(S.eyebrow) + '</div><h1>' + esc(S.title) + '</h1><p class="lede">' + S.lede + '</p>';
  S.chapters.forEach(c => { h += '<h2 id="' + esc(c.id) + '">' + esc(c.title) + '</h2>' + (c.dates ? '<div class="eyebrow">' + esc(c.dates) + '</div>' : '') +
    c.paras.map(p => '<p>' + p + '</p>').join('') + (c.tests && c.tests.length ? '<div class="src">Tests in this chapter:</div>' + chips(c.tests) : ''); });
  return h;
}
function lessons() {
  let h = '<div class="eyebrow">What the tests taught us</div><h1>Lessons learned</h1><p class="lede">' + D.story.lessons_intro + '</p>';
  D.story.lessons.forEach(g => { h += '<h2>' + esc(g.title) + '</h2><ul class="lessons">' + g.items.map(i => '<li>' + i.text + (i.tests ? ' ' + chips(i.tests) : '') + '</li>').join('') + '</ul>'; });
  return h;
}
function test(i) {
  const t = D.tests[i], v = vclass(t.verdict);
  let h = '<div class="eyebrow">' + esc(t.phase) + ' · ' + esc(t.date) + ' · ' + esc(t.id) + '</div><h1>' + esc(t.title) + '</h1>' +
    '<span class="badge b-' + v + '">' + esc(t.verdict) + '</span>' +
    '<div class="panel"><h3 style="margin-top:0">The question</h3><p>' + esc(t.question) + '</p></div>' +
    '<div class="panel"><dl class="kv"><dt>Settings</dt><dd>' + esc(t.settings) + '</dd><dt>Data</dt><dd>' + esc(t.data) + '</dd></dl></div>' +
    '<div class="panel"><h3 style="margin-top:0">Results</h3><ul class="results">' + (t.results||[]).map(r => '<li>' + esc(r) + '</li>').join('') + '</ul></div>' +
    '<div class="panel"><h3 style="margin-top:0">Lesson</h3><p>' + esc(t.lesson) + '</p><div class="src">Sources: ' + (t.sources||[]).map(esc).join(' · ') + '</div></div>';
  const prev = D.tests[i - 1], next = D.tests[i + 1];
  h += '<div class="pager"><span>' + (prev ? '<a href="#test/' + encodeURIComponent(prev.id) + '">← ' + esc(prev.id) + ' ' + esc(prev.title) + '</a>' : '') + '</span><span>' +
       (next ? '<a href="#test/' + encodeURIComponent(next.id) + '">' + esc(next.id) + ' ' + esc(next.title) + ' →</a>' : '') + '</span></div>';
  return h;
}
function route() {
  const hsh = decodeURIComponent(location.hash.slice(1));
  if (hsh.startsWith('test/') && hsh.slice(5) in byId) main.innerHTML = test(byId[hsh.slice(5)]);
  else if (hsh === 'lessons') main.innerHTML = lessons();
  else { main.innerHTML = story(); if (hsh && document.getElementById(hsh)) document.getElementById(hsh).scrollIntoView(); }
  if (!hsh.startsWith('test/') && hsh !== 'lessons' && !hsh) window.scrollTo(0, 0);
  if (hsh.startsWith('test/') || hsh === 'lessons') window.scrollTo(0, 0);
  buildNav(document.getElementById('q') ? document.getElementById('q').value : '');
  nav.classList.remove('open');
}
window.addEventListener('hashchange', route);
document.getElementById('menu').onclick = () => nav.classList.toggle('open');
const tg = document.getElementById('theme');
tg.onclick = () => { const r = document.documentElement; const dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  r.dataset.theme = dark ? 'light' : 'dark'; try { localStorage.setItem('story-theme', r.dataset.theme); } catch (e) {} };
try { const s = localStorage.getItem('story-theme'); if (s) document.documentElement.dataset.theme = s; } catch (e) {}
route();
"""


def main() -> int:
    tests = json.loads((HERE / "tests.json").read_text(encoding="utf-8"))
    story = json.loads((HERE / "story.json").read_text(encoding="utf-8"))
    data = json.dumps({"tests": tests, "story": story}, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research Story</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
</head>
<body>
<button class="menu" id="menu">Tests</button>
<button class="toggle" id="theme">Light / dark</button>
<div class="app"><nav id="nav"></nav><main id="main"></main></div>
<script type="application/json" id="data">{data}</script>
<script>{JS}</script>
</body>
</html>
"""
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({len(page) // 1024} KB, {len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
