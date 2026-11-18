"""A one-page UI: ask a question, see the answer, click a citation to see the exact excerpt."""

from __future__ import annotations

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .answer import ask

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Docchat</title><style>
:root{--bg:#f7f7f8;--card:#fff;--ink:#16181d;--muted:#5f6670;--line:#e2e4e8;--accent:#2f5bea;--warn:#a15c00}
@media (prefers-color-scheme:dark){:root{--bg:#101216;--card:#191c22;--ink:#e8ebf0;--muted:#9aa2ad;--line:#2b3038;--accent:#86a2ff;--warn:#f0a543}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:760px;margin:auto;padding:28px 16px}h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);font-size:14px;margin-bottom:18px}
form{display:flex;gap:8px}input{flex:1;min-width:0;font:inherit;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
button{font:inherit;font-weight:600;padding:10px 16px;border-radius:10px;border:0;background:var(--accent);color:#fff;cursor:pointer}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:16px}
.cite{color:var(--accent);cursor:pointer;font-weight:600}.src{font-size:14px;color:var(--muted);border-left:3px solid var(--line);padding:4px 10px;margin:8px 0}
.src.on{border-color:var(--accent);color:var(--ink)}.warn{color:var(--warn);font-size:14px}
</style></head><body><main><h1>Ask the document</h1><div class="sub">Answers cite the page they came from. If it isn't in the document, it says so.</div>
<form id="f"><input id="q" placeholder="e.g. How much PTO do I get?" aria-label="Question" autofocus><button>Ask</button></form><div id="out"></div></main>
<script>
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
document.getElementById("f").onsubmit=async e=>{e.preventDefault();const q=document.getElementById("q").value.trim();if(!q)return;
const out=document.getElementById("out");out.innerHTML='<div class="card">Thinking…</div>';
const r=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json","X-Docchat":"1"},body:JSON.stringify({q})});const a=await r.json();
const text=esc(a.text).replace(/\\[(\\d+)\\]/g,'<span class="cite" data-n="$1">[$1]</span>');
out.innerHTML=`<div class="card"><div>${text}</div>${a.issues.length?`<div class="warn">⚠ ${esc(a.issues.join("; "))}</div>`:""}
${a.retrieved.map(s=>`<div class="src" id="s${s.n}"><b>[${s.n}] page ${s.page}</b>${s.heading?" · "+esc(s.heading):""}<br>${esc(s.text)}</div>`).join("")}</div>`;
out.querySelectorAll(".cite").forEach(c=>c.onclick=()=>{out.querySelectorAll(".src").forEach(s=>s.classList.remove("on"));const s=document.getElementById("s"+c.dataset.n);if(s){s.classList.add("on");s.scrollIntoView({behavior:"smooth",block:"center"})}});};
</script></body></html>"""


def make_handler(index, llm):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, kind="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            self._send(404, b"{}")

        def do_POST(self):
            if self.path != "/api/ask" or self.headers.get("X-Docchat") != "1":
                return self._send(403, b"{}")
            try:
                q = str(json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))["q"])[:500]
            except (ValueError, KeyError):
                return self._send(400, b"{}")
            self._send(200, json.dumps(asdict(ask(index, q, llm))).encode())
    return H


def serve(index, llm, port=8796):
    print(f"docchat on http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(index, llm)).serve_forever()
