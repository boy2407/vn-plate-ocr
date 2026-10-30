"""Small local web page for labelling the rows in ``disagreements.csv`` fast.

One crop at a time with both readings; keys pick an answer and save it straight
back to the CSV, so ``consensus.py`` picks the decisions up on its next run.
Rows where Codex and the VLM read different texts come first.

    python3 -m custom_alpr.label_web            # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from custom_alpr.consensus import COLLECTED, DISAGREEMENT_FIELDS, read_rows, write_rows
from custom_alpr.plate_tracks import normalized_text, valid_plate

VERDICTS = {"label", "unreadable", "partial", "not_plate"}


def label_rows(collected: Path) -> list[dict]:
    """Disagreement rows with the line counts each labeller gave, different-text rows first."""
    queue = {row["crop_id"]: row for row in read_rows(collected / "queue.csv")}
    vlm = {row["crop_id"]: row for row in read_rows(collected / "vlm-labels.csv")}
    rows = []
    for row in read_rows(collected / "disagreements.csv"):
        crop = queue.get(row["crop_id"], {})
        rows.append(
            dict(
                row,
                pad_path=crop.get("image_path", "").replace(".png", "_pad.png"),
                codex_lines=crop.get("lines", ""),
                vlm_lines=vlm.get(row["crop_id"], {}).get("lines", ""),
            )
        )
    both_read = lambda row: row["codex_verdict"] == "label" and row["vlm_verdict"] == "label"  # noqa: E731
    return sorted(rows, key=lambda row: not both_read(row))


def check_answer(answer: dict) -> tuple[dict, str]:
    """Normalized ``verdict, plate_text, lines, note`` and an error message ('' when valid)."""
    verdict = answer.get("verdict", "").strip()
    text = normalized_text(answer.get("plate_text", "")) if verdict == "label" else ""
    lines = str(answer.get("lines", "")).strip()
    clean = dict(verdict=verdict, plate_text=text, lines=lines, note=answer.get("note", "").strip())
    if verdict and verdict not in VERDICTS:
        return clean, f"unknown verdict {verdict!r}"
    if verdict == "label" and not valid_plate(text):
        return clean, f"{text!r} is not a VN plate format"
    if verdict in ("label", "partial", "unreadable") and lines not in ("1", "2", ""):
        return clean, "lines must be 1 or 2"
    if verdict == "label" and not lines:
        return clean, "lines must be 1 or 2"
    return clean, ""


def save_answer(collected: Path, crop_id: str, clean: dict):
    path = collected / "disagreements.csv"
    rows = read_rows(path)
    for row in rows:
        if row["crop_id"] == crop_id:
            row.update(clean)
            break
    else:
        raise KeyError(crop_id)
    temp = path.with_suffix(".csv.tmp")
    write_rows(temp, DISAGREEMENT_FIELDS, rows)
    os.replace(temp, path)


PAGE = """<!doctype html><html lang="vi"><meta charset="utf-8"><title>Plate labels</title>
<style>
body{font:15px system-ui,sans-serif;margin:0;background:#111;color:#eee}
main{max-width:900px;margin:auto;padding:16px}
#bar{display:flex;gap:12px;align-items:center;font-size:13px;color:#aaa}
#progress{flex:1;height:6px;background:#333}#fill{height:100%;background:#4c8}
.imgs{display:flex;gap:16px;align-items:center;margin:16px 0;min-height:260px}
#view{height:260px;image-rendering:pixelated;background:#000}
#pad{height:160px;image-rendering:pixelated;opacity:.85}
.reads{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.read{border:1px solid #444;border-radius:6px;padding:10px;cursor:pointer}
.read b{font:600 26px ui-monospace,monospace;letter-spacing:2px}
.read:hover{border-color:#4c8}
.form{display:flex;gap:8px;margin-top:14px;align-items:center;flex-wrap:wrap}
input{font:600 24px ui-monospace,monospace;background:#222;color:#fff;border:1px solid #555;padding:6px 10px;border-radius:4px}
#text{width:260px;text-transform:uppercase}#lines{width:50px}#note{font:14px system-ui;width:200px}
button{background:#333;color:#eee;border:1px solid #555;padding:8px 12px;border-radius:4px;cursor:pointer}
#status{min-height:20px;margin-top:10px}.err{color:#f66}.ok{color:#4c8}
#keys{font-size:12px;color:#888;margin-top:16px;line-height:1.7}kbd{background:#333;padding:1px 5px;border-radius:3px}
.done{color:#4c8;font-weight:600}
</style>
<main>
<div id="bar"><span id="count"></span><div id="progress"><div id="fill"></div></div><span id="crop"></span></div>
<div class="imgs"><img id="view"><img id="pad" title="crop có viền"></div>
<div class="reads">
 <div class="read" id="r1">1 · Codex <span id="cv"></span><br><b id="ct"></b> <span id="cl"></span></div>
 <div class="read" id="r2">2 · VLM <span id="vv"></span><br><b id="vt"></b> <span id="vl"></span></div>
</div>
<div class="form">
 <input id="text" placeholder="biển số" autocomplete="off">
 lines <input id="lines" maxlength="1">
 <input id="note" placeholder="ghi chú">
 <button id="save">Lưu (Enter)</button>
 <button id="skip">Bỏ, quá nhiễu (B)</button>
</div>
<div id="status"></div>
<div id="keys">
 <kbd>1</kbd>/<kbd>2</kbd> lấy cách đọc của Codex/VLM vào ô ·
 gõ biển rồi <kbd>Enter</kbd> = label ·
 <kbd>B</kbd> bỏ (quá nhiễu, không đưa vào train) ·
 <kbd>U</kbd> unreadable · <kbd>P</kbd> partial · <kbd>N</kbd> not_plate ·
 <kbd>L</kbd> đổi 1/2 dòng ·
 <kbd>←</kbd>/<kbd>→</kbd> lùi/tiếp · <kbd>S</kbd> nhảy tới dòng chưa làm ·
 <kbd>Esc</kbd> rời ô nhập
</div>
</main>
<script>
let rows=[],i=0;const $=id=>document.getElementById(id);
const done=r=>r.verdict.trim()!=="";
async function load(){rows=await (await fetch("api/rows")).json();i=Math.max(0,rows.findIndex(r=>!done(r)));show()}
function show(){
 const n=rows.filter(done).length;$("count").textContent=`${n}/${rows.length} đã làm · dòng ${i+1}`;
 $("fill").style.width=(100*n/rows.length)+"%";
 const r=rows[i];if(!r)return;
 $("crop").innerHTML=r.crop_id+(done(r)?` · <span class=done>${r.verdict} ${r.plate_text}</span>`:"");
 $("view").src="img/"+r.view_path;$("pad").src="img/"+r.pad_path;
 $("cv").textContent=r.codex_verdict;$("ct").textContent=r.codex_text;$("cl").textContent=r.codex_lines?`(${r.codex_lines} dòng)`:"";
 $("vv").textContent=r.vlm_verdict;$("vt").textContent=r.vlm_text;$("vl").textContent=r.vlm_lines?`(${r.vlm_lines} dòng)`:"";
 $("text").value=r.verdict==="label"?r.plate_text:"";
 $("lines").value=r.lines||r.codex_lines||r.vlm_lines||"";$("note").value=r.note||"";
 $("status").textContent="";
 const next=rows[i+1];if(next){new Image().src="img/"+next.view_path}
}
function pick(k){const r=rows[i];const t=k===1?r.codex_text:r.vlm_text;const l=k===1?r.codex_lines:r.vlm_lines;
 if(t){$("text").value=t;if(l)$("lines").value=l}$("text").focus()}
async function save(verdict,note){
 const r=rows[i];const body={crop_id:r.crop_id,verdict,plate_text:$("text").value,lines:$("lines").value,note:note??$("note").value};
 const res=await (await fetch("api/save",{method:"POST",body:JSON.stringify(body)})).json();
 if(res.error){$("status").className="err";$("status").textContent=res.error;return}
 Object.assign(r,res.row);if(i<rows.length-1)i++;show();
 $("status").className="ok";$("status").textContent=`đã lưu ${r.crop_id}: ${res.row.verdict} ${res.row.plate_text}`;
 if(document.activeElement!==$("text"))document.activeElement.blur();
}
$("r1").onclick=()=>pick(1);$("r2").onclick=()=>pick(2);$("save").onclick=()=>save("label");
// Too noisy to learn from: stored as unreadable, so it never reaches the training manifest.
const skip=()=>save("unreadable","skip");$("skip").onclick=skip;
document.addEventListener("keydown",e=>{
 const typing=["text","lines","note"].includes(document.activeElement.id);
 if(e.key==="Enter"){e.preventDefault();save("label");return}
 if(e.key==="Escape"){document.activeElement.blur();return}
 if(typing)return;
 const k=e.key.toLowerCase();
 if(k==="1"||k==="2"){e.preventDefault();pick(+k)}
 else if(k==="b")skip();
 else if(k==="u")save("unreadable");else if(k==="p")save("partial");else if(k==="n")save("not_plate");
 else if(k==="l")$("lines").value=$("lines").value==="1"?"2":"1";
 else if(e.key==="ArrowRight"&&i<rows.length-1){i++;show()}
 else if(e.key==="ArrowLeft"&&i>0){i--;show()}
 else if(k==="s"){const j=rows.findIndex(r=>!done(r));if(j>=0){i=j;show()}}
 else if(/^[a-z0-9]$/i.test(e.key)){$("text").focus()}
});
load();
</script></html>"""


def make_handler(collected: Path):
    root = collected.resolve()

    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, kind: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                self.send(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/rows":
                self.send(json.dumps(label_rows(collected)).encode(), "application/json")
            elif self.path.startswith("/img/"):
                image = Path(os.path.normpath(root / self.path[len("/img/") :]))
                if image.suffix != ".png" or root not in image.parents or not image.exists():
                    self.send(b"not found", "text/plain", 404)
                else:
                    self.send(image.read_bytes(), "image/png")
            else:
                self.send(b"not found", "text/plain", 404)

        def do_POST(self):  # noqa: N802
            if self.path != "/api/save":
                self.send(b"not found", "text/plain", 404)
                return
            answer = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            clean, error = check_answer(answer)
            if not error:
                try:
                    save_answer(collected, answer.get("crop_id", ""), clean)
                except KeyError:
                    error = "unknown crop_id"
            self.send(json.dumps(dict(row=clean, error=error)).encode(), "application/json")

        def log_message(self, *args):
            pass

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collected", type=Path, default=COLLECTED)
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = HTTPServer(("127.0.0.1", args.port), make_handler(args.collected))
    print(f"labelling {args.collected / 'disagreements.csv'} at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
