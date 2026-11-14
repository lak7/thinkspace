import { showToken, type Frame } from "../protocol";

const list = document.getElementById("candidates") as HTMLDivElement;
const caption = document.getElementById("candidates-caption") as HTMLDivElement;
const resid = document.getElementById("resid") as HTMLDivElement;

let key = "";

/**
 * Next-token probabilities for the current token (the model's distribution at
 * temperature 1) and the residual-stream norm after each block.
 *
 * `ready` is false until every layer is lit: the prediction only exists once
 * the last layer has run, so it is not shown earlier.
 */
export function showCandidates(frame: Frame | null, next: Frame | null, ready: boolean, residMax: number): void {
  const k = frame ? `${frame.h.pos}|${next?.h.token_id ?? ""}|${ready}|${residMax}` : "none";
  if (k === key) return;
  key = k;
  list.replaceChildren();
  resid.replaceChildren();
  if (!frame) {
    caption.textContent = "Run a prompt to see what the model predicts.";
    return;
  }

  const norms = frame.h.resid;
  norms.forEach((n, i) => {
    const bar = document.createElement("div");
    bar.className = "resid-bar";
    bar.style.height = `${Math.max(4, Math.min(n / residMax, 1) * 100)}%`;
    bar.title = `${i === 0 ? "Embedding" : `After layer ${i}`}: ${n.toFixed(2)}`;
    if (n > residMax) bar.classList.add("clipped");
    resid.append(bar);
  });

  if (!ready) {
    caption.textContent = "Appears once the last layer has run.";
    return;
  }
  const chosen = next?.h.token_id;
  caption.textContent = `After “${showToken(frame.h.token)}”, the model predicts:`;
  for (const c of frame.h.top) {
    const row = document.createElement("div");
    row.className = "cand" + (c.id === chosen ? " chosen" : "");
    const tok = document.createElement("span");
    tok.className = "cand-token";
    tok.textContent = showToken(c.token);
    const bar = document.createElement("span");
    bar.className = "cand-bar";
    const fill = document.createElement("i");
    fill.style.width = `${(c.p * 100).toFixed(2)}%`;
    bar.append(fill);
    const pct = document.createElement("span");
    pct.className = "cand-p";
    pct.textContent = `${(c.p * 100).toFixed(1)}%`;
    row.append(tok, bar, pct);
    list.append(row);
  }
  if (next && !frame.h.top.some((c) => c.id === chosen)) {
    const row = document.createElement("div");
    row.className = "cand-note";
    row.textContent = `${next.h.is_prompt ? "Your prompt continues with" : "Sampled"} “${showToken(next.h.token)}”, outside the top 10.`;
    list.append(row);
  }
}
