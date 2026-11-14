const card = document.getElementById("hover") as HTMLDivElement;

export interface HoverRow {
  label: string;
  value: string;
}

/** Show the hover card near the pointer, or hide it with null. */
export function showHover(title: string | null, rows: HoverRow[] = [], x = 0, y = 0): void {
  if (title === null) {
    card.hidden = true;
    return;
  }
  card.replaceChildren();
  const h = document.createElement("div");
  h.className = "hover-title";
  h.textContent = title;
  card.append(h);
  for (const r of rows) {
    const row = document.createElement("div");
    row.className = "hover-row";
    const k = document.createElement("span");
    k.textContent = r.label;
    const v = document.createElement("b");
    v.textContent = r.value;
    row.append(k, v);
    card.append(row);
  }
  card.hidden = false;
  const pad = 16;
  const w = card.offsetWidth;
  const hgt = card.offsetHeight;
  const left = x + pad + w > window.innerWidth ? x - pad - w : x + pad;
  const top = y + pad + hgt > window.innerHeight ? y - pad - hgt : y + pad;
  card.style.left = `${Math.max(8, left)}px`;
  card.style.top = `${Math.max(8, top)}px`;
}
