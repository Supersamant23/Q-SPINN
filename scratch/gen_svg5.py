#!/usr/bin/env python3
OUT = r'c:\Users\adity\Documents\College\Projects\Q-SPINN\assets\q_spinn_cicd_gantt.svg'

W, H = 950, 950

def esc(s): return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

parts = []
# Background
parts.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

# Title
parts.append(f'<text x="{W//2}" y="50" text-anchor="middle" fill="#212121" font-size="24" font-weight="bold" font-family="Arial,sans-serif">Q-SPINN: CI/CD Pipeline \u2014 Gantt Overview (Conceptual/Developmental)</text>')

# X-axis Phases
PHASES = [
    (150, 380, "Push"),
    (380, 650, "Pull Request"),
    (650, 900, "Merge to main / Manual approval")
]
for p1, p2, label in PHASES:
    parts.append(f'<line x1="{p1}" y1="100" x2="{p1}" y2="800" stroke="#B0BEC5" stroke-width="2" stroke-dasharray="8,8"/>')
    parts.append(f'<text x="{(p1+p2)//2}" y="90" text-anchor="middle" fill="#546E7A" font-size="14" font-weight="bold" font-family="Arial,sans-serif">{esc(label)}</text>')

# Rows
ROWS = [
    ("Lint + type-check", 150),
    ("Unit tests", 250),
    ("Integration test", 350),
    ("Model regression test", 450),
    ("Container build & push", 550),
    ("Deploy", 650)
]
for name, y in ROWS:
    # Row background zebra
    parts.append(f'<rect x="20" y="{y-40}" width="{W-40}" height="80" fill="#FAFAFA"/>')
    parts.append(f'<text x="40" y="{y+5}" fill="#424242" font-size="14" font-weight="bold" font-family="Arial,sans-serif">{esc(name)}</text>')

# Bars (x, y, w, color, tool, duration)
GREEN = "#43A047"
AMBER = "#F57C00"
RED   = "#D32F2F"

BARS = [
    (160, 150, 100, GREEN, "ruff / mypy", "~30s"),
    (270, 250, 100, GREEN, "pytest", "~45s"),
    (390, 350, 120, AMBER, "pytest+docker-compose", "~3m"),
    (520, 450, 120, RED,   "MLflow", "~10m"),
    (660, 550, 120, AMBER, "docker buildx", "~4m"),
    (790, 650, 100, GREEN, "docker-compose up -d", "~30s")
]

for x, y, w, color, tool, dur in BARS:
    bh = 40
    by = y - bh//2
    # Shadow
    parts.append(f'<rect x="{x+3}" y="{by+3}" width="{w}" height="{bh}" rx="6" fill="#00000020"/>')
    parts.append(f'<rect x="{x}" y="{by}" width="{w}" height="{bh}" rx="6" fill="{color}"/>')
    parts.append(f'<text x="{x+w//2}" y="{by+16}" text-anchor="middle" fill="#FFFFFF" font-size="12" font-weight="bold" font-family="Monospace,sans-serif">{esc(tool)}</text>')
    parts.append(f'<text x="{x+w//2}" y="{by+32}" text-anchor="middle" fill="#FFFFFF" font-size="11" font-weight="bold" font-family="Arial,sans-serif">{esc(dur)}</text>')

# Arrows
def marker(mid, color, size=10):
    hs = size // 2
    return f'<marker id="{mid}" markerWidth="{size}" markerHeight="{size}" refX="{size-2}" refY="{hs}" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,{size} L{size},{hs} z" fill="{color}"/></marker>'

parts.insert(0, f'<defs>\n  {marker("arr","#757575")}\n</defs>')

for i in range(len(BARS)-1):
    x1 = BARS[i][0] + BARS[i][2]
    y1 = BARS[i][1]
    x2 = BARS[i+1][0]
    y2 = BARS[i+1][1]
    
    # Path: Right, Down, Right
    mx = (x1 + x2) // 2
    path = f"M {x1} {y1} L {mx} {y1} L {mx} {y2} L {x2-4} {y2}"
    parts.append(f'<path d="{path}" stroke="#757575" stroke-width="2" fill="none" marker-end="url(#arr)"/>')

# Legend
leg_y = 800
parts.append(f'<rect x="{W//2-180}" y="{leg_y}" width="360" height="40" rx="4" fill="#F5F5F5" stroke="#E0E0E0" stroke-width="1.5"/>')
items = [(GREEN, "Fast (<1m)"), (AMBER, "Medium (1-5m)"), (RED, "Slow (>5m)")]
lx = W//2 - 150
for c, l in items:
    parts.append(f'<rect x="{lx}" y="{leg_y+12}" width="20" height="16" rx="4" fill="{c}"/>')
    parts.append(f'<text x="{lx+25}" y="{leg_y+25}" fill="#424242" font-size="12" font-weight="bold" font-family="Arial,sans-serif">{esc(l)}</text>')
    lx += 110

# Footnote
parts.append(f'<text x="{W//2}" y="{880}" text-anchor="middle" fill="#757575" font-size="12" font-style="italic" font-family="Arial,sans-serif">Future scope: replace docker-compose deploy step with kubectl apply (Kubernetes) \u2014 see \u00a710.</text>')

svg = f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n  {"".join(chr(10)+"  "+p for p in parts)}\n</svg>'
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(svg)
print("SUCCESS")
