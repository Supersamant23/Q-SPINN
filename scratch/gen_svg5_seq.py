#!/usr/bin/env python3
OUT = r'c:\Users\adity\Documents\College\Projects\Q-SPINN\assets\q_spinn_cicd_pipeline.svg'

W, H = 1600, 500

def esc(s): return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

parts = []
# Background
parts.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

# Title
parts.append(f'<text x="{W//2}" y="50" text-anchor="middle" fill="#212121" font-size="24" font-weight="bold" font-family="Arial,sans-serif">Q-SPINN: CI/CD Pipeline \u2014 Sequential Overview (Conceptual/Developmental)</text>')

# Phases Backgrounds
PHASES = [
    (40, 520, "Phase 1: Push", "#E3F2FD", "#90CAF9"),
    (540, 1060, "Phase 2: Pull Request", "#FFF3E0", "#FFCC80"),
    (1080, 1560, "Phase 3: Merge to main / Manual approval", "#E8F5E9", "#A5D6A7")
]
for p1, p2, label, bg, border in PHASES:
    pw = p2 - p1
    parts.append(f'<rect x="{p1}" y="{90}" width="{pw}" height="{280}" rx="12" fill="{bg}" stroke="{border}" stroke-width="2"/>')
    parts.append(f'<rect x="{p1}" y="{90}" width="{pw}" height="{40}" rx="12" fill="{border}"/>')
    parts.append(f'<text x="{p1+pw//2}" y="{116}" text-anchor="middle" fill="#212121" font-size="16" font-weight="bold" font-family="Arial,sans-serif">{esc(label)}</text>')

def node(cx, cy, color, title, tool, dur):
    nw, nh = 200, 100
    x = cx - nw//2
    y = cy - nh//2
    p = []
    # Shadow
    p.append(f'<rect x="{x+4}" y="{y+4}" width="{nw}" height="{nh}" rx="8" fill="#00000020"/>')
    p.append(f'<rect x="{x}" y="{y}" width="{nw}" height="{nh}" rx="8" fill="#FFFFFF" stroke="{color}" stroke-width="2.5"/>')
    p.append(f'<rect x="{x}" y="{y}" width="{nw}" height="{6}" fill="{color}" rx="8"/>')
    
    p.append(f'<text x="{cx}" y="{y+35}" text-anchor="middle" fill="#424242" font-size="15" font-weight="bold" font-family="Arial,sans-serif">{esc(title)}</text>')
    p.append(f'<text x="{cx}" y="{y+60}" text-anchor="middle" fill="#757575" font-size="13" font-family="Monospace,sans-serif">{esc(tool)}</text>')
    
    # Duration badge
    p.append(f'<rect x="{cx-30}" y="{y+75}" width="{60}" height="{18}" rx="9" fill="{color}"/>')
    p.append(f'<text x="{cx}" y="{y+88}" text-anchor="middle" fill="#FFFFFF" font-size="11" font-weight="bold" font-family="Arial,sans-serif">{esc(dur)}</text>')
    return '\n  '.join(p)

def arrow_path(x1, y1, x2, y2, color='#9E9E9E'):
    return f'<path d="M {x1} {y1} L {x2-8} {y2}" stroke="{color}" stroke-width="3" fill="none" marker-end="url(#arr)"/>'

def marker(mid, color, size=12):
    hs = size // 2
    return f'<marker id="{mid}" markerWidth="{size}" markerHeight="{size}" refX="{size-2}" refY="{hs}" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,{size} L{size},{hs} z" fill="{color}"/></marker>'

parts.insert(0, f'<defs>\n  {marker("arr","#9E9E9E")}\n</defs>')

GREEN = "#43A047"
AMBER = "#F57C00"
RED   = "#E53935"

# Nodes (cx, cy, color, name, tool, dur)
NODES = [
    (170, 240, GREEN, "Lint & Type-check", "ruff / mypy", "~30s"),
    (410, 240, GREEN, "Unit Tests", "pytest", "~45s"),
    (670, 240, AMBER, "Integration Test", "pytest+compose", "~3min"),
    (930, 240, RED,   "Regression Test", "MLflow", "~10min"),
    (1200, 240, AMBER, "Build & Push", "docker buildx", "~4min"),
    (1440, 240, GREEN, "Deploy", "docker-compose up -d", "~30s")
]

for cx, cy, color, name, tool, dur in NODES:
    parts.append(node(cx, cy, color, name, tool, dur))

# Arrows
for i in range(len(NODES)-1):
    x1 = NODES[i][0] + 100
    y1 = NODES[i][1]
    x2 = NODES[i+1][0] - 100
    y2 = NODES[i+1][1]
    parts.append(arrow_path(x1, y1, x2, y2))

# Legend
leg_y = 420
parts.append(f'<rect x="{W//2-190}" y="{leg_y}" width="380" height="40" rx="4" fill="#F5F5F5" stroke="#E0E0E0" stroke-width="1.5"/>')
items = [(GREEN, "Fast (<1min)"), (AMBER, "Medium (1-5min)"), (RED, "Slow (>5min)")]
lx = W//2 - 170
for c, l in items:
    parts.append(f'<rect x="{lx}" y="{leg_y+12}" width="20" height="16" rx="4" fill="{c}"/>')
    parts.append(f'<text x="{lx+25}" y="{leg_y+25}" fill="#424242" font-size="12" font-weight="bold" font-family="Arial,sans-serif">{esc(l)}</text>')
    lx += 130

# Footnote
parts.append(f'<text x="{W//2}" y="{480}" text-anchor="middle" fill="#757575" font-size="13" font-style="italic" font-family="Arial,sans-serif">Future scope: replace docker-compose deploy step with kubectl apply (Kubernetes) \u2014 see \u00a710.</text>')

svg = f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n  {"".join(chr(10)+"  "+p for p in parts)}\n</svg>'
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(svg)
print("SUCCESS")
