#!/usr/bin/env python3
import math

OUT = r'c:\Users\adity\Documents\College\Projects\Q-SPINN\assets\q_spinn_inference_pipeline.svg'

# ─────────────────── Colour schemes ───────────────────
TEAL   = {'bg':'#E0F2F1','stroke':'#00897B','badge':'#00897B','text':'#004D40','dark':'#00796B'}
AMBER  = {'bg':'#FFF3E0','stroke':'#F57C00','badge':'#F57C00','text':'#E65100','dark':'#EF6C00'}
INDIGO = {'bg':'#E8EAF6','stroke':'#3F51B5','badge':'#3F51B5','text':'#1A237E','dark':'#303F9F'}
GREEN  = {'bg':'#E8F5E9','stroke':'#43A047','badge':'#43A047','text':'#1B5E20','dark':'#388E3C'}
SLATE  = {'bg':'#F5F5F5','stroke':'#757575','badge':'#757575','text':'#212121','dark':'#616161'}

W, H = 1600, 900
NW, NH = 180, 140   # Node sizes

Y = 450 # Main horizontal center line

# 7 Columns
C = [120, 340, 560, 780, 1000, 1220, 1440]

def esc(s): return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def shadow(x, y, w, h, rx=10):
    return f'<rect x="{x+4}" y="{y+4}" width="{w}" height="{h}" rx="{rx}" fill="#00000015"/>'

def node(cx, cy, w, h, sc, badge, title, subs):
    x, y = cx - w//2, cy - h//2
    RX = 12
    bh = 26
    parts = []
    parts.append(shadow(x, y, w, h, RX))
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{RX}" '
                 f'fill="{sc["bg"]}" stroke="{sc["stroke"]}" stroke-width="2"/>')
    parts.append(f'<path d="M {x+RX} {y} L {x+w-RX} {y} A {RX} {RX} 0 0 1 {x+w} {y+RX} '
                 f'L {x+w} {y+bh} L {x} {y+bh} L {x} {y+RX} A {RX} {RX} 0 0 1 {x+RX} {y} Z" '
                 f'fill="{sc["badge"]}"/>')
    parts.append(f'<text x="{cx}" y="{y+17}" text-anchor="middle" fill="white" '
                 f'font-size="11" font-weight="600" font-family="Arial,sans-serif">'
                 f'{esc(badge)}</text>')
    
    ty = y + bh + 22
    lines = title.split('\n')
    for tl in lines:
        parts.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" fill="{sc["text"]}" '
                     f'font-size="13" font-weight="bold" font-family="Arial,sans-serif">'
                     f'{esc(tl)}</text>')
        ty += 16
    
    sy = ty + 6
    for s in subs:
        if s:
            parts.append(f'<text x="{cx}" y="{sy}" text-anchor="middle" '
                         f'fill="{sc["text"]}" font-size="11" opacity="0.9" '
                         f'font-family="Arial,sans-serif">{esc(s)}</text>')
            sy += 14
    return '\n  '.join(parts)

def diamond(cx, cy, r, color, text):
    parts = []
    parts.append(f'<polygon points="{cx},{cy-r} {cx+r},{cy} {cx},{cy+r} {cx-r},{cy}" fill="white" stroke="{color}" stroke-width="2"/>')
    parts.append(f'<text x="{cx}" y="{cy+4}" text-anchor="middle" fill="{color}" font-size="11" font-weight="bold" font-family="Arial,sans-serif">{esc(text)}</text>')
    return '\n  '.join(parts)

def arrow_path(d, label, lx, ly, color, mid_id, anchor="middle", dashed=False, bold=False):
    dash = 'stroke-dasharray="6,5" ' if dashed else ''
    fw = 'font-weight="bold"' if bold else 'font-weight="600"'
    parts = [f'<path d="{d}" stroke="{color}" stroke-width="2.5" fill="none" {dash}marker-end="url(#{mid_id})"/>']
    if label:
        lines = label.split('\n')
        offsets = [-6, 12] if len(lines) > 1 else [-6]
        for i, text_line in enumerate(lines):
            y_pos = ly + offsets[i]
            parts.append(f'<text x="{lx}" y="{y_pos}" text-anchor="{anchor}" fill="white" stroke="white" stroke-width="4" stroke-linejoin="round" font-size="12" {fw} font-family="Arial,sans-serif">{esc(text_line)}</text>')
            parts.append(f'<text x="{lx}" y="{y_pos}" text-anchor="{anchor}" fill="{color}" font-size="12" {fw} font-family="Arial,sans-serif">{esc(text_line)}</text>')
    return '\n  '.join(parts)

def marker(mid, color, size=10):
    hs = size // 2
    return f'<marker id="{mid}" markerWidth="{size}" markerHeight="{size}" refX="{size-2}" refY="{hs}" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,{size} L{size},{hs} z" fill="{color}"/></marker>'

parts = []
parts.append(f'<defs>\n  {marker("arr","#757575")}\n  {marker("arr-r","#D32F2F")}\n  {marker("arr-g","#388E3C")}\n  {marker("arr-s","#9E9E9E",8)}\n</defs>')
parts.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

# Title
parts.append(f'<text x="{W//2}" y="60" text-anchor="middle" fill="#212121" font-size="24" font-weight="bold" font-family="Arial,sans-serif">Q-SPINN: Inference/Production Architecture \u2014 Single-Node, v1.0 (Conceptual/Developmental)</text>')

# Nodes
parts.append(node(C[0], Y, NW, NH, TEAL, 'Node 1', 'Live Data\n(Ingestor)', ['Binance WebSocket']))
parts.append(node(C[1], Y, NW, NH, TEAL, 'Node 2', 'Delta Encoder', ['Stateless transform']))
parts.append(node(C[2], Y, NW, NH, AMBER, 'Node 3', 'SNN Gate', ['Loaded trained', 'SNN weights']))
# Diamond for SNN Gate
parts.append(diamond(C[2]+NW//2+25, Y, 25, AMBER['stroke'], 'Spike?'))

parts.append(node(C[3], Y, NW, NH, INDIGO, 'Node 4', '\u2102-Covariance\nState Update', ['Updates in-memory', 'running covariance']))
parts.append(node(C[4], Y, NW, NH, GREEN, 'Node 5', 'PINN Inference', ['Forward pass only']))
parts.append(node(C[5], Y, NW, NH, SLATE, 'Node 6', 'API Gateway', ['FastAPI endpoint']))
parts.append(node(C[6], Y, NW, NH, SLATE, 'Node 7', 'Dashboard', ['Custom HTML/JS', 'health-check UI']))
parts.append(node(C[5], Y-200, NW, 80, SLATE, 'External', 'Consumer', []))

# Arrows
parts.append(arrow_path(f'M {20} {Y} L {C[0]-NW//2} {Y}', 'raw tick stream\n(WebSocket JSON)', (20+C[0]-NW//2)//2, Y, '#757575', 'arr'))
parts.append(arrow_path(f'M {C[0]+NW//2} {Y} L {C[1]-NW//2} {Y}', 'raw tick dict', (C[0]+C[1])//2, Y, '#757575', 'arr'))
parts.append(arrow_path(f'M {C[1]+NW//2} {Y} L {C[2]-NW//2} {Y}', '\u0394E scalar', (C[1]+C[2])//2, Y, '#757575', 'arr'))

# SNN to Diamond
parts.append(f'<line x1="{C[2]+NW//2}" y1="{Y}" x2="{C[2]+NW//2}" y2="{Y}" stroke="{AMBER["stroke"]}" stroke-width="2.5"/>') # Small gap connector

# Diamond YES -> C-Cov
parts.append(arrow_path(f'M {C[2]+NW//2+50} {Y} L {C[3]-NW//2} {Y}', 'YES\nS(t)=1', (C[2]+NW//2+50+C[3]-NW//2)//2, Y, '#388E3C', 'arr-g', bold=True))

# Diamond NO -> Loop back to Node 1
parts.append(arrow_path(f'M {C[2]+NW//2+25} {Y-25} C {C[2]+NW//2+25} {Y-180}, {C[0]} {Y-180}, {C[0]} {Y-NH//2}', 'NO / DORMANT\n~99% of ticks', (C[0]+C[2])//2, Y-180, '#D32F2F', 'arr-r', dashed=True, bold=True))

# C-Cov -> PINN
parts.append(arrow_path(f'M {C[3]+NW//2} {Y} L {C[4]-NW//2} {Y}', 'complex64 3\u00d73\n(updated C)', (C[3]+C[4])//2, Y, '#757575', 'arr'))

# PINN -> API
parts.append(arrow_path(f'M {C[4]+NW//2} {Y} L {C[5]-NW//2} {Y}', 'softmax [\u2191,\u2192,\u2193]\n+ confidence', (C[4]+C[5])//2, Y, '#757575', 'arr'))

# API -> Dashboard
parts.append(arrow_path(f'M {C[5]+NW//2} {Y} L {C[6]-NW//2} {Y}', 'metrics stream', (C[5]+C[6])//2, Y, '#757575', 'arr'))

# API -> External Consumer
parts.append(arrow_path(f'M {C[5]} {Y-NH//2} L {C[5]} {Y-200+40}', 'JSON prediction\nresponse', C[5], (Y-NH//2+Y-160)//2, '#757575', 'arr', anchor='middle'))

# Latency Bracket (Bottom)
ly = Y + 150
parts.append(f'<path d="M {C[0]-NW//2} {ly-10} L {C[0]-NW//2} {ly} L {C[5]+NW//2} {ly} L {C[5]+NW//2} {ly-10}" stroke="#424242" stroke-width="2" fill="none"/>')
parts.append(f'<text x="{(C[0]-NW//2+C[5]+NW//2)//2}" y="{ly+20}" text-anchor="middle" fill="#424242" font-size="14" font-weight="bold" font-family="Arial,sans-serif">Target: &lt;50ms end-to-end</text>')

# Legend
leg_x, leg_y = 40, H - 150
parts.append(f'<rect x="{leg_x}" y="{leg_y}" width="210" height="130" rx="8" fill="#FAFAFA" stroke="#E0E0E0" stroke-width="1.5"/>')
parts.append(f'<text x="{leg_x+105}" y="{leg_y+20}" text-anchor="middle" fill="#424242" font-size="12" font-weight="bold" font-family="Arial,sans-serif">Legend</text>')
items = [
    (TEAL['badge'], 'Data Sources / Preprocessing'),
    (AMBER['badge'], 'SNN Gatekeeper'),
    (INDIGO['badge'], '\u2102-Covariance Layer'),
    (GREEN['badge'], 'PINN Engine / Output'),
    (SLATE['badge'], 'Infrastructure'),
]
iy = leg_y + 40
for color, label in items:
    parts.append(f'<rect x="{leg_x+15}" y="{iy-10}" width="14" height="14" rx="3" fill="{color}"/>')
    parts.append(f'<text x="{leg_x+38}" y="{iy+1}" fill="#424242" font-size="11.5" font-family="Arial,sans-serif">{esc(label)}</text>')
    iy += 20

svg = f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n  {"".join(chr(10)+"  "+p for p in parts)}\n</svg>'
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(svg)
print("SUCCESS")
