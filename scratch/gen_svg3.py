#!/usr/bin/env python3
import math

OUT = r'c:\Users\adity\Documents\College\Projects\Q-SPINN\assets\q_spinn_ablation_study.svg'

# ─────────────────── Colour schemes ───────────────────
TEAL   = {'bg':'#E0F2F1','stroke':'#00897B','badge':'#00897B','text':'#004D40','dark':'#00796B'}
AMBER  = {'bg':'#FFF3E0','stroke':'#F57C00','badge':'#F57C00','text':'#E65100','dark':'#EF6C00'}
INDIGO = {'bg':'#E8EAF6','stroke':'#3F51B5','badge':'#3F51B5','text':'#1A237E','dark':'#303F9F'}
GREEN  = {'bg':'#E8F5E9','stroke':'#43A047','badge':'#43A047','text':'#1B5E20','dark':'#388E3C'}
BLUE   = {'bg':'#E3F2FD','stroke':'#1E88E5','badge':'#1E88E5','text':'#0D47A1','dark':'#1565C0'}
PURPLE = {'bg':'#F3E5F5','stroke':'#8E24AA','badge':'#8E24AA','text':'#4A148C','dark':'#6A1B9A'}
SLATE  = {'bg':'#F5F5F5','stroke':'#757575','badge':'#757575','text':'#212121','dark':'#616161'}
GHOST  = {'bg':'#FAFAFA','stroke':'#E0E0E0','badge':'#E0E0E0','text':'#9E9E9E','dark':'#BDBDBD'}

W, H = 950, 950
NW, NH = 140, 90  # Box size
COL_W = 180       # Column width

def esc(s): return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def shadow(x, y, w, h, rx=10):
    return f'<rect x="{x+4}" y="{y+4}" width="{w}" height="{h}" rx="{rx}" fill="#00000015"/>'

def ghost_cross(x, y, w, h):
    parts = []
    parts.append(f'<line x1="{x+10}" y1="{y+10}" x2="{x+w-10}" y2="{y+h-10}" stroke="#EEEEEE" stroke-width="2"/>')
    parts.append(f'<line x1="{x+10}" y1="{y+h-10}" x2="{x+w-10}" y2="{y+10}" stroke="#EEEEEE" stroke-width="2"/>')
    return '\n  '.join(parts)

def node(cx, cy, sc, title, active=True):
    x, y = cx - NW//2, cy - NH//2
    RX = 8
    parts = []
    
    if active:
        parts.append(shadow(x, y, NW, NH, RX))
        parts.append(f'<rect x="{x}" y="{y}" width="{NW}" height="{NH}" rx="{RX}" fill="{sc["bg"]}" stroke="{sc["stroke"]}" stroke-width="2"/>')
        parts.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" fill="{sc["text"]}" font-size="13" font-weight="bold" font-family="Arial,sans-serif">{esc(title)}</text>')
    else:
        parts.append(f'<rect x="{x}" y="{y}" width="{NW}" height="{NH}" rx="{RX}" fill="{GHOST["bg"]}" stroke="{GHOST["stroke"]}" stroke-width="2" stroke-dasharray="4,4"/>')
        parts.append(ghost_cross(x, y, NW, NH))
        parts.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" fill="{GHOST["text"]}" font-size="13" font-weight="bold" font-family="Arial,sans-serif">{esc(title)}</text>')
        
    return '\n  '.join(parts)

def arrow_path(cx, y1, y2, color='#9E9E9E'):
    return f'<path d="M {cx} {y1} L {cx} {y2}" stroke="{color}" stroke-width="2.5" fill="none" marker-end="url(#arr)"/>'

def marker(mid, color, size=10):
    hs = size // 2
    return f'<marker id="{mid}" markerWidth="{size}" markerHeight="{size}" refX="{size-2}" refY="{hs}" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,{size} L{size},{hs} z" fill="{color}"/></marker>'

parts = []
parts.append(f'<defs>\n  {marker("arr","#9E9E9E")}\n</defs>')
parts.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

# Title
parts.append(f'<text x="{W//2}" y="50" text-anchor="middle" fill="#212121" font-size="22" font-weight="bold" font-family="Arial,sans-serif">Q-SPINN: Ablation Study \u2014 5-Variant Comparison (Results Pending)</text>')

COLS = [
    ("Full Q-SPINN", [True, True, True, True]),
    ("SNN-only", [True, False, False, True]),
    ("PINN-only", [False, False, True, True]),
    ("LSTM Baseline", [False, False, False, False]),
    ("Transformer/DeepLOB", [False, False, False, False])
]

# Rows Y coords
RY = [180, 310, 440, 570]
IN_Y = 80
OUT_Y = 690
METRICS_Y = 780

start_x = (W - (5 * COL_W)) // 2 + COL_W // 2

for i, (col_title, acts) in enumerate(COLS):
    cx = start_x + i * COL_W
    
    # Column Title
    parts.append(f'<text x="{cx}" y="90" text-anchor="middle" fill="#424242" font-size="14" font-weight="bold" font-family="Arial,sans-serif">{esc(col_title)}</text>')
    
    # Input
    parts.append(f'<rect x="{cx-60}" y="{IN_Y+30}" width="120" height="30" rx="4" fill="#EEEEEE"/>')
    parts.append(f'<text x="{cx}" y="{IN_Y+50}" text-anchor="middle" fill="#616161" font-size="12" font-weight="bold" font-family="Arial,sans-serif">Data Input</text>')
    
    parts.append(arrow_path(cx, IN_Y+60, RY[0]-NH//2))
    
    # Components
    parts.append(node(cx, RY[0], AMBER, 'SNN Gatekeeper', acts[0]))
    parts.append(arrow_path(cx, RY[0]+NH//2, RY[1]-NH//2))
    
    parts.append(node(cx, RY[1], INDIGO, '\u2102-Covariance', acts[1]))
    parts.append(arrow_path(cx, RY[1]+NH//2, RY[2]-NH//2))
    
    if i == 3: # LSTM
        parts.append(node(cx, RY[2], BLUE, 'LSTM Cell', True))
    elif i == 4: # Transformer
        parts.append(node(cx, RY[2], PURPLE, 'Transformer Block', True))
    else:
        parts.append(node(cx, RY[2], GREEN, 'PINN Engine', acts[2]))
        
    parts.append(arrow_path(cx, RY[2]+NH//2, RY[3]-NH//2))
    
    parts.append(node(cx, RY[3], GREEN, 'Prediction Head', acts[3]))
    parts.append(arrow_path(cx, RY[3]+NH//2, OUT_Y))
    
    # Output
    parts.append(f'<rect x="{cx-65}" y="{OUT_Y}" width="130" height="30" rx="4" fill="#EEEEEE"/>')
    parts.append(f'<text x="{cx}" y="{OUT_Y+20}" text-anchor="middle" fill="#616161" font-size="12" font-weight="bold" font-family="Arial,sans-serif">Prediction Output</text>')
    
    # Metrics Box
    parts.append(f'<rect x="{cx-70}" y="{METRICS_Y}" width="140" height="90" fill="#FAFAFA" stroke="#E0E0E0" stroke-width="1.5"/>')
    my = METRICS_Y + 22
    for mt in ["F1: \u2014", "DSR: \u2014", "FLOPs: \u2014", "PDE Residual: \u2014"]:
        parts.append(f'<text x="{cx-60}" y="{my}" fill="#616161" font-size="12" font-family="Monospace,sans-serif">{esc(mt)}</text>')
        my += 20

# Legend
leg_y = H - 50
parts.append(f'<rect x="{W//2-180}" y="{leg_y}" width="360" height="40" rx="4" fill="#FAFAFA" stroke="#E0E0E0" stroke-width="1.5"/>')
parts.append(f'<rect x="{W//2-160}" y="{leg_y+12}" width="24" height="16" rx="4" fill="#E8F5E9" stroke="#43A047" stroke-width="2"/>')
parts.append(f'<text x="{W//2-125}" y="{leg_y+25}" fill="#424242" font-size="13" font-family="Arial,sans-serif">Active Component</text>')
parts.append(f'<rect x="{W//2+20}" y="{leg_y+12}" width="24" height="16" rx="4" fill="#FAFAFA" stroke="#E0E0E0" stroke-width="2" stroke-dasharray="4,4"/>')
parts.append(f'<text x="{W//2+55}" y="{leg_y+25}" fill="#424242" font-size="13" font-family="Arial,sans-serif">Bypassed / Absent (Ghost)</text>')

svg = f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n  {"".join(chr(10)+"  "+p for p in parts)}\n</svg>'
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(svg)
print("SUCCESS")
