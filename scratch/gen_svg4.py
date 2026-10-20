#!/usr/bin/env python3
OUT = r'c:\Users\adity\Documents\College\Projects\Q-SPINN\assets\q_spinn_dashboard_mockup.svg'

W, H = 1600, 900

def esc(s): return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

parts = []
# Background
parts.append(f'<rect width="{W}" height="{H}" fill="#F5F5F5"/>')

# Left Sidebar
SIDE_W = 180
parts.append(f'<rect width="{SIDE_W}" height="{H}" fill="#263238"/>')
parts.append(f'<text x="20" y="40" fill="#FFFFFF" font-size="16" font-weight="bold" font-family="Arial,sans-serif">Q-SPINN Monitor</text>')
parts.append(f'<line x1="20" y1="55" x2="{SIDE_W-20}" y2="55" stroke="#37474F" stroke-width="2"/>')

links = ["Overview", "Containers", "Metrics", "Predictions", "Ablation"]
ly = 90
for i, l in enumerate(links):
    color = "#80CBC4" if i == 0 else "#B0BEC5" # Highlight first
    parts.append(f'<circle cx="25" cy="{ly-4}" r="4" fill="{color}"/>')
    parts.append(f'<text x="40" y="{ly}" fill="{color}" font-size="14" font-weight="bold" font-family="Arial,sans-serif">{esc(l)}</text>')
    ly += 40

# Top Bar
TOP_H = 50
parts.append(f'<rect x="{SIDE_W}" y="0" width="{W-SIDE_W}" height="{TOP_H}" fill="#FFFFFF" stroke="#E0E0E0" stroke-width="1"/>')
parts.append(f'<text x="{SIDE_W+20}" y="30" fill="#424242" font-size="16" font-weight="bold" font-family="Arial,sans-serif">Q-SPINN Health Monitor \u2014 v1.0 (Conceptual/Developmental)</text>')
parts.append(f'<text x="{W-20}" y="30" text-anchor="end" fill="#757575" font-size="14" font-weight="bold" font-family="Monospace,sans-serif">11:34:00 IST</text>')

# Card Grid
GAP = 15
CONTENT_W = W - SIDE_W - 2*GAP
CONTENT_H = H - TOP_H - 2*GAP

# We have 7 cards. Layout:
# Row 1: 3 cards
# Row 2: 2 cards
# Row 3: 2 cards
card_w3 = (CONTENT_W - 2*GAP) // 3
card_w2 = (CONTENT_W - GAP) // 2
card_h = (CONTENT_H - 2*GAP) // 3

def card(cx, cy, cw, ch, title):
    p = []
    # Shadow
    p.append(f'<rect x="{cx+3}" y="{cy+3}" width="{cw}" height="{ch}" rx="8" fill="#00000015"/>')
    p.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="8" fill="#FFFFFF" stroke="#E0E0E0" stroke-width="1"/>')
    # Title bar
    p.append(f'<path d="M {cx} {cy+8} A 8 8 0 0 1 {cx+8} {cy} L {cx+cw-8} {cy} A 8 8 0 0 1 {cx+cw} {cy+8} L {cx+cw} {cy+35} L {cx} {cy+35} Z" fill="#ECEFF1"/>')
    p.append(f'<line x1="{cx}" y1="{cy+35}" x2="{cx+cw}" y2="{cy+35}" stroke="#CFD8DC" stroke-width="1"/>')
    p.append(f'<text x="{cx+15}" y="{cy+23}" fill="#37474F" font-size="13" font-weight="bold" font-family="Arial,sans-serif">{esc(title)}</text>')
    return '\n  '.join(p)

def grid_status(cx, cy, cw, ch):
    p = []
    services = ["ingestor", "snn-worker", "pinn-broker", "evaluator", "api-gateway", "dashboard", "message-queue", "mlflow-server"]
    tw = (cw - 30) // 2
    th = (ch - 55) // 4
    for i, s in enumerate(services):
        tx = cx + 10 + (i%2)*(tw+10)
        ty = cy + 45 + (i//2)*(th+5)
        p.append(f'<rect x="{tx}" y="{ty}" width="{tw}" height="{th}" rx="4" fill="#F5F5F5" stroke="#E0E0E0" stroke-width="1"/>')
        p.append(f'<text x="{tx+5}" y="{ty+15}" fill="#424242" font-size="11" font-weight="bold" font-family="Arial,sans-serif">{esc(s)}</text>')
        p.append(f'<rect x="{tx+tw-45}" y="{ty+6}" width="40" height="12" rx="2" fill="#43A047"/>')
        p.append(f'<text x="{tx+tw-25}" y="{ty+15}" text-anchor="middle" fill="#FFFFFF" font-size="9" font-weight="bold" font-family="Arial,sans-serif">RUNNING</text>')
        # Resource bars
        p.append(f'<rect x="{tx+5}" y="{ty+22}" width="{tw//2-10}" height="4" rx="2" fill="#CFD8DC"/>')
        p.append(f'<rect x="{tx+5}" y="{ty+22}" width="20" height="4" rx="2" fill="#1E88E5"/>')
        p.append(f'<rect x="{tx+tw//2+5}" y="{ty+22}" width="{tw//2-10}" height="4" rx="2" fill="#CFD8DC"/>')
        p.append(f'<rect x="{tx+tw//2+5}" y="{ty+22}" width="35" height="4" rx="2" fill="#FB8C00"/>')
    return '\n  '.join(p)

def chart_line(cx, cy, cw, ch, is_loss=False):
    p = []
    plot_x = cx + 30
    plot_y = cy + 50
    plot_w = cw - 50
    plot_h = ch - 80
    
    # Axes
    p.append(f'<polyline points="{plot_x},{plot_y} {plot_x},{plot_y+plot_h} {plot_x+plot_w},{plot_y+plot_h}" stroke="#B0BEC5" stroke-width="2" fill="none"/>')
    
    if not is_loss:
        # Regime shift shade
        rx = plot_x + 60
        rw = 40
        p.append(f'<rect x="{rx}" y="{plot_y}" width="{rw}" height="{plot_h}" fill="#FFF9C4" opacity="0.5"/>')
        p.append(f'<text x="{rx+20}" y="{plot_y+10}" text-anchor="middle" fill="#F57F17" font-size="10" font-family="Arial,sans-serif">Regime Shift</text>')
        
        # 3 lines
        p.append(f'<path d="M {plot_x} {plot_y+plot_h-10} Q {plot_x+60} {plot_y+20}, {plot_x+120} {plot_y+20} Q {plot_x+250} {plot_y+20}, {plot_x+plot_w} {plot_y+plot_h-40}" fill="none" stroke="#1E88E5" stroke-width="2"/>') # AAPL
        p.append(f'<path d="M {plot_x} {plot_y+plot_h-5} Q {plot_x+70} {plot_y+50}, {plot_x+140} {plot_y+50} Q {plot_x+250} {plot_y+50}, {plot_x+plot_w} {plot_y+plot_h-30}" fill="none" stroke="#F57C00" stroke-width="2"/>') # MSFT
        p.append(f'<path d="M {plot_x} {plot_y+plot_h-15} Q {plot_x+80} {plot_y+10}, {plot_x+150} {plot_y+10} Q {plot_x+280} {plot_y+10}, {plot_x+plot_w} {plot_y+plot_h-50}" fill="none" stroke="#43A047" stroke-width="2"/>') # NVDA
    else:
        # 3 lines for loss
        p.append(f'<path d="M {plot_x} {plot_y+10} Q {plot_x+40} {plot_y+plot_h-40}, {plot_x+plot_w} {plot_y+plot_h-50}" fill="none" stroke="#1E88E5" stroke-width="2"/>')
        p.append(f'<path d="M {plot_x} {plot_y+30} Q {plot_x+60} {plot_y+plot_h-20}, {plot_x+plot_w} {plot_y+plot_h-30}" fill="none" stroke="#FB8C00" stroke-width="2"/>')
        p.append(f'<path d="M {plot_x} {plot_y+5} Q {plot_x+50} {plot_y+plot_h-50}, {plot_x+plot_w} {plot_y+plot_h-60}" fill="none" stroke="#43A047" stroke-width="2" stroke-dasharray="4,4"/>')
        
    return '\n  '.join(p)

def dual_chart(cx, cy, cw, ch):
    p = []
    plot_x = cx + 40
    plot_y = cy + 50
    plot_w = cw - 80
    plot_h = ch - 80
    # Axes
    p.append(f'<polyline points="{plot_x},{plot_y} {plot_x},{plot_y+plot_h} {plot_x+plot_w},{plot_y+plot_h}" stroke="#B0BEC5" stroke-width="2" fill="none"/>')
    p.append(f'<line x1="{plot_x+plot_w}" y1="{plot_y}" x2="{plot_x+plot_w}" y2="{plot_y+plot_h}" stroke="#B0BEC5" stroke-width="2"/>')
    
    # F1
    p.append(f'<path d="M {plot_x} {plot_y+60} L {plot_x+40} {plot_y+50} L {plot_x+80} {plot_y+30} L {plot_x+120} {plot_y+35} L {plot_x+plot_w} {plot_y+20}" fill="none" stroke="#8E24AA" stroke-width="2.5"/>')
    # DSR
    p.append(f'<path d="M {plot_x} {plot_y+70} L {plot_x+40} {plot_y+65} L {plot_x+80} {plot_y+20} L {plot_x+120} {plot_y+25} L {plot_x+plot_w} {plot_y+10}" fill="none" stroke="#00897B" stroke-width="2.5"/>')
    return '\n  '.join(p)

def area_chart(cx, cy, cw, ch):
    p = []
    plot_x = cx + 30
    plot_y = cy + 50
    plot_w = cw - 50
    plot_h = ch - 80
    
    p.append(f'<polyline points="{plot_x},{plot_y} {plot_x},{plot_y+plot_h} {plot_x+plot_w},{plot_y+plot_h}" stroke="#B0BEC5" stroke-width="2" fill="none"/>')
    
    path_d = f"M {plot_x} {plot_y+plot_h} L {plot_x} {plot_y+plot_h-20} L {plot_x+40} {plot_y+plot_h-30} L {plot_x+80} {plot_y+plot_h-80} L {plot_x+120} {plot_y+plot_h-50} L {plot_x+160} {plot_y+plot_h-100} L {plot_x+200} {plot_y+plot_h-40} L {plot_x+plot_w} {plot_y+plot_h-45} L {plot_x+plot_w} {plot_y+plot_h} Z"
    p.append(f'<path d="{path_d}" fill="#81D4FA" opacity="0.6" stroke="#0288D1" stroke-width="2"/>')
    
    # Overload
    oy = plot_y + plot_h - 70
    p.append(f'<line x1="{plot_x}" y1="{oy}" x2="{plot_x+plot_w}" y2="{oy}" stroke="#E53935" stroke-width="2" stroke-dasharray="6,4"/>')
    p.append(f'<text x="{plot_x+10}" y="{oy-5}" fill="#E53935" font-size="10" font-weight="bold" font-family="Arial,sans-serif">Overload Threshold</text>')
    
    return '\n  '.join(p)

def heatmap(cx, cy, cw, ch):
    p = []
    plot_x = cx + 40
    plot_y = cy + 50
    tw = (cw - 60) // 10
    th = (ch - 70) // 3
    
    assets = ["AAPL", "MSFT", "NVDA"]
    colors = ["#FFEBEE", "#FFCDD2", "#EF9A9A", "#E57373", "#EF5350", "#F44336", "#E53935", "#D32F2F", "#C62828", "#B71C1C"]
    
    for i, a in enumerate(assets):
        p.append(f'<text x="{plot_x-5}" y="{plot_y+i*th+th//2+4}" text-anchor="end" fill="#616161" font-size="11" font-family="Arial,sans-serif">{a}</text>')
        import random
        for j in range(10):
            c_idx = random.randint(0, 5) if j < 4 else random.randint(3, 9) if j == 5 else random.randint(0, 4)
            c = colors[c_idx]
            p.append(f'<rect x="{plot_x+j*tw}" y="{plot_y+i*th}" width="{tw-2}" height="{th-2}" rx="2" fill="{c}"/>')
    
    return '\n  '.join(p)

def network(cx, cy, cw, ch):
    p = []
    nx, ny = cx + cw//2, cy + ch//2 + 10
    R = 40
    nodes = [
        (nx-60, ny-50, "#00897B"), (nx, ny-70, "#F57C00"), (nx+60, ny-50, "#3F51B5"),
        (nx+80, ny+10, "#43A047"), (nx+40, ny+60, "#757575"), (nx-40, ny+60, "#757575"),
        (nx-80, ny+10, "#757575"), (nx, ny, "#757575")
    ]
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (0,7), (1,7), (3,7)]
    for e1, e2 in edges:
        p.append(f'<line x1="{nodes[e1][0]}" y1="{nodes[e1][1]}" x2="{nodes[e2][0]}" y2="{nodes[e2][1]}" stroke="#CFD8DC" stroke-width="2"/>')
    for x, y, c in nodes:
        p.append(f'<circle cx="{x}" cy="{y}" r="15" fill="{c}"/>')
    return '\n  '.join(p)

# Row 1 (3 cards)
r1_y = TOP_H + GAP
c1 = SIDE_W + GAP
c2 = c1 + card_w3 + GAP
c3 = c2 + card_w3 + GAP

parts.append(card(c1, r1_y, card_w3, card_h, "Container Status Grid"))
parts.append(grid_status(c1, r1_y, card_w3, card_h))

parts.append(card(c2, r1_y, card_w3, card_h, "Spike Rate Timeline"))
parts.append(chart_line(c2, r1_y, card_w3, card_h))

parts.append(card(c3, r1_y, card_w3, card_h, "Loss Curves"))
parts.append(chart_line(c3, r1_y, card_w3, card_h, True))

# Row 2 (2 cards)
r2_y = r1_y + card_h + GAP
c4 = SIDE_W + GAP
c5 = c4 + card_w2 + GAP

parts.append(card(c4, r2_y, card_w2, card_h, "Prediction Accuracy"))
parts.append(dual_chart(c4, r2_y, card_w2, card_h))

parts.append(card(c5, r2_y, card_w2, card_h, "Queue Depth"))
parts.append(area_chart(c5, r2_y, card_w2, card_h))

# Row 3 (2 cards)
r3_y = r2_y + card_h + GAP
c6 = SIDE_W + GAP
c7 = c6 + card_w2 + GAP

parts.append(card(c6, r3_y, card_w2, card_h, "Latency Heatmap"))
parts.append(heatmap(c6, r3_y, card_w2, card_h))

parts.append(card(c7, r3_y, card_w2, card_h, "Docker Network Topology"))
parts.append(network(c7, r3_y, card_w2, card_h))

svg = f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n  {"".join(chr(10)+"  "+p for p in parts)}\n</svg>'
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(svg)
print("SUCCESS")
