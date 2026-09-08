"""Design tokens, injected CSS and the robot mark for the dashboard.

Streamlit's own theming (`.streamlit/config.toml`) sets the palette for its native widgets;
this module supplies the CSS for everything drawn by hand, so every colour decision is made
once, here, and the two theme sources stay in sync deliberately.

Two registers of colour, used for two different jobs:

* DENSE / SPARSE / FUSION are functional, not decorative -- dense (vector) retrieval is a
  continuous similarity signal, sparse (BM25 keyword) retrieval is a discrete term match,
  and these three badge which index (or both, fused via RRF) actually surfaced a result.
  They stay legible everywhere they label data.
* PINK, the gradient stops, and the glow are the page's *hero* register -- the reference
  mockup's near-black-with-warm-plum, glossy pink-to-violet mood -- reserved for the small
  set of elements meant to be the page's signature: the robot, the brand orb, the ask
  button, a focused input, the composite confidence ring. Everything measuring or labelling
  data stays flat so the hero register doesn't drown it out.
"""

from __future__ import annotations

BG = "#0A0510"
SURFACE = "#170F1F"
SURFACE_ALT = "#1F1428"
BORDER = "#2E1E3B"
TEXT = "#F5F1F8"
TEXT_MUTED = "#9C8CAE"
DENSE = "#5EE6C9"
SPARSE = "#F2A93B"
FUSION = "#B355F0"
PINK = "#EC6FC6"
DANGER = "#FF6B6B"

FUSION_GRADIENT = f"linear-gradient(135deg, #F0C6FF 0%, {FUSION} 55%, #6B21A8 100%)"
CTA_GRADIENT = f"linear-gradient(120deg, {PINK} 0%, {FUSION} 60%, #6B21A8 100%)"
GLOW = "rgba(179, 85, 240, 0.5)"

# A compact, hand-drawn robot -- pink-violet body, amber trim and antenna, glowing violet
# eyes -- echoing the reference mockup's mascot without an external image asset. Its own
# colour decisions still come from the tokens above, so it moves if the palette does.
ROBOT_SVG = f"""
<svg class="robot" viewBox="0 0 200 220" width="132" height="145" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="robotBody" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#F5D6FF"/>
      <stop offset="50%" stop-color="{PINK}"/>
      <stop offset="100%" stop-color="{FUSION}"/>
    </linearGradient>
    <radialGradient id="robotEye" cx="35%" cy="30%" r="75%">
      <stop offset="0%" stop-color="#F6ECFF"/>
      <stop offset="45%" stop-color="{FUSION}"/>
      <stop offset="100%" stop-color="#4C1D77"/>
    </radialGradient>
  </defs>
  <line x1="100" y1="8" x2="100" y2="32" stroke="{SPARSE}" stroke-width="4" stroke-linecap="round"/>
  <circle cx="100" cy="8" r="7" fill="{SPARSE}"/>
  <rect x="40" y="32" width="120" height="100" rx="36" fill="url(#robotBody)"/>
  <circle cx="34" cy="78" r="16" fill="{SPARSE}"/>
  <circle cx="166" cy="78" r="16" fill="{SPARSE}"/>
  <rect x="62" y="56" width="76" height="52" rx="20" fill="{BG}"/>
  <circle cx="86" cy="82" r="11" fill="url(#robotEye)"/>
  <circle cx="114" cy="82" r="11" fill="url(#robotEye)"/>
  <rect x="55" y="138" width="90" height="64" rx="22" fill="url(#robotBody)"/>
  <circle cx="75" cy="166" r="5" fill="{SPARSE}"/>
  <circle cx="125" cy="166" r="5" fill="{SPARSE}"/>
  <rect x="88" y="158" width="24" height="24" rx="6" fill="{BG}"/>
</svg>
"""

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    /* Warm plum-violet glow seated at the top of the page -- the reference's near-black
    glossy mood -- one glow, not one per panel, so a data dashboard stays legible. */
    background:
        radial-gradient(
            ellipse 1000px 520px at 50% -12%, rgba(179, 85, 240, 0.18), transparent 60%
        ),
        {BG};
    color: {TEXT};
}}

#MainMenu, footer {{ visibility: hidden; }}

.block-container {{
    padding-top: 2rem;
    max-width: 1180px;
}}

/* -- console header -- */
.console-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 0.9rem;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
    gap: 0.4rem;
}}
.console-header .brand {{
    display: flex;
    align-items: center;
    gap: 0.7rem;
}}
.brand-orb {{
    width: 30px;
    height: 30px;
    border-radius: 50%;
    flex-shrink: 0;
    background:
        radial-gradient(circle at 32% 26%, rgba(255,255,255,0.95), rgba(255,255,255,0) 38%),
        radial-gradient(circle at 68% 74%, {PINK} 0%, {FUSION} 45%, #4C1D77 100%);
    box-shadow: 0 0 16px 2px {GLOW}, inset 0 0 4px rgba(255,255,255,0.25);
}}
.console-header h1 {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    letter-spacing: 0.02em;
    margin: 0;
    color: {TEXT};
}}
.console-header .thesis {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: {TEXT_MUTED};
}}
.console-header .thesis .dense {{ color: {DENSE}; }}
.console-header .thesis .sparse {{ color: {SPARSE}; }}
.console-header .thesis .fusion {{ color: {FUSION}; }}

.status-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    margin-bottom: 1.6rem;
}}
.status-strip .dot {{
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 0.4rem;
    background: {DENSE};
    box-shadow: 0 0 6px {DENSE};
}}
.status-strip .dot.down {{ background: {DANGER}; box-shadow: 0 0 6px {DANGER}; }}

/* -- empty state -- */
.empty-state {{
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 2.2rem 1rem 2.4rem;
    gap: 0.7rem;
}}
.empty-state .robot svg {{ filter: drop-shadow(0 0 26px {GLOW}); }}
.empty-state h2 {{
    font-family: 'Inter', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: {TEXT};
    margin: 0.5rem 0 0;
}}
.empty-state p {{
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    color: {TEXT_MUTED};
    max-width: 440px;
    margin: 0;
    line-height: 1.55;
}}

/* -- ask console -- */
div[data-testid="stTextInput"] input {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    color: {TEXT};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    padding: 0.8rem 1.3rem;
    border-radius: 999px;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}}
div[data-testid="stTextInput"] input:focus {{
    border-color: {FUSION};
    box-shadow: 0 0 0 1px {FUSION}, 0 0 22px {GLOW};
}}
div[data-testid="stTextInput"] label {{ display: none; }}

.stButton button {{
    /* Dark fill with a glossy gradient ring, not a solid gradient fill -- the reference
    CTA's own treatment. */
    background:
        linear-gradient({SURFACE}, {SURFACE}) padding-box,
        {CTA_GRADIENT} border-box;
    border: 2px solid transparent;
    color: {TEXT};
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    border-radius: 999px;
    padding: 0.6rem 1.6rem;
    letter-spacing: 0.03em;
    box-shadow: 0 0 20px {GLOW};
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}}
.stButton button:hover {{
    box-shadow: 0 0 32px {GLOW};
    color: {TEXT};
    transform: translateY(-1px);
}}
.stButton button:focus:not(:active) {{ color: {TEXT}; }}

/* -- answer cards -- */
.answer-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 1.2rem 1.3rem;
    margin-bottom: 0.8rem;
}}
.answer-card.dense {{ border-top: 2px solid {DENSE}; }}
.answer-card.hybrid {{ border-top: 2px solid {FUSION}; }}
.answer-card.refused {{
    border-color: #ff6b6b55;
    background: linear-gradient(180deg, rgba(255,107,107,0.07), {SURFACE} 65%);
}}

.answer-card .channel-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    margin-bottom: 0.5rem;
}}
.answer-card.dense .channel-label {{ color: {DENSE}; }}
.answer-card.hybrid .channel-label {{ color: {FUSION}; }}

.answer-card .answer-text {{
    font-size: 0.94rem;
    line-height: 1.55;
    color: {TEXT};
    margin-bottom: 0.9rem;
    white-space: pre-wrap;
}}

.answer-card .refusal-kind {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: {DANGER};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
}}
.near-miss {{
    font-size: 0.82rem;
    color: {TEXT_MUTED};
    margin: 0.15rem 0;
}}
.near-miss .sim {{ color: {TEXT}; font-family: 'JetBrains Mono', monospace; }}

/* -- confidence stats: ring gauges, not bars -- */
.stat-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    margin-bottom: 0.6rem;
}}
.stat-grid.hero-row {{ grid-template-columns: 1fr; margin-bottom: 0.5rem; }}
.stat-tile {{
    background: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 0.5rem 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.55rem;
}}
.stat-tile.hero {{ border-color: {FUSION}55; padding: 0.7rem 0.9rem; }}
.stat-ring {{
    position: relative;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    flex-shrink: 0;
}}
.stat-tile.hero .stat-ring {{ width: 52px; height: 52px; }}
.stat-ring-hole {{
    position: absolute;
    inset: 5px;
    border-radius: 50%;
    background: {SURFACE_ALT};
    display: flex;
    align-items: center;
    justify-content: center;
}}
.stat-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 700;
    color: {TEXT};
}}
.stat-tile.hero .stat-value {{ font-size: 0.88rem; }}
.stat-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}
.stat-tile.hero .stat-label {{ font-size: 0.75rem; }}
.stat-empty {{
    color: {TEXT_MUTED};
    font-size: 0.78rem;
    margin-bottom: 0.6rem;
}}

/* -- citations & chunks -- */
.citation-tag {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: {TEXT};
    background: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    margin: 0 0.35rem 0.35rem 0;
}}
.citation-tag .n {{ color: {FUSION}; font-weight: 700; margin-right: 0.35rem; }}

.chunk-card {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.5rem;
    background: {SURFACE_ALT};
}}
.chunk-card .chunk-meta {{
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    margin-bottom: 0.3rem;
}}
.chunk-card .chunk-meta .score {{ color: {FUSION}; flex-shrink: 0; }}
.chunk-card .chunk-text {{
    font-size: 0.82rem;
    color: {TEXT_MUTED};
    line-height: 1.45;
}}

/* -- telemetry footer -- */
.telemetry {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
    padding-top: 0.5rem;
    margin-top: 0.3rem;
}}

/* -- document list -- */
.doc-row {{
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid {BORDER};
    color: {TEXT_MUTED};
}}
.doc-row .path {{ color: {TEXT}; }}
</style>
"""
