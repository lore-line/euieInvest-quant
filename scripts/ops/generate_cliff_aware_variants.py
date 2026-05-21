"""Generate per-variant JSON files for issue #25 cliff-aware sweep.

Translates 3-direction design matrices from the issue into the 4-regime
schema server team's harness expects (per scripts/backtest-crypto-dca-grid.py
load_multiplier_matrix() in companion repo). Writes to:
  data/server_research_labels/cliff_aware_variants/variant_*.json

Direction → regime_label translation:
    bull     → steady_bull
    sideways → sideways_range AND choppy_recovery (both)
    bear     → bear_trend
    (fallback) unknown → identity 1.0 [server-team-required row]

Files generated (8 total):
  - variant_a_hard_alpha.json   (mode:hard,  scale_safety_orders:false)  [HEADLINE]
  - variant_a_hard_beta.json    (mode:hard,  scale_safety_orders:true)
  - variant_a_soft_alpha.json   (mode:soft,  scale_safety_orders:false)
  - variant_a_soft_beta.json    (mode:soft,  scale_safety_orders:true)
  - variant_b_hard_alpha.json   (mode:hard,  scale_safety_orders:false)
  - variant_c_hard_alpha.json   (mode:hard,  scale_safety_orders:false)
  - variant_d_hard_alpha.json   (vol-only ablation)
  - variant_e_hard_alpha.json   (direction-only ablation)
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "data" / "server_research_labels" / "cliff_aware_variants"

# 3-direction design matrices from issue #25
DESIGN_3D = {
    "A": {
        "name": "Variant A — Symmetric Kelly (headline)",
        "description": "Theory-driven 3-step gradient. Bull/low = idle-capital compensation; Bear/high = cliff insurance.",
        "matrix_3d": {
            "bull":     {"low": 1.5, "mid": 1.2, "high": 0.8},
            "sideways": {"low": 1.3, "mid": 1.0, "high": 0.7},
            "bear":     {"low": 1.0, "mid": 0.7, "high": 0.5},
        },
    },
    "B": {
        "name": "Variant B — Aggressive Kelly (steeper)",
        "description": "Larger swings around baseline. Min 0.3x, max 2.0x. Tests sharper Kelly bets.",
        "matrix_3d": {
            "bull":     {"low": 2.0, "mid": 1.4, "high": 0.6},
            "sideways": {"low": 1.5, "mid": 1.0, "high": 0.5},
            "bear":     {"low": 1.0, "mid": 0.5, "high": 0.3},
        },
    },
    "C": {
        "name": "Variant C — Conservative Kelly (gentler)",
        "description": "Smaller swings around baseline. Min 0.7x, max 1.25x. Tests subtle modulation.",
        "matrix_3d": {
            "bull":     {"low": 1.25, "mid": 1.10, "high": 0.90},
            "sideways": {"low": 1.15, "mid": 1.00, "high": 0.85},
            "bear":     {"low": 1.00, "mid": 0.85, "high": 0.70},
        },
    },
    "D": {
        "name": "Variant D — Vol-only ablation (1D control)",
        "description": "Strip direction axis; multiplier depends only on vol tercile. Ablation control.",
        "matrix_3d": {
            "bull":     {"low": 1.5, "mid": 1.0, "high": 0.5},
            "sideways": {"low": 1.5, "mid": 1.0, "high": 0.5},
            "bear":     {"low": 1.5, "mid": 1.0, "high": 0.5},
        },
    },
    "E": {
        "name": "Variant E — Direction-only ablation (1D control)",
        "description": "Strip vol axis; multiplier depends only on regime direction. Ablation control.",
        "matrix_3d": {
            "bull":     {"low": 1.3, "mid": 1.3, "high": 1.3},
            "sideways": {"low": 1.0, "mid": 1.0, "high": 1.0},
            "bear":     {"low": 0.7, "mid": 0.7, "high": 0.7},
        },
    },
}


def to_4regime_matrix(m3d: dict) -> dict:
    """Translate 3-direction matrix to 4-regime + unknown fallback matrix.

    Convention:
      bull     → steady_bull
      sideways → sideways_range AND choppy_recovery
      bear     → bear_trend
      unknown  → identity 1.0  (required by harness loader)
    """
    return {
        "bear_trend":      m3d["bear"],
        "choppy_recovery": m3d["sideways"],
        "sideways_range":  m3d["sideways"],
        "steady_bull":     m3d["bull"],
        "unknown":         {"low": 1.0, "mid": 1.0, "high": 1.0},
    }


def make_variant(letter: str, mode: str, scale_safety_orders: bool) -> dict:
    """Build the harness-loader-compatible JSON dict for one variant config."""
    spec = DESIGN_3D[letter]
    label_suffix = []
    if mode == "soft":
        label_suffix.append("soft")
    label_suffix.append("β" if scale_safety_orders else "α")
    return {
        "name": f"{spec['name']} [{', '.join(label_suffix)}]",
        "description": spec["description"],
        "mode": mode,
        "scale_safety_orders": scale_safety_orders,
        "min_safety_orders": 4,
        "matrix": to_4regime_matrix(spec["matrix_3d"]),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Configs to generate.
    configs = [
        ("A", "hard", False),
        ("A", "hard", True),
        ("A", "soft", False),
        ("A", "soft", True),
        ("B", "hard", False),
        ("C", "hard", False),
        ("D", "hard", False),
        ("E", "hard", False),
    ]

    written = 0
    for letter, mode, beta in configs:
        suffix = "beta" if beta else "alpha"
        filename = f"variant_{letter.lower()}_{mode}_{suffix}.json"
        path = OUT_DIR / filename
        config = make_variant(letter, mode, beta)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        written += 1
        print(f"[ok] {filename}")

    print(f"\nGenerated {written} variant files in {OUT_DIR.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
