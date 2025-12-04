"""
Filter fundus images by anatomical quality.

Detects and validates:
- Optic disc (bright yellowish region)
- Blood vessels (radial pattern)
- Macula (dark avascular region)
- Proper exposure and sharpness
- Absence of artifacts

Usage:
    # Dry run (preview)
    python scripts/filter_anatomical.py --root dataset/fundus/train --dry-run

    # Strict mode
    python scripts/filter_anatomical.py --root dataset/fundus/train --strict --dry-run

    # Move bad images to quarantine
    python scripts/filter_anatomical.py --root dataset/fundus/train --quarantine

    # Generate HTML report
    python scripts/filter_anatomical.py --root dataset/fundus/train --report
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.anatomical_filter import AnatomicalFilter, filter_fundus_anatomical


def main():
    parser = argparse.ArgumentParser(description="Filter fundus images by anatomical quality")
    parser.add_argument("--root", type=Path, required=True, help="Directory to scan")
    parser.add_argument("--strict", action="store_true", help="Use stricter thresholds")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't move files")
    parser.add_argument("--quarantine", action="store_true", help="Move rejected to quarantine")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--quarantine-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.root.exists():
        print(f"Directory not found: {args.root}")
        return

    print(f"🔬 Anatomical Filter - Fundus Images")
    print(f"   Directory: {args.root}")
    print(f"   Mode: {'STRICT' if args.strict else 'NORMAL'}")
    print("-" * 60)
    print("Checking for:")
    print("   ✓ Optic disc (bright yellow region)")
    print("   ✓ Blood vessels (radial pattern)")
    print("   ✓ Proper exposure (not too dark/bright)")
    print("   ✓ Sharpness (not blurry)")
    print("   ✓ No artifacts (reflections, vignette)")
    print("   ✓ Good background quality")
    print("-" * 60)

    passed, rejected = filter_fundus_anatomical(args.root, strict=args.strict)

    print(f"\n✅ Passed: {len(passed)} images")
    print(f"❌ Rejected: {len(rejected)} images")

    if rejected:
        # Group by rejection reason
        reasons: Dict[str, int] = {}
        for _, reason, _ in rejected:
            category = reason.split("]")[0] + "]" if "]" in reason else reason
            reasons[category] = reasons.get(category, 0) + 1

        print("\n📊 Rejection summary:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"   {reason}: {count}")

        print("\n--- Sample rejected images ---")
        for path, reason, scores in rejected[:15]:
            rel = path.relative_to(args.root) if path.is_relative_to(args.root) else path.name
            print(f"   {rel}")
            print(f"      → {reason}")
        if len(rejected) > 15:
            print(f"   ... and {len(rejected) - 15} more")

    # Quarantine
    if args.quarantine and rejected and not args.dry_run:
        q_dir = args.quarantine_dir or (args.root.parent / f"{args.root.name}_anatomical_quarantine")
        q_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📦 Moving {len(rejected)} images to: {q_dir}")
        for path, reason, scores in rejected:
            dest = q_dir / path.relative_to(args.root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
            
            # Save reason and scores
            info = f"Reason: {reason}\n\nScores:\n"
            for k, v in scores.items():
                info += f"  {k}: {v:.4f}\n"
            (dest.parent / f"{dest.stem}_info.txt").write_text(info)

        print("✅ Done!")

    # HTML Report
    if args.report and rejected:
        report_path = args.root.parent / f"{args.root.name}_anatomical_report.html"
        generate_report(passed, rejected, report_path, args.root)
        print(f"\n📄 Report: {report_path}")


def generate_report(
    passed: List[Path],
    rejected: List[Tuple[Path, str, Dict[str, float]]],
    output: Path,
    root: Path,
) -> None:
    """Generate detailed HTML report with anatomy scores."""
    import base64
    from io import BytesIO
    from PIL import Image

    def to_b64(path: Path, size: int = 180) -> str:
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((size, size))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=75)
            return base64.b64encode(buf.getvalue()).decode()
        except:
            return ""

    def score_bar(value: float, max_val: float = 1.0) -> str:
        pct = min(100, max(0, (value / max_val) * 100))
        color = "#4caf50" if pct > 50 else "#ff9800" if pct > 25 else "#f44336"
        return f'<div style="background:#eee;height:8px;width:100px;border-radius:4px;"><div style="background:{color};height:8px;width:{pct}%;border-radius:4px;"></div></div>'

    html = [
        "<!DOCTYPE html><html><head>",
        "<title>Anatomical Quality Report</title>",
        "<style>",
        "body{font-family:system-ui;margin:20px;background:#f5f5f5;}",
        "h1{color:#333;} h2{color:#666;margin-top:30px;}",
        ".summary{background:#fff;padding:20px;border-radius:8px;margin-bottom:20px;}",
        ".grid{display:flex;flex-wrap:wrap;gap:15px;}",
        ".card{background:#fff;border-radius:8px;padding:12px;width:220px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}",
        ".card.fail{border-left:4px solid #f44336;}",
        ".card img{width:100%;height:150px;object-fit:contain;background:#000;border-radius:4px;}",
        ".reason{color:#d32f2f;font-size:12px;margin:8px 0;font-weight:500;}",
        ".scores{font-size:11px;color:#666;}",
        ".scores div{display:flex;justify-content:space-between;align-items:center;margin:3px 0;}",
        ".name{font-size:10px;color:#999;word-break:break-all;}",
        "</style></head><body>",
        f"<h1>🔬 Anatomical Quality Report</h1>",
        f"<div class='summary'>",
        f"<strong>Directory:</strong> {root}<br>",
        f"<strong>Passed:</strong> {len(passed)} | <strong>Rejected:</strong> {len(rejected)}<br>",
        f"<strong>Pass rate:</strong> {len(passed)/(len(passed)+len(rejected))*100:.1f}%",
        f"</div>",
        "<h2>❌ Rejected Images</h2><div class='grid'>",
    ]

    for path, reason, scores in rejected[:60]:
        b64 = to_b64(path)
        rel = path.relative_to(root) if path.is_relative_to(root) else path.name
        
        scores_html = ""
        for k, v in scores.items():
            if isinstance(v, float):
                scores_html += f"<div><span>{k}</span>{score_bar(v)}</div>"
        
        html.append(f"""
        <div class='card fail'>
            <img src='data:image/jpeg;base64,{b64}'>
            <div class='name'>{rel}</div>
            <div class='reason'>{reason}</div>
            <div class='scores'>{scores_html}</div>
        </div>
        """)

    html.append("</div>")
    
    if len(rejected) > 60:
        html.append(f"<p>... and {len(rejected) - 60} more</p>")

    # Show some passed examples
    html.append("<h2>✅ Passed Examples</h2><div class='grid'>")
    import random
    samples = random.sample(passed, min(12, len(passed)))
    for path in samples:
        b64 = to_b64(path)
        rel = path.relative_to(root) if path.is_relative_to(root) else path.name
        html.append(f"""
        <div class='card'>
            <img src='data:image/jpeg;base64,{b64}'>
            <div class='name'>{rel}</div>
        </div>
        """)
    html.append("</div></body></html>")

    output.write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()
