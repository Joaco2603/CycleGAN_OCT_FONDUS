"""
Scan dataset and filter out low-quality images.

Usage:
    # Preview what would be filtered (dry run)
    python scripts/filter_quality.py --root dataset/fundus/train --domain fundus --dry-run

    # Actually move bad images to quarantine folder
    python scripts/filter_quality.py --root dataset/fundus/train --domain fundus --quarantine

    # Use strict filtering
    python scripts/filter_quality.py --root dataset/fundus/train --domain fundus --strict --dry-run

    # Generate report with thumbnails
    python scripts/filter_quality.py --root dataset/fundus/train --domain fundus --report
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct import to avoid torch dependency
from src.data.quality_filter import CompositeFilter, filter_dataset


def main():
    parser = argparse.ArgumentParser(description="Filter low-quality images from dataset")
    parser.add_argument("--root", type=Path, required=True, help="Dataset directory to scan")
    parser.add_argument("--domain", choices=["fundus", "oct"], default="fundus", help="Image domain")
    parser.add_argument("--strict", action="store_true", help="Use stricter filtering")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be filtered")
    parser.add_argument("--quarantine", action="store_true", help="Move bad images to quarantine folder")
    parser.add_argument("--report", action="store_true", help="Generate HTML report with thumbnails")
    parser.add_argument("--quarantine-dir", type=Path, default=None, help="Quarantine directory")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"Directory not found: {args.root}")
        return

    # Select filter
    if args.domain == "fundus":
        qf = CompositeFilter.strict_fundus() if args.strict else CompositeFilter.default_fundus()
    else:
        qf = CompositeFilter.strict_oct() if args.strict else CompositeFilter.default_oct()

    print(f"Scanning: {args.root}")
    print(f"Domain: {args.domain}, Strict: {args.strict}")
    print("-" * 60)

    passed, rejected = filter_dataset(args.root, qf)

    print(f"\n✓ Passed: {len(passed)} images")
    print(f"✗ Rejected: {len(rejected)} images")

    if rejected:
        print("\n--- Rejected images ---")
        for path, reason in rejected[:20]:  # Show first 20
            rel_path = path.relative_to(args.root) if path.is_relative_to(args.root) else path
            print(f"  {rel_path}: {reason}")
        if len(rejected) > 20:
            print(f"  ... and {len(rejected) - 20} more")

    # Quarantine bad images
    if args.quarantine and rejected and not args.dry_run:
        q_dir = args.quarantine_dir or (args.root.parent / f"{args.root.name}_quarantine")
        q_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nMoving {len(rejected)} images to: {q_dir}")
        for path, reason in rejected:
            dest = q_dir / path.relative_to(args.root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
            # Also save reason
            (dest.parent / f"{dest.stem}_reason.txt").write_text(reason)

        print("Done!")

    # Generate HTML report
    if args.report and rejected:
        report_path = args.root.parent / f"{args.root.name}_quality_report.html"
        generate_html_report(rejected, passed, report_path, args.root)
        print(f"\nReport saved to: {report_path}")


def generate_html_report(
    rejected: list,
    passed: list,
    output_path: Path,
    root: Path,
) -> None:
    """Generate HTML report with thumbnail previews."""
    import base64
    from io import BytesIO
    from PIL import Image

    def img_to_base64(path: Path, size: int = 150) -> str:
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((size, size))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=70)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return ""

    html = [
        "<!DOCTYPE html><html><head>",
        "<title>Quality Filter Report</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 20px; }",
        ".grid { display: flex; flex-wrap: wrap; gap: 10px; }",
        ".card { border: 1px solid #ccc; padding: 8px; width: 180px; }",
        ".card.rejected { border-color: #f55; background: #fee; }",
        ".card img { width: 100%; height: 150px; object-fit: contain; background: #000; }",
        ".reason { font-size: 11px; color: #900; margin-top: 5px; }",
        ".name { font-size: 10px; color: #666; word-break: break-all; }",
        "</style></head><body>",
        f"<h1>Quality Report: {root.name}</h1>",
        f"<p>Passed: {len(passed)} | Rejected: {len(rejected)}</p>",
        "<h2>Rejected Images</h2><div class='grid'>",
    ]

    for path, reason in rejected[:50]:  # Limit to 50 thumbnails
        b64 = img_to_base64(path)
        rel = path.relative_to(root) if path.is_relative_to(root) else path.name
        html.append(f"""
        <div class='card rejected'>
            <img src='data:image/jpeg;base64,{b64}' alt='{rel}'>
            <div class='name'>{rel}</div>
            <div class='reason'>{reason}</div>
        </div>
        """)

    html.append("</div>")

    if len(rejected) > 50:
        html.append(f"<p>... and {len(rejected) - 50} more rejected images</p>")

    html.append("</body></html>")

    output_path.write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()
