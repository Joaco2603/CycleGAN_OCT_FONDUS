"""
DVC Adapter for dataset versioning.

Tracks which dataset version was used for each training run.
Keeps dataset management separate from training logic.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class DatasetInfo:
    """Dataset metadata for tracking."""
    
    # Paths
    fundus_path: str
    oct_path: str
    
    # Counts
    fundus_train_count: int = 0
    fundus_val_count: int = 0
    oct_train_count: int = 0
    oct_val_count: int = 0
    
    # Version info
    dvc_hash: Optional[str] = None
    git_commit: Optional[str] = None
    timestamp: Optional[str] = None
    
    # Optional metadata
    preprocessing: Optional[str] = None
    image_size: Optional[int] = None
    notes: Optional[str] = None
    
    def total_images(self) -> int:
        return (
            self.fundus_train_count + self.fundus_val_count +
            self.oct_train_count + self.oct_val_count
        )
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_images"] = self.total_images()
        return d


class DVCAdapter:
    """
    Adapter for DVC dataset versioning.
    
    Usage:
        adapter = DVCAdapter(dataset_root="dataset")
        info = adapter.get_dataset_info()
        adapter.track(info, run_name="cyclegan_20241210")
    """
    
    def __init__(self, dataset_root: str = "dataset"):
        self.dataset_root = Path(dataset_root)
    
    def get_dataset_info(
        self,
        fundus_subdir: str = "fundus",
        oct_subdir: str = "oct",
        preprocessing: Optional[str] = None,
        image_size: Optional[int] = None,
    ) -> DatasetInfo:
        """Extract dataset info from directory structure."""
        fundus_path = self.dataset_root / fundus_subdir
        oct_path = self.dataset_root / oct_subdir
        
        return DatasetInfo(
            fundus_path=str(fundus_path),
            oct_path=str(oct_path),
            fundus_train_count=self._count_images(fundus_path / "train"),
            fundus_val_count=self._count_images(fundus_path / "val"),
            oct_train_count=self._count_images(oct_path / "train"),
            oct_val_count=self._count_images(oct_path / "val"),
            dvc_hash=self._get_dvc_hash(),
            git_commit=self._get_git_commit(),
            timestamp=datetime.now().isoformat(),
            preprocessing=preprocessing,
            image_size=image_size,
        )
    
    def _count_images(self, path: Path) -> int:
        """Count images in directory (recursive)."""
        if not path.exists():
            return 0
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        count = 0
        for f in path.rglob("*"):
            if f.suffix.lower() in extensions:
                count += 1
        return count
    
    def _get_dvc_hash(self) -> Optional[str]:
        """Get DVC dataset hash if available."""
        dvc_file = self.dataset_root.with_suffix(".dvc")
        if not dvc_file.exists():
            dvc_file = self.dataset_root / ".dvc"
        
        if dvc_file.exists():
            content = dvc_file.read_text()
            # Extract md5 hash from DVC file
            for line in content.split("\n"):
                if "md5:" in line:
                    return line.split("md5:")[-1].strip()
        return None
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()[:8]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def track(
        self,
        info: DatasetInfo,
        run_name: str,
        output_dir: str = "logs/dataset_versions",
    ) -> Path:
        """
        Save dataset info as JSON for the training run.
        
        Returns path to saved JSON file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{run_name}_dataset.json"
        filepath = output_path / filename
        
        with open(filepath, "w") as f:
            json.dump(info.to_dict(), f, indent=2)
        
        return filepath
    
    def init_dvc(self) -> bool:
        """Initialize DVC in repository if not already."""
        try:
            subprocess.run(["dvc", "version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("DVC not installed. Run: pip install dvc")
            return False
        
        dvc_dir = Path(".dvc")
        if dvc_dir.exists():
            return True
        
        try:
            subprocess.run(["dvc", "init"], check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def add_dataset(self) -> bool:
        """Add dataset folder to DVC tracking."""
        try:
            subprocess.run(
                ["dvc", "add", str(self.dataset_root)],
                check=True
            )
            print(f"Dataset tracked: {self.dataset_root}")
            print(f"Created: {self.dataset_root}.dvc")
            print("Run 'git add' to commit the .dvc file")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"DVC add failed: {e}")
            return False


def track_dataset(
    run_name: str,
    dataset_root: str = "dataset",
    preprocessing: Optional[str] = None,
    image_size: Optional[int] = None,
) -> DatasetInfo:
    """
    Convenience function to track dataset for a training run.
    
    Usage:
        info = track_dataset("cyclegan_20241210", preprocessing="standard")
    """
    adapter = DVCAdapter(dataset_root=dataset_root)
    info = adapter.get_dataset_info(
        preprocessing=preprocessing,
        image_size=image_size,
    )
    adapter.track(info, run_name=run_name)
    return info
