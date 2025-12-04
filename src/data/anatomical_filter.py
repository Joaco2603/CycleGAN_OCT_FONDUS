"""
Anatomical quality filters for fundus images.

Detects presence and quality of key anatomical structures:
- Optic disc (papila)
- Blood vessels (radial pattern)
- Macula (dark spot, avascular zone)
- Retinal background quality

Filters out images lacking proper anatomical features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

try:
    from scipy import ndimage
    from scipy.ndimage import label, binary_dilation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class AnatomyResult:
    """Result of anatomical analysis."""
    passed: bool
    reason: str
    scores: Dict[str, float]
    details: Dict[str, str]


# =============================================================================
# 1. OPTIC DISC DETECTOR
# =============================================================================
class OpticDiscDetector:
    """
    Detect optic disc (bright yellowish circular region).
    
    Characteristics:
    - High luminosity region
    - Rounded shape
    - Yellow-red dominant color
    - Located nasally
    """
    
    def __init__(
        self,
        min_brightness: float = 0.6,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.08,
        min_circularity: float = 0.4,
    ):
        self.min_brightness = min_brightness
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_circularity = min_circularity

    def detect(self, img: Image.Image) -> Tuple[bool, float, str]:
        """
        Detect optic disc presence.
        
        Returns: (found, confidence, detail)
        """
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        h, w = arr.shape[:2]
        total_pixels = h * w
        
        # Extract channels
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        
        # Luminosity (weighted)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Yellow-ish detection: high R, high G, lower B
        yellow_mask = (r > 0.5) & (g > 0.4) & (r > b * 1.2) & (lum > self.min_brightness)
        
        # Bright region detection
        bright_mask = lum > self.min_brightness
        
        # Combine: bright AND yellowish
        disc_candidate = yellow_mask & bright_mask
        
        # Check area
        disc_area = disc_candidate.sum()
        area_ratio = disc_area / total_pixels
        
        if area_ratio < self.min_area_ratio:
            return False, area_ratio, f"No bright disc region found (area={area_ratio:.3%})"
        
        if area_ratio > self.max_area_ratio:
            # Could be overexposed
            return False, area_ratio, f"Overexposed or too large bright area ({area_ratio:.1%})"
        
        # Check if the bright region is somewhat circular (using bounding box ratio)
        if HAS_SCIPY:
            labeled, num_features = label(disc_candidate)
            if num_features > 0:
                # Find largest connected component
                sizes = ndimage.sum(disc_candidate, labeled, range(1, num_features + 1))
                largest_idx = np.argmax(sizes) + 1
                largest_mask = labeled == largest_idx
                
                # Bounding box circularity check
                rows = np.any(largest_mask, axis=1)
                cols = np.any(largest_mask, axis=0)
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    height = rmax - rmin + 1
                    width = cmax - cmin + 1
                    aspect = min(height, width) / max(height, width) if max(height, width) > 0 else 0
                    
                    if aspect < self.min_circularity:
                        return False, aspect, f"Disc shape not circular (aspect={aspect:.2f})"
        
        return True, area_ratio, f"Optic disc detected (area={area_ratio:.2%})"


# =============================================================================
# 2. BLOOD VESSEL DETECTOR
# =============================================================================
class VesselDetector:
    """
    Detect radial blood vessel pattern.
    
    Characteristics:
    - Dark lines on lighter background
    - Branching pattern
    - Radial from optic disc
    - High local contrast
    """
    
    def __init__(
        self,
        min_vessel_ratio: float = 0.02,
        edge_threshold: float = 0.08,
    ):
        self.min_vessel_ratio = min_vessel_ratio
        self.edge_threshold = edge_threshold

    def detect(self, img: Image.Image) -> Tuple[bool, float, str]:
        """
        Detect blood vessel presence using edge detection.
        
        Returns: (found, vessel_score, detail)
        """
        # Convert to green channel (best vessel contrast)
        arr = np.array(img.convert("RGB"))
        green = arr[:, :, 1].astype(np.float32) / 255.0
        
        # Apply edge detection
        img_pil = Image.fromarray((green * 255).astype(np.uint8))
        edges = img_pil.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges).astype(np.float32) / 255.0
        
        # Calculate edge density (vessel indicator)
        strong_edges = edge_arr > self.edge_threshold
        edge_ratio = strong_edges.sum() / edge_arr.size
        
        # Also check for linear structures using variance in local windows
        if HAS_SCIPY:
            # Sobel-like gradient
            sobel_x = ndimage.sobel(green, axis=1)
            sobel_y = ndimage.sobel(green, axis=0)
            gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            gradient_score = (gradient_mag > 0.05).sum() / gradient_mag.size
        else:
            gradient_score = edge_ratio
        
        combined_score = (edge_ratio + gradient_score) / 2
        
        if combined_score < self.min_vessel_ratio:
            return False, combined_score, f"Insufficient vessel structure ({combined_score:.2%})"
        
        return True, combined_score, f"Vessels detected (score={combined_score:.2%})"


# =============================================================================
# 3. MACULA DETECTOR
# =============================================================================
class MaculaDetector:
    """
    Detect macula (darker region, avascular).
    
    Characteristics:
    - Darker than background
    - Circular/elliptical shape
    - Located temporally (opposite to disc)
    - No large vessels inside
    """
    
    def __init__(
        self,
        dark_threshold: float = 0.35,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.15,
    ):
        self.dark_threshold = dark_threshold
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def detect(self, img: Image.Image) -> Tuple[bool, float, str]:
        """
        Detect macula presence.
        
        Returns: (found, confidence, detail)
        """
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        h, w = arr.shape[:2]
        
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Create mask for non-black areas (exclude background)
        non_black = lum > 0.05
        
        if non_black.sum() == 0:
            return False, 0.0, "Image too dark to detect macula"
        
        # Calculate mean luminosity of visible area
        mean_lum = lum[non_black].mean()
        
        # Macula should be darker than average
        dark_mask = (lum < mean_lum * 0.7) & non_black
        
        # Also check for reddish-brown color
        macula_color = dark_mask & (r > b) & (r > g * 0.8)
        
        dark_ratio = dark_mask.sum() / non_black.sum()
        
        if dark_ratio < self.min_area_ratio:
            return False, dark_ratio, f"No distinct macula region ({dark_ratio:.2%})"
        
        if dark_ratio > self.max_area_ratio:
            # Could be underexposed
            return False, dark_ratio, f"Too much dark area, may be underexposed ({dark_ratio:.1%})"
        
        return True, dark_ratio, f"Macula region detected ({dark_ratio:.2%})"


# =============================================================================
# 4. RETINAL BACKGROUND ANALYZER
# =============================================================================
class BackgroundAnalyzer:
    """
    Analyze retinal background quality.
    
    Good background has:
    - Smooth reddish/yellowish texture
    - No harsh artifacts
    - R > G > B color relationship
    - Not completely black or white
    """
    
    def __init__(
        self,
        min_uniformity: float = 0.3,
        min_color_ratio: float = 0.6,
    ):
        self.min_uniformity = min_uniformity
        self.min_color_ratio = min_color_ratio

    def analyze(self, img: Image.Image) -> Tuple[bool, float, str]:
        """
        Analyze background quality.
        
        Returns: (good_quality, score, detail)
        """
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        
        # Check for typical fundus color: R > G > B
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Non-black mask
        visible = lum > 0.05
        if visible.sum() < arr.shape[0] * arr.shape[1] * 0.1:
            return False, 0.0, "Less than 10% visible content"
        
        # Check R > G > B relationship in visible areas
        r_vis, g_vis, b_vis = r[visible], g[visible], b[visible]
        
        proper_color = ((r_vis > g_vis) | (np.abs(r_vis - g_vis) < 0.1)).mean()
        proper_color2 = (g_vis >= b_vis).mean()
        color_score = (proper_color + proper_color2) / 2
        
        if color_score < self.min_color_ratio:
            return False, color_score, f"Unnatural color distribution ({color_score:.1%} proper)"
        
        # Check for smooth texture (low local variance)
        if HAS_SCIPY:
            # Calculate local standard deviation
            from scipy.ndimage import uniform_filter
            mean_local = uniform_filter(lum, size=15)
            sq_local = uniform_filter(lum**2, size=15)
            local_std = np.sqrt(np.maximum(sq_local - mean_local**2, 0))
            
            # In visible areas, check smoothness
            smoothness = 1.0 - np.clip(local_std[visible].mean() * 5, 0, 1)
        else:
            smoothness = 0.5  # Assume OK if scipy not available
        
        combined = (color_score + smoothness) / 2
        
        if combined < self.min_uniformity:
            return False, combined, f"Poor background quality ({combined:.1%})"
        
        return True, combined, f"Good background (score={combined:.1%})"


# =============================================================================
# 5. EXPOSURE & SHARPNESS CHECKS
# =============================================================================
class ExposureChecker:
    """Check for proper exposure (not too dark/bright)."""
    
    def __init__(
        self,
        min_percentile_10: float = 0.05,
        max_percentile_90: float = 0.95,
        ideal_range: Tuple[float, float] = (0.15, 0.75),
    ):
        self.min_p10 = min_percentile_10
        self.max_p90 = max_percentile_90
        self.ideal_range = ideal_range

    def check(self, img: Image.Image) -> Tuple[bool, float, str]:
        """Check exposure quality."""
        arr = np.array(img.convert("L")).astype(np.float32) / 255.0
        
        # Exclude pure black (background)
        visible = arr > 0.02
        if visible.sum() < arr.size * 0.1:
            return False, 0.0, "Image mostly black"
        
        vis_pixels = arr[visible]
        
        p10 = np.percentile(vis_pixels, 10)
        p90 = np.percentile(vis_pixels, 90)
        median = np.median(vis_pixels)
        
        # Check dynamic range
        dynamic_range = p90 - p10
        
        if dynamic_range < 0.15:
            return False, dynamic_range, f"Low dynamic range ({dynamic_range:.2f})"
        
        if median < self.ideal_range[0]:
            return False, median, f"Underexposed (median={median:.2f})"
        
        if median > self.ideal_range[1]:
            return False, median, f"Overexposed (median={median:.2f})"
        
        score = min(1.0, dynamic_range / 0.5)
        return True, score, f"Good exposure (range={dynamic_range:.2f}, median={median:.2f})"


class SharpnessChecker:
    """Check image sharpness using Laplacian variance."""
    
    def __init__(self, min_sharpness: float = 80.0):
        self.min_sharpness = min_sharpness

    def check(self, img: Image.Image) -> Tuple[bool, float, str]:
        """Check sharpness using edge detection variance."""
        gray = np.array(img.convert("L")).astype(np.float32)
        
        # Simple Laplacian using PIL
        edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges).astype(np.float32)
        
        # Variance of edges indicates sharpness
        sharpness = edge_arr.var()
        
        if sharpness < self.min_sharpness:
            return False, sharpness, f"Image too blurry (sharpness={sharpness:.1f})"
        
        return True, sharpness, f"Adequate sharpness ({sharpness:.1f})"


class ArtifactDetector:
    """Detect common artifacts: reflections, vignetting, unnatural patterns."""
    
    def __init__(
        self,
        max_reflection_ratio: float = 0.03,
        max_vignette_score: float = 0.4,
    ):
        self.max_reflection_ratio = max_reflection_ratio
        self.max_vignette_score = max_vignette_score

    def detect(self, img: Image.Image) -> Tuple[bool, float, str]:
        """Detect artifacts."""
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        h, w = arr.shape[:2]
        
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        
        # 1. Check for strong reflections (very bright spots)
        very_bright = lum > 0.95
        reflection_ratio = very_bright.sum() / lum.size
        
        if reflection_ratio > self.max_reflection_ratio:
            return False, reflection_ratio, f"Strong reflections detected ({reflection_ratio:.1%})"
        
        # 2. Check for extreme vignetting
        # Compare center brightness vs edge brightness
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        
        y_grid, x_grid = np.ogrid[:h, :w]
        center_mask = ((y_grid - center_y)**2 + (x_grid - center_x)**2) < radius**2
        edge_mask = ((y_grid - center_y)**2 + (x_grid - center_x)**2) > (radius * 1.5)**2
        
        # Exclude black background from edge calculation
        edge_mask = edge_mask & (lum > 0.05)
        
        if center_mask.sum() > 0 and edge_mask.sum() > 100:
            center_brightness = lum[center_mask].mean()
            edge_brightness = lum[edge_mask].mean()
            
            if center_brightness > 0:
                vignette_score = abs(center_brightness - edge_brightness) / center_brightness
                
                # Extreme vignette: edges much darker OR brighter than center
                if edge_brightness > center_brightness * 1.5:
                    return False, vignette_score, f"Inverted vignette (bright edges)"
        
        return True, 0.0, "No significant artifacts"


# =============================================================================
# COMPOSITE ANATOMICAL FILTER
# =============================================================================
class AnatomicalFilter:
    """
    Complete anatomical quality filter for fundus images.
    
    Combines all detectors to ensure image has proper:
    - Optic disc
    - Blood vessels
    - Macula (optional, harder to detect)
    - Background quality
    - Exposure
    - Sharpness
    - No artifacts
    """
    
    def __init__(
        self,
        require_disc: bool = True,
        require_vessels: bool = True,
        require_macula: bool = False,  # Optional, harder to detect
        check_exposure: bool = True,
        check_sharpness: bool = True,
        check_artifacts: bool = True,
        check_background: bool = True,
        strict: bool = False,
    ):
        self.require_disc = require_disc
        self.require_vessels = require_vessels
        self.require_macula = require_macula
        self.check_exposure = check_exposure
        self.check_sharpness = check_sharpness
        self.check_artifacts = check_artifacts
        self.check_background = check_background
        
        # Initialize detectors with appropriate thresholds
        if strict:
            self.disc_detector = OpticDiscDetector(min_brightness=0.55, min_area_ratio=0.008)
            self.vessel_detector = VesselDetector(min_vessel_ratio=0.03)
            self.macula_detector = MaculaDetector(min_area_ratio=0.015)
            self.background_analyzer = BackgroundAnalyzer(min_uniformity=0.4)
            self.exposure_checker = ExposureChecker(ideal_range=(0.20, 0.70))
            self.sharpness_checker = SharpnessChecker(min_sharpness=120.0)
            self.artifact_detector = ArtifactDetector(max_reflection_ratio=0.02)
        else:
            self.disc_detector = OpticDiscDetector()
            self.vessel_detector = VesselDetector()
            self.macula_detector = MaculaDetector()
            self.background_analyzer = BackgroundAnalyzer()
            self.exposure_checker = ExposureChecker()
            self.sharpness_checker = SharpnessChecker()
            self.artifact_detector = ArtifactDetector()

    def analyze(self, img: Image.Image) -> AnatomyResult:
        """
        Perform complete anatomical analysis.
        
        Returns AnatomyResult with pass/fail and detailed scores.
        """
        scores = {}
        details = {}
        
        # 1. Exposure check (do first, affects everything else)
        if self.check_exposure:
            passed, score, detail = self.exposure_checker.check(img)
            scores["exposure"] = score
            details["exposure"] = detail
            if not passed:
                return AnatomyResult(False, f"[exposure] {detail}", scores, details)
        
        # 2. Sharpness check
        if self.check_sharpness:
            passed, score, detail = self.sharpness_checker.check(img)
            scores["sharpness"] = score
            details["sharpness"] = detail
            if not passed:
                return AnatomyResult(False, f"[sharpness] {detail}", scores, details)
        
        # 3. Artifact detection
        if self.check_artifacts:
            passed, score, detail = self.artifact_detector.detect(img)
            scores["artifacts"] = score
            details["artifacts"] = detail
            if not passed:
                return AnatomyResult(False, f"[artifacts] {detail}", scores, details)
        
        # 4. Optic disc detection
        if self.require_disc:
            passed, score, detail = self.disc_detector.detect(img)
            scores["optic_disc"] = score
            details["optic_disc"] = detail
            if not passed:
                return AnatomyResult(False, f"[optic_disc] {detail}", scores, details)
        
        # 5. Blood vessel detection
        if self.require_vessels:
            passed, score, detail = self.vessel_detector.detect(img)
            scores["vessels"] = score
            details["vessels"] = detail
            if not passed:
                return AnatomyResult(False, f"[vessels] {detail}", scores, details)
        
        # 6. Macula detection (optional)
        if self.require_macula:
            passed, score, detail = self.macula_detector.detect(img)
            scores["macula"] = score
            details["macula"] = detail
            if not passed:
                return AnatomyResult(False, f"[macula] {detail}", scores, details)
        
        # 7. Background quality
        if self.check_background:
            passed, score, detail = self.background_analyzer.analyze(img)
            scores["background"] = score
            details["background"] = detail
            if not passed:
                return AnatomyResult(False, f"[background] {detail}", scores, details)
        
        return AnatomyResult(True, "All anatomical checks passed", scores, details)


def filter_fundus_anatomical(
    root: Path,
    strict: bool = False,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif"),
) -> Tuple[List[Path], List[Tuple[Path, str, Dict[str, float]]]]:
    """
    Filter fundus images by anatomical quality.
    
    Returns:
        (passed_images, rejected_with_reasons_and_scores)
    """
    root = Path(root)
    all_images = sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)
    
    anat_filter = AnatomicalFilter(strict=strict)
    
    passed = []
    rejected = []
    
    for img_path in all_images:
        try:
            img = Image.open(img_path).convert("RGB")
            result = anat_filter.analyze(img)
            if result.passed:
                passed.append(img_path)
            else:
                rejected.append((img_path, result.reason, result.scores))
        except Exception as e:
            rejected.append((img_path, f"Error: {e}", {}))
    
    return passed, rejected
