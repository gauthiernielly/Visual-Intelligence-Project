"""
Wrap OpenTAD's eager imports of optional models in try/except so that missing
CUDA extensions, mmcv or mmaction do not break the ActionFormer import path.
Idempotent, safe to re-run.
"""

import argparse
from pathlib import Path


# (file_relative_to_opentad, single_line_to_replace, multi_line_replacement)
PATCHES = [
    ("opentad/models/roi_heads/roi_extractors/__init__.py",
     "from .roialign_extractor import ROIAlignExtractor",
     ('try:\n'
      '    from .roialign_extractor import ROIAlignExtractor\n'
      'except Exception as _e:\n'
      '    print(f"[opentad-patch] ROIAlignExtractor unavailable: {_e}")\n'
      '    ROIAlignExtractor = None')),
    ("opentad/models/roi_heads/roi_extractors/__init__.py",
     "from .gtad_extractor import GTADExtractor",
     ('try:\n'
      '    from .gtad_extractor import GTADExtractor\n'
      'except Exception as _e:\n'
      '    print(f"[opentad-patch] GTADExtractor unavailable: {_e}")\n'
      '    GTADExtractor = None')),
    ("opentad/models/detectors/__init__.py",
     "from .tadtr import TadTR",
     ('try:\n'
      '    from .tadtr import TadTR\n'
      'except Exception as _e:\n'
      '    print(f"[opentad-patch] TadTR unavailable: {_e}")\n'
      '    TadTR = None')),
    ("opentad/models/transformer/__init__.py",
     "from .tadtr_transformer import TadTRTransformer",
     ('try:\n'
      '    from .tadtr_transformer import TadTRTransformer\n'
      'except Exception as _e:\n'
      '    print(f"[opentad-patch] TadTRTransformer unavailable: {_e}")\n'
      '    TadTRTransformer = None')),

    # AFSD refinement head requires boundary_max_pooling_cuda.
    ("opentad/models/roi_heads/__init__.py",
     "from .afsd_roi_head import AFSDRefineHead",
     ('try:\n'
      '    from .afsd_roi_head import AFSDRefineHead\n'
      'except Exception as _e:\n'
      '    print(f"[opentad-patch] AFSDRefineHead unavailable: {_e}")\n'
      '    AFSDRefineHead = None')),

    # VSGN ROI head requires Align1D.
    ("opentad/models/roi_heads/__init__.py",
     "from .vsgn_roi_head import VSGNRoIHead",
     ('try:\n'
      '    from .vsgn_roi_head import VSGNRoIHead\n'
      'except Exception as _e:\n'
      '    print(f"[opentad-patch] VSGNRoIHead unavailable: {_e}")\n'
      '    VSGNRoIHead = None')),

    # AFSD detector requires boundary_max_pooling_cuda.
    ("opentad/models/detectors/__init__.py",
     "from .afsd import AFSD",
     ('try:\n'
      '    from .afsd import AFSD\n'
      'except Exception as _e:\n'
      '    print(f"[opentad-patch] AFSD unavailable: {_e}")\n'
      '    AFSD = None')),

    # AFSD coarse head requires boundary_max_pooling_cuda.
    ("opentad/models/dense_heads/__init__.py",
     "from .afsd_coarse_head import AFSDCoarseHead",
     ('try:\n'
      '    from .afsd_coarse_head import AFSDCoarseHead\n'
      'except Exception as _e:\n'
      '    print(f"[opentad-patch] AFSDCoarseHead unavailable: {_e}")\n'
      '    AFSDCoarseHead = None')),

    # AFSD neck requires boundary_max_pooling_cuda.
    ("opentad/models/necks/__init__.py",
     "from .afsd_neck import AFSDNeck",
     ('try:\n'
      '    from .afsd_neck import AFSDNeck\n'
      'except Exception as _e:\n'
      '    print(f"[opentad-patch] AFSDNeck unavailable: {_e}")\n'
      '    AFSDNeck = None')),
]

# Heavy backbones require mmcv or mmaction. Wrap each one individually.
HEAVY_BACKBONES = [
    ("from .re2tal_swin import SwinTransformer3D_inv",    "SwinTransformer3D_inv"),
    ("from .re2tal_slowfast import ResNet3dSlowFast_inv", "ResNet3dSlowFast_inv"),
    ("from .vit import VisionTransformerCP",              "VisionTransformerCP"),
    ("from .vit_adapter import VisionTransformerAdapter", "VisionTransformerAdapter"),
    ("from .vit_ladder import VisionTransformerLadder",   "VisionTransformerLadder"),
]
for line, name in HEAVY_BACKBONES:
    PATCHES.append((
        "opentad/models/backbones/__init__.py",
        line,
        ('try:\n'
         f'    {line}\n'
         f'except Exception as _e:\n'
         f'    print(f"[opentad-patch] {name} unavailable: {{_e}}")\n'
         f'    {name} = None')
    ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opentad", required=True, help="path to the OpenTAD checkout")
    args = p.parse_args()
    root = Path(args.opentad)
    if not root.exists():
        raise SystemExit(f"OpenTAD not found at {root}")

    for rel, old, new_block in PATCHES:
        path = root / rel
        if not path.exists():
            print(f"  [skip] missing file: {path}")
            continue
        txt = path.read_text()
        if new_block in txt:
            print(f"  [skip] already patched: {rel} :: {old[:50]}...")
            continue
        if old not in txt:
            print(f"  [warn] target line not found in {rel}: {old[:60]}")
            continue
        path.write_text(txt.replace(old, new_block))
        print(f"  [patched] {rel}")


if __name__ == "__main__":
    main()
