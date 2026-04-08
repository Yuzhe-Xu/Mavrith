from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "Manifest export requires PyYAML. Install mavrith[yaml] or add PyYAML to your environment."
    ) from exc

from mavrith import SimulationConfig, build_detail_manifest, write_manifest_bundle
from water_cooling import build_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export AI-oriented YAML manifests for the water_cooling example."
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / ".mavrith-ai" / "water_cooling"),
        help="Output directory for graph/detail manifests.",
    )
    parser.add_argument(
        "--initial-water-temperature",
        type=float,
        default=80.0,
        help="Initial water temperature used when building the example system.",
    )
    parser.add_argument(
        "--room-temperature",
        type=float,
        default=20.0,
        help="Ambient room temperature used when building the example system.",
    )
    parser.add_argument(
        "--cooling-rate",
        type=float,
        default=0.10,
        help="Cooling rate used when building the example system.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Simulation start time used for the runtime detail snapshot.",
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=30.0,
        help="Simulation stop time used for the runtime detail snapshot.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Simulation step size used for the runtime detail snapshot.",
    )
    return parser.parse_args()


def write_runtime_detail(system, out_dir: Path, config: SimulationConfig) -> Path:
    runtime_detail = build_detail_manifest(system, config=config)
    runtime_path = out_dir / "detail" / "system.runtime.yaml"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(
        yaml.safe_dump(runtime_detail, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
        newline="\n",
    )
    return runtime_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()

    system = build_system(
        initial_water_temperature=args.initial_water_temperature,
        room_temperature=args.room_temperature,
        cooling_rate=args.cooling_rate,
    )
    config = SimulationConfig(start=args.start, stop=args.stop, dt=args.dt)

    bundle = write_manifest_bundle(system, out_dir)
    runtime_path = write_runtime_detail(system, out_dir, config)

    print("exported water_cooling manifests")
    print(f"  graph      : {bundle.graph_path}")
    if bundle.detail_index_path is not None:
        print(f"  detail idx : {bundle.detail_index_path}")
    print(f"  detail cnt : {len(bundle.detail_paths)}")
    print(f"  runtime    : {runtime_path}")


if __name__ == "__main__":
    main()
