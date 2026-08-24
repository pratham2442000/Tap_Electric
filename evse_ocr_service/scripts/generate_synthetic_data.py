import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.ml.synthetic_generator import SyntheticEVSEGenerator
from app.core.logging import logger

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic degraded EVSE ID training data")
    parser.add_argument("--count", type=int, default=50, help="Number of synthetic samples to create")
    parser.add_argument("--output_dir", type=str, default="./data/synthetic", help="Output directory")
    args = parser.parse_args()

    logger.info(f"Generating {args.count} synthetic samples in {args.output_dir}...")
    generator = SyntheticEVSEGenerator()
    manifest = generator.generate_dataset(output_dir=args.output_dir, count=args.count)
    logger.info(f"Generated {len(manifest)} synthetic pairs successfully.")

if __name__ == "__main__":
    main()
