import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.ml.inference import get_inference_engine
from app.ml.augmentation import DegradationSimulator
from app.ml.training import EVSETrainingPipeline
from app.ml.synthetic_generator import SyntheticEVSEGenerator
from app.core.logging import logger

def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR model on EVSE ID dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./data/models/trocr_evse", help="Model output dir")
    args = parser.parse_args()

    logger.info("Initializing TrOCR Engine and Degradation Simulator...")
    engine = get_inference_engine()
    simulator = DegradationSimulator()
    
    # Generate synthetic training & eval sets for demonstration
    generator = SyntheticEVSEGenerator(simulator=simulator)
    train_manifest = generator.generate_dataset("./data/synthetic/train", count=40)
    eval_manifest = generator.generate_dataset("./data/synthetic/eval", count=10)
    
    pipeline = EVSETrainingPipeline(engine)
    
    try:
        logger.info("Tokenizing and augmenting dataset...")
        train_ds = pipeline.prepare_dataset(train_manifest, simulator)
        eval_ds = pipeline.prepare_dataset(eval_manifest, simulator)
        
        pipeline.execute_training_run(
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    except Exception as e:
        logger.error(f"Training pipeline error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
