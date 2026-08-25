import os
from typing import List, Dict, Any
from PIL import Image
from app.ml.inference import EVSEInferenceEngine
from app.ml.augmentation import DegradationSimulator
from app.ml.metrics import calculate_cer, calculate_wer
from app.core.logging import logger

try:
    import torch
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
    from datasets import Dataset
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

class EVSETrainingPipeline:
    """
    Fine-tunes the TrOCR model using HuggingFace Seq2SeqTrainer on augmented datasets.
    Masks padding tokens with -100 to omit them from Cross-Entropy loss.
    Computes Character Error Rate (CER) during evaluation loops.
    """
    def __init__(self, model_engine: EVSEInferenceEngine):
        self.engine = model_engine

    def prepare_dataset(self, records: List[Dict[str, Any]], simulator: DegradationSimulator):
        if not TRAINING_AVAILABLE:
            raise RuntimeError("Hugging Face Datasets / Transformers not installed.")

        def preprocess_function(examples):
            images = [Image.open(path).convert("RGB") for path in examples['image_path']]
            augmented_images = [simulator.apply(img) for img in images]
            
            pixel_values = self.engine.processor(augmented_images, return_tensors="pt").pixel_values
            labels = self.engine.processor.tokenizer(
                examples['text'],
                padding="max_length",
                max_length=40,
                truncation=True
            ).input_ids
            
            # Mask padding token IDs with -100 for PyTorch CrossEntropyLoss
            labels = [
                [-100 if token == self.engine.processor.tokenizer.pad_token_id else token for token in label]
                for label in labels
            ]
            return {"pixel_values": pixel_values, "labels": labels}

        raw_dataset = Dataset.from_list(records)
        processed_dataset = raw_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["image_path", "text"]
        )
        return processed_dataset

    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        decoded_preds = self.engine.processor.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 in labels before decoding
        labels = [
            [token for token in label if token != -100]
            for label in labels
        ]
        decoded_labels = self.engine.processor.batch_decode(labels, skip_special_tokens=True)
        
        cer = calculate_cer(decoded_preds, decoded_labels)
        wer = calculate_wer(decoded_preds, decoded_labels)
        return {"cer": cer, "wer": wer}

    def execute_training_run(
        self,
        train_dataset,
        eval_dataset,
        output_dir: str = "./data/models/trocr_evse",
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ):
        if not TRAINING_AVAILABLE or self.engine.model is None:
            raise RuntimeError("Training dependencies or model not initialized.")

        # Support both modern eval_strategy and legacy evaluation_strategy
        eval_strategy_kwargs = {}
        try:
            Seq2SeqTrainingArguments(output_dir="tmp", eval_strategy="steps")
            eval_strategy_kwargs["eval_strategy"] = "steps"
        except (TypeError, ValueError):
            eval_strategy_kwargs["evaluation_strategy"] = "steps"

        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            fp16=torch.cuda.is_available(),
            output_dir=output_dir,
            logging_steps=20,
            save_steps=100,
            eval_steps=100,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            save_total_limit=2,
            **eval_strategy_kwargs
        )


        trainer = Seq2SeqTrainer(
            model=self.engine.model,
            tokenizer=self.engine.processor.feature_extractor,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
        )

        logger.info("Starting TrOCR Fine-Tuning execution...")
        trainer.train()
        
        prod_path = os.path.join(output_dir, "production_model")
        self.engine.model.save_pretrained(prod_path)
        self.engine.processor.save_pretrained(prod_path)
        logger.info(f"Saved production model to {prod_path}")
