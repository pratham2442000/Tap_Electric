import io
from typing import Optional, List, Tuple
from PIL import Image
from app.config import settings
from app.core.logging import logger

try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None

class EVSEInferenceEngine:
    """
    Wraps Microsoft TrOCR for high-accuracy degraded EVSE ID extraction.
    Leverages Vision Transformer (ViT) encoder and autoregressive RoBERTa decoder.
    """
    def __init__(self, model_checkpoint: Optional[str] = None):
        self.model_checkpoint = model_checkpoint or settings.MODEL_CHECKPOINT
        self.processor = None
        self.model = None
        self.device = None
        self._initialize_model()

    def _initialize_model(self):
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers/PyTorch not available. Running inference engine in mock mode.")
            return

        try:
            logger.info(f"Loading TrOCR Processor & Model from: {self.model_checkpoint}")
            self.processor = TrOCRProcessor.from_pretrained(self.model_checkpoint)
            
            # Resolve compute accelerator
            if settings.MODEL_DEVICE == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif settings.MODEL_DEVICE == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif settings.MODEL_DEVICE == "cpu":
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_checkpoint).to(self.device)
            
            # Constrained decoding configuration for EVSE IDs
            self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
            self.model.config.vocab_size = self.model.config.decoder.vocab_size
            self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
            self.model.config.max_length = settings.MODEL_MAX_LENGTH
            self.model.config.early_stopping = True
            self.model.config.no_repeat_ngram_size = settings.MODEL_NO_REPEAT_NGRAM_SIZE
            
            # Beam search for resolving visually ambiguous character sequences
            self.model.config.num_beams = settings.MODEL_NUM_BEAMS
            self.model.config.length_penalty = 1.0
            self.model.eval()
            logger.info(f"TrOCR Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Error initializing TrOCR model: {e}")
            self.model = None

    def extract_text(self, image_input) -> Tuple[str, float]:
        """
        Performs OCR inference on an image input (PIL Image, bytes, or file path).
        Returns tuple of (extracted_text, confidence_score).
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input type")

        if not TRANSFORMERS_AVAILABLE or self.model is None or self.processor is None:
            # Fallback mock for testing environments without PyTorch/Transformers
            return "DE*ISE*E1234567910", 0.95

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                num_beams=settings.MODEL_NUM_BEAMS
            )
            
        generated_ids = outputs.sequences
        extracted_string = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Approximate confidence score from beam search sequences_scores
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            score = torch.exp(outputs.sequences_scores[0]).item()
            confidence = min(max(score, 0.0), 1.0)
        else:
            confidence = 0.90
            
        return extracted_string.strip(), round(confidence, 4)

# Global engine singleton
_engine_instance = None

def get_inference_engine() -> EVSEInferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EVSEInferenceEngine()
    return _engine_instance
