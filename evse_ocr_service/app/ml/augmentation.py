import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import random

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


class DegradationSimulator:
    """
    Applies synthetic noise and optical degradation matching real-world outdoor EV charger conditions:
    1. Focus and motion blur (camera unsteadiness / lens unfocused)
    2. Overexposed and underexposed conditions (direct sunlight / dark night)
    3. UV fading and thermal ink washout (sun exposure)
    4. Hardware weathering, scratches, dirt, and peeling stickers
    5. Sensor noise in low-light camera environments
    """
    def __init__(self):
        if ALBUMENTATIONS_AVAILABLE:
            def _get_coarse_dropout():
                try:
                    return A.CoarseDropout(
                        num_holes_range=(2, 10),
                        hole_height_range=(4, 8),
                        hole_width_range=(4, 8),
                        fill=255,
                        p=0.3
                    )
                except (TypeError, ValueError):
                    return A.CoarseDropout(
                        max_holes=10,
                        max_height=8,
                        max_width=8,
                        min_holes=2,
                        fill_value=255,
                        p=0.3
                    )

            def _get_gauss_noise():
                try:
                    return A.GaussNoise(std_range=(0.08, 0.25), p=0.4)
                except (TypeError, ValueError):
                    return A.GaussNoise(var_limit=(20.0, 60.0), p=0.4)

            self.pipeline = A.Compose([
                # 1. Blur simulation
                A.OneOf([
                    A.MotionBlur(blur_limit=15, p=0.6),
                    A.GaussianBlur(blur_limit=(3, 11), p=0.4),
                    A.Defocus(radius=(3, 8), alias_blur=(0.1, 0.5), p=0.4)
                ], p=0.5),
                
                # 2. Exposure & Contrast variations
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.4, 0.4),
                    contrast_limit=(-0.3, 0.3),
                    p=0.6
                ),
                
                # 3. UV fading / sticker ink washout
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.2,
                    saturation=0.05,
                    hue=0.02,
                    p=0.5
                ),
                
                # 4. Physical weathering & scratches
                _get_coarse_dropout(),
                
                # 5. Sensor noise
                _get_gauss_noise()
            ])
        else:
            self.pipeline = None

    def apply(self, image: Image.Image) -> Image.Image:
        """Applies synthetic degradation pipeline to a PIL Image."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        if ALBUMENTATIONS_AVAILABLE and self.pipeline is not None:
            image_array = np.array(image)
            augmented = self.pipeline(image=image_array)
            return Image.fromarray(augmented['image'])
        else:
            # High-fidelity PIL fallback if Albumentations is not available
            img = image.copy()
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
            if random.random() < 0.6:
                brightness = ImageEnhance.Brightness(img)
                img = brightness.enhance(random.uniform(0.6, 1.4))
                contrast = ImageEnhance.Contrast(img)
                img = contrast.enhance(random.uniform(0.6, 1.3))
            if random.random() < 0.4:
                color = ImageEnhance.Color(img)
                img = color.enhance(random.uniform(0.1, 0.5))
            if random.random() < 0.3:
                # Add synthetic scratch lines
                draw = ImageDraw.Draw(img)
                w, h = img.size
                for _ in range(random.randint(2, 6)):
                    x1 = random.randint(0, w)
                    y1 = random.randint(0, h)
                    x2 = x1 + random.randint(-15, 15)
                    y2 = y1 + random.randint(-10, 10)
                    draw.line([(x1, y1), (x2, y2)], fill=(220, 220, 220), width=random.randint(1, 2))
            return img

