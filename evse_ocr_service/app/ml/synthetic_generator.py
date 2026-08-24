import os
import random
import string
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont
from app.ml.augmentation import DegradationSimulator

# Standard charging network operator codes in Europe
SAMPLE_OPERATORS = ["TNM", "ISE", "FST", "ALL", "ION", "ENB", "IBD", "EON", "VTT", "NRG"]
COUNTRY_CODES = ["DE", "NL", "FR", "UK", "BE", "AT", "CH", "SE", "NO", "IT", "ES"]

class SyntheticEVSEGenerator:
    """
    Generates synthetic EVSE ID sticker labels with realistic typography,
    standard layouts, QR codes, and applies simulated environmental degradation.
    """
    def __init__(self, simulator: Optional[DegradationSimulator] = None):
        self.simulator = simulator or DegradationSimulator()

    def generate_random_evse_id(self, standard: str = "ISO_15118") -> str:
        country = random.choice(COUNTRY_CODES)
        operator = random.choice(SAMPLE_OPERATORS)
        outlet_digits = "".join(random.choices(string.digits, k=random.randint(6, 12)))
        
        if standard == "ISO_15118":
            delim = "*" if random.random() < 0.7 else ""
            return f"{country}{delim}{operator}{delim}E{outlet_digits}"
        else:  # DIN_91286 / EVCO
            prefix = "+49" if random.random() < 0.5 else country
            delim = "*" if random.random() < 0.8 else "-"
            return f"{prefix}{delim}{operator}{delim}{outlet_digits}"

    def render_sticker_image(self, evse_id: str, width: int = 400, height: int = 150) -> Image.Image:
        # Create white/silver sticker background with subtle gradient/noise
        bg_color = random.choice([(255, 255, 255), (245, 245, 245), (240, 242, 245)])
        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw bounding border (EVSE sticker border)
        border_color = (180, 180, 180)
        draw.rectangle([(5, 5), (width - 6, height - 6)], outline=border_color, width=2)
        
        # Header text
        header = random.choice(["EVSE ID:", "CHARGING OUTLET:", "STATION IDENTIFIER:", "IDENT-NR:"])
        draw.text((20, 20), header, fill=(60, 60, 60))
        
        # Main ID text
        text_color = random.choice([(0, 0, 0), (20, 20, 20), (10, 10, 10)])
        draw.text((20, 55), evse_id, fill=text_color)
        
        # Helper footer text
        footer = random.choice(["Tap Electric B.V.", "Operator Network", "DIN SPEC 91286 / ISO 15118", ""])
        if footer:
            draw.text((20, 110), footer, fill=(120, 120, 120))
            
        return img

    def generate_pair(self, standard: str = "ISO_15118") -> Tuple[Image.Image, str]:
        evse_id = self.generate_random_evse_id(standard=standard)
        clean_img = self.render_sticker_image(evse_id)
        degraded_img = self.simulator.apply(clean_img)
        return degraded_img, evse_id

    def generate_dataset(self, output_dir: str, count: int = 100) -> List[dict]:
        os.makedirs(output_dir, exist_ok=True)
        manifest = []
        for i in range(count):
            standard = "ISO_15118" if random.random() < 0.6 else "DIN_91286"
            degraded_img, evse_id = self.generate_pair(standard=standard)
            
            filename = f"synthetic_evse_{i:05d}.png"
            filepath = os.path.join(output_dir, filename)
            degraded_img.save(filepath)
            
            manifest.append({
                "image_path": filepath,
                "text": evse_id,
                "standard": standard
            })
        return manifest
