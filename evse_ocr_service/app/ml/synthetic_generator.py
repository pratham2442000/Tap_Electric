import os
import random
import string
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont
from app.ml.augmentation import DegradationSimulator

# Standard charging network operator codes in Europe
SAMPLE_OPERATORS = ["TNM", "ISE", "FST", "ALL", "ION", "ENB", "IBD", "EON", "VTT", "NRG"]
COUNTRY_CODES = ["DE", "NL", "FR", "UK", "BE", "AT", "CH", "SE", "NO", "IT", "ES"]

# Common system font search paths across Linux, Windows, macOS
FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/mnt/c/Windows/Fonts/arialbd.ttf",
    "/mnt/c/Windows/Fonts/calibrib.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]

FONT_CANDIDATES_REGULAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/mnt/c/Windows/Fonts/arial.ttf",
    "/mnt/c/Windows/Fonts/calibri.ttf",
    "/Library/Fonts/Arial.ttf",
]


class SyntheticEVSEGenerator:
    """
    Generates synthetic EVSE ID sticker labels with realistic typography,
    standard layouts, simulated QR codes, operator branding, and applies
    simulated environmental degradation.
    """
    def __init__(self, simulator: Optional[DegradationSimulator] = None):
        self.simulator = simulator or DegradationSimulator()

    @staticmethod
    def _get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
        """Loads scalable TrueType system font if available, falling back to PIL default."""
        candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES_REGULAR
        for path in candidates:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def generate_random_evse_id(self, standard: str = "ISO_15118") -> str:
        """Generates random EVSE ID conforming to ISO 15118 or DIN 91286 / EVCO standard."""
        country = random.choice(COUNTRY_CODES)
        operator = random.choice(SAMPLE_OPERATORS)
        outlet_digits = "".join(random.choices(string.digits, k=random.randint(6, 10)))
        
        if standard == "ISO_15118":
            delim = "*" if random.random() < 0.75 else ""
            return f"{country}{delim}{operator}{delim}E{outlet_digits}"
        else:  # DIN_91286 / EVCO
            prefix = "+49" if random.random() < 0.5 else country
            delim = "*" if random.random() < 0.75 else "-"
            return f"{prefix}{delim}{operator}{delim}{outlet_digits}"

    def _draw_simulated_qr_code(self, draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 80):
        """Draws a stylized, realistic 2D DataMatrix / QR Code square."""
        draw.rectangle([(x, y), (x + size, y + size)], fill=(255, 255, 255), outline=(120, 120, 120), width=1)
        grid_cells = 9
        cell_size = size / grid_cells
        
        # Draw position detection markers (top-left, top-right, bottom-left)
        corners = [(0, 0), (grid_cells - 3, 0), (0, grid_cells - 3)]
        for cx, cy in corners:
            # Outer 3x3 box
            draw.rectangle(
                [(x + cx * cell_size, y + cy * cell_size),
                 (x + (cx + 3) * cell_size - 1, y + (cy + 3) * cell_size - 1)],
                fill=(20, 20, 20)
            )
            # Inner white 1x1
            draw.rectangle(
                [(x + (cx + 0.6) * cell_size, y + (cy + 0.6) * cell_size),
                 (x + (cx + 2.4) * cell_size - 1, y + (cy + 2.4) * cell_size - 1)],
                fill=(255, 255, 255)
            )
            # Center dark dot
            draw.rectangle(
                [(x + (cx + 1.1) * cell_size, y + (cy + 1.1) * cell_size),
                 (x + (cx + 1.9) * cell_size - 1, y + (cy + 1.9) * cell_size - 1)],
                fill=(20, 20, 20)
            )

        # Draw pseudo-random payload grid dots
        for r in range(grid_cells):
            for c in range(grid_cells):
                # Skip finder corners
                if (r < 3 and c < 3) or (r < 3 and c >= grid_cells - 3) or (r >= grid_cells - 3 and c < 3):
                    continue
                if random.random() < 0.5:
                    px = int(x + c * cell_size)
                    py = int(y + r * cell_size)
                    draw.rectangle([(px, py), (px + int(cell_size) - 1, py + int(cell_size) - 1)], fill=(30, 30, 30))

    def render_sticker_image(
        self,
        evse_id: str,
        width: int = 420,
        height: int = 160,
        include_qr: Optional[bool] = None
    ) -> Image.Image:
        """
        Renders a realistic EVSE charger label sticker with scalable bold typography,
        operator metadata, subtle textures, and optional simulated QR/DataMatrix codes.
        """
        # Sticker background palette (metallic silver, clean white, warm tint, soft grey)
        bg_color = random.choice([
            (255, 255, 255),
            (248, 249, 251),
            (242, 244, 248),
            (252, 250, 245)
        ])
        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Draw realistic outer sticker border & corner accent
        border_color = random.choice([(160, 160, 160), (130, 140, 155), (100, 100, 100)])
        draw.rectangle([(6, 6), (width - 7, height - 7)], outline=border_color, width=2)

        # Top banner / accent line (optional operator brand bar)
        if random.random() < 0.5:
            banner_color = random.choice([(41, 128, 185), (39, 174, 96), (52, 73, 94), (230, 126, 34)])
            draw.rectangle([(7, 7), (width - 8, 14)], fill=banner_color)

        # Decide whether to render QR code (default ~60% probability)
        show_qr = include_qr if include_qr is not None else (random.random() < 0.65)
        qr_size = 76
        qr_x = width - qr_size - 20
        qr_y = (height - qr_size) // 2 + 5

        if show_qr:
            self._draw_simulated_qr_code(draw, qr_x, qr_y, size=qr_size)

        # Load fonts with proportional sizes
        font_header = self._get_font(13, bold=True)
        font_main = self._get_font(21, bold=True)
        font_footer = self._get_font(11, bold=False)

        # Header text
        header = random.choice([
            "EVSE ID / IDENT-NR:",
            "CHARGING OUTLET ID:",
            "STATION IDENTIFIER:",
            "LADEPUNKT ID:",
            "EVSE IDENTIFIER:"
        ])
        draw.text((20, 22), header, fill=(70, 75, 85), font=font_header)

        # Main ID text (prominent, bold, high contrast)
        text_color = random.choice([(10, 10, 10), (0, 0, 0), (25, 25, 30)])
        draw.text((20, 58), evse_id, fill=text_color, font=font_main)

        # Power badge & secondary info
        power_spec = random.choice(["22 kW AC", "11 kW AC", "50 kW DC", "150 kW DC", "300 kW Ultra-Fast", ""])
        footer_brand = random.choice(["Tap Electric B.V.", "DIN SPEC 91286 / ISO 15118", "Operator Network", "24/7 Support"])

        footer_text = f"{footer_brand}  •  {power_spec}" if power_spec else footer_brand
        draw.text((20, 120), footer_text, fill=(110, 115, 125), font=font_footer)

        return img

    def generate_pair(self, standard: str = "ISO_15118") -> Tuple[Image.Image, str]:
        """Generates a degraded synthetic EVSE image and ground-truth text pair."""
        evse_id = self.generate_random_evse_id(standard=standard)
        clean_img = self.render_sticker_image(evse_id)
        degraded_img = self.simulator.apply(clean_img)
        return degraded_img, evse_id

    def generate_dataset(self, output_dir: str, count: int = 100) -> List[dict]:
        """Generates a dataset of degraded synthetic EVSE images with metadata manifest."""
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

