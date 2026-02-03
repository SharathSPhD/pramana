#!/usr/bin/env python3
"""Convert Mermaid diagrams to PDF using mermaid.ink API and reportlab."""

import base64
import sys
from pathlib import Path
from urllib.parse import quote

try:
    import requests
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from PIL import Image
    from io import BytesIO
except ImportError as e:
    print(f"Error: Missing required package. Install with:")
    print(f"  pip install requests reportlab pillow")
    print(f"\nMissing: {e.name}")
    sys.exit(1)


def convert_mermaid_to_pdf(mmd_file: Path, pdf_file: Path) -> None:
    """Convert a Mermaid diagram file to PDF."""
    # Read the Mermaid diagram
    mermaid_code = mmd_file.read_text()
    
    # Encode the diagram for the API
    encoded = base64.urlsafe_b64encode(mermaid_code.encode()).decode()
    
    # Fetch SVG from mermaid.ink API
    api_url = f"https://mermaid.ink/img/{encoded}"
    
    print(f"Fetching diagram from API...")
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    
    # Convert SVG to image (PIL can handle SVG if rsvg is available, otherwise use PNG)
    # mermaid.ink returns PNG by default, so we can use it directly
    img = Image.open(BytesIO(response.content))
    
    # Create PDF
    c = canvas.Canvas(str(pdf_file), pagesize=A4)
    width, height = A4
    
    # Calculate scaling to fit page with margins
    margin = 40  # 40 points margin
    max_width = width - 2 * margin
    max_height = height - 2 * margin
    
    img_width, img_height = img.size
    scale = min(max_width / img_width, max_height / img_height)
    
    # Center the image
    x = (width - img_width * scale) / 2
    y = (height - img_height * scale) / 2
    
    # Draw image
    img_io = BytesIO()
    img.save(img_io, format='PNG')
    img_io.seek(0)
    
    c.drawImage(ImageReader(img_io), x, y, width=img_width * scale, height=img_height * scale)
    c.save()
    
    print(f"✓ Converted {mmd_file.name} → {pdf_file.name}")


def main() -> None:
    """Convert all Mermaid files to PDF."""
    figures_dir = Path(__file__).parent
    
    diagrams = [
        ("architecture.mmd", "architecture.pdf"),
        ("nyaya_flow.mmd", "nyaya_flow.pdf"),
    ]
    
    for mmd_name, pdf_name in diagrams:
        mmd_file = figures_dir / mmd_name
        pdf_file = figures_dir / pdf_name
        
        if not mmd_file.exists():
            print(f"Error: {mmd_file} not found")
            continue
        
        try:
            convert_mermaid_to_pdf(mmd_file, pdf_file)
        except Exception as e:
            print(f"Error converting {mmd_name}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
