import pymupdf
import io
import os
import sys
from PIL import Image
from unittest.mock import MagicMock

# --- WINDOWS C-LIBRARY BYPASS ---
# Tricks the neural network into skipping missing Linux graphics/noise engines
sys.modules['cairosvg'] = MagicMock()
sys.modules['pyfastnoiselite'] = MagicMock()

# Force Python to recognize the current directory so it can find the local AI module
sys.path.insert(0, os.path.abspath('.'))

try:
    from chess_diagram_to_fen import get_fen
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)

def sanity_check_local_ml():
    pdf_filename = "Bobby Fischer teaches chess (Fischer, Bobby, 1943-2008, Margulies etc.) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
    
    # Automatically find the PDF whether it is in this folder or the parent folder
    if not os.path.exists(pdf_filename):
        parent_path = os.path.join("..", pdf_filename)
        if os.path.exists(parent_path):
            pdf_filename = parent_path
        else:
            print(f"Error: '{pdf_filename}' not found. Please move the PDF into your current folder.")
            return

    print(f"Opening '{pdf_filename}' and warming up local PyTorch CNN...")
    doc = pymupdf.open(pdf_filename)
    
    for page_num in range(15, len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if not text: continue
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_str = line.strip()
            
            # Locate Frame 6 (the first active exercise in the book)
            if line_str.isdigit() and int(line_str) == 6:
                print(f"-> Found Frame 6 on page {page_num + 1}. Extracting image...")
                
                pix = page.get_pixmap(dpi=200)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                print("-> Processing through local model (this may take a few seconds)...")
                try:
                    # The local CNN strips the vintage hatching and reads the pieces
                    result = get_fen(img=img, game="chess", auto_rotate_image=True, auto_rotate_board=True)
                    print("\n========================================")
                    print("SUCCESS! LOCAL ML SANITY CHECK PASSED")
                    print("========================================")
                    print(f"Frame 6 FEN: {result.fen}")
                    print("========================================\n")
                    print("Your bypasses work. You can now use the full extraction script.")
                except Exception as e:
                    print(f"\nLocal Model Error: {e}")
                
                return # Shut down after one successful extraction

if __name__ == "__main__":
    sanity_check_local_ml()