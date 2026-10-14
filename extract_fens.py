import pymupdf
import json
import io
import os
import google.generativeai as genai
from PIL import Image

# Initialize the Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash-latest')

def parse_single_board_sanity_check():
    pdf_filename = "Bobby Fischer teaches chess (Fischer, Bobby, 1943-2008, Margulies etc.) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
    
    if not os.path.exists(pdf_filename):
        print(f"Error: Could not find '{pdf_filename}'.")
        return

    doc = pymupdf.open(pdf_filename)
    exercises = []
    current_chapter = "Chapter 1: Elements of Checkmate"
    
    print(f"Running Sanity Check on '{pdf_filename}'...")
    print("The script will extract ONLY the first exercise and then stop.\n")
    
    for page_num in range(15, len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if not text: continue
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_str = line.strip()
            
            # Identify standalone frame numbers indicating an exercise
            if line_str.isdigit() and 0 < int(line_str) < 300:
                frame_id = int(line_str)
                context_lines = [lines[j].strip() for j in range(i + 1, min(i + 10, len(lines))) if lines[j].strip()]
                q_text = " ".join(context_lines)
                
                is_exercise = any(kw in q_text.lower() for kw in ["can capture", "checkmate", "interpose", "yes", "no", "flee", "defense", "defend"])
                
                if is_exercise:
                    print(f"Found Frame {frame_id}. Sending image to Gemini API...")
                    
                    pix = page.get_pixmap(dpi=200)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    prompt = """
                    You are a chess grandmaster engine. Look at the chessboard diagram on this page.
                    The dark squares have vintage diagonal hatching—ignore the hatching and focus on the piece silhouettes.
                    Output ONLY the valid FEN (Forsyth-Edwards Notation) string for the position shown. 
                    Assume it is White's turn if unclear (w - - 0 1). Do not include any formatting, markdown, or other text.
                    """
                    
                    try:
                        response = model.generate_content([prompt, img])
                        fen_str = response.text.strip()
                    except Exception as e:
                        print(f"API failed to parse Frame {frame_id}: {e}")
                        fen_str = "8/8/8/8/8/8/8/8 w - - 0 1"

                    options = ["can capture checking piece", "cannot capture checking piece"]
                    if "yes" in q_text.lower() and "no" in q_text.lower():
                        options = ["yes", "no"]
                    elif "flee" in q_text.lower() or "interpose" in q_text.lower():
                        options = ["can capture attacker", "can flee", "can interpose", "can do neither"]

                    exercises.append({
                        "chapter": current_chapter,
                        "frame": frame_id,
                        "text": q_text[:120].strip() + "...",
                        "fen": fen_str,
                        "options": options,
                        "correct": 0,
                    })
                    
                    print(f"Success! Extracted Frame {frame_id} FEN: {fen_str}")
                    
                    # --- SANITY CHECK STOP ---
                    with open("exerciseBank_sanity.js", "w", encoding="utf-8") as f:
                        f.write("const exerciseBank = ")
                        json.dump(exercises, f, indent=4, ensure_ascii=False)
                        f.write(";\n")
                        
                    print("\nSanity check complete. Data saved to 'exerciseBank_sanity.js'.")
                    print("Check the JS file or paste the FEN into Lichess.org/editor to verify accuracy.")
                    
                    # Return exits the entire function, ensuring only 1 API call is made
                    return 

if __name__ == "__main__":
    parse_single_board_sanity_check()