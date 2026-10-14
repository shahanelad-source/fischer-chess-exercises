import os
import glob
import webbrowser
import fitz  # PyMuPDF
import cv2
import numpy as np
import chess

def find_pdf_file() -> str:
    """Finds the Bobby Fischer PDF in the current folder or parent folder."""
    for pattern in ["*Fischer*.pdf", "*.pdf", "../*Fischer*.pdf", "../*.pdf"]:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "Could not find the PDF file. Place the Bobby Fischer PDF in the same folder as this script."
    )

def extract_chessboard_from_page(pdf_path: str, page_number: int, output_dir: str = "poc_output") -> str:
    """Renders a PDF page, crops the square chessboard, and slices it into 64 squares."""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    pix = page.get_pixmap(dpi=300)

    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    board_bbox = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if 0.90 <= aspect_ratio <= 1.10 and area > max_area:
            max_area = area
            board_bbox = (x, y, w, h)

    if not board_bbox:
        raise ValueError(f"No chessboard detected on page {page_number}.")

    x, y, w, h = board_bbox
    margin_x, margin_y = int(w * 0.015), int(h * 0.015)
    cropped_board = img[y + margin_y : y + h - margin_y, x + margin_x : x + w - margin_x]

    board_std = cv2.resize(cropped_board, (512, 512), interpolation=cv2.INTER_AREA)
    board_path = os.path.join(output_dir, f"page_{page_number}_board.png")
    cv2.imwrite(board_path, board_std)

    squares_dir = os.path.join(output_dir, "squares")
    os.makedirs(squares_dir, exist_ok=True)
    sq_size = 512 // 8
    for r_idx, rank in enumerate("87654321"):
        for f_idx, file in enumerate("abcdefgh"):
            sq_img = board_std[r_idx * sq_size : (r_idx + 1) * sq_size,
                               f_idx * sq_size : (f_idx + 1) * sq_size]
            cv2.imwrite(os.path.join(squares_dir, f"{file}{rank}.png"), sq_img)

    return board_path

def launch_interactive_puzzle(puzzle_data: dict, board_img_path: str, output_html: str = "frame_19_poc.html"):
    """Verifies the mate with python-chess and opens a playable HTML board side-by-side with the book scan."""
    board = chess.Board(puzzle_data["fen"])
    mating_moves = []
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            mating_moves.append(board.uci(move))
        board.pop()

    print(f"✅ Extracted Board Image: {os.path.abspath(board_img_path)}")
    print(f"✅ Verified Mating Move(s) via python-chess: {mating_moves}")

    rel_img_path = os.path.relpath(board_img_path, os.path.dirname(os.path.abspath(output_html))).replace("\\", "/")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Bobby Fischer Teaches Chess - Frame {puzzle_data['frame']} PoC</title>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 30px auto; text-align: center; background: #f4f4f2; }}
        .container {{ display: flex; gap: 24px; justify-content: center; align-items: flex-start; flex-wrap: wrap; }}
        .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); width: 400px; }}
        #board {{ width: 380px; margin: 15px auto; }}
        .scan-img {{ width: 380px; height: 380px; object-fit: contain; border: 2px solid #222; margin: 15px auto; display: block; }}
        #status {{ font-weight: bold; margin-top: 12px; padding: 10px; border-radius: 6px; background: #eee; }}
        .success {{ background: #d4edda !important; color: #155724; }}
        .error {{ background: #f8d7da !important; color: #721c24; }}
        button {{ margin-top: 12px; padding: 8px 16px; cursor: pointer; border-radius: 6px; border: 1px solid #ccc; font-weight: 600; }}
    </style>
</head>
<body>
    <h2>Frame {puzzle_data['frame']} (Book Page 38 / PDF Page {puzzle_data['pdf_page']})</h2>
    <p><strong>Prompt:</strong> {puzzle_data['prompt']}</p>
    <div class="container">
        <div class="card">
            <h3>1. Extracted Book Diagram</h3>
            <img class="scan-img" src="{rel_img_path}" alt="Extracted Board">
            <p style="font-size: 0.9em; color: #555;">Cropped automatically via PyMuPDF + OpenCV</p>
        </div>
        <div class="card">
            <h3>2. Interactive Board</h3>
            <div id="board"></div>
            <div id="status">Black to move: Drag a Black piece to deliver checkmate!</div>
            <button onclick="resetPuzzle()">Reset Position</button>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
    <script>
        const startFen = "{puzzle_data['fen']}";
        const game = new Chess(startFen);
        const statusEl = document.getElementById('status');

        function onDragStart (source, piece) {{
            if (game.game_over()) return false;
            if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
                (game.turn() === 'b' && piece.search(/^w/) !== -1)) {{
                return false;
            }}
        }}

        function onDrop (source, target) {{
            const move = game.move({{ from: source, to: target, promotion: 'q' }});
            if (move === null) return 'snapback';

            if (game.in_checkmate()) {{
                statusEl.className = 'success';
                statusEl.innerHTML = "✅ Checkmate! (" + move.san + ") — Fischer vs. Keres, Bled 1959!";
            }} else {{
                statusEl.className = 'error';
                statusEl.innerHTML = "❌ " + move.san + " is not checkmate. Try again!";
                setTimeout(() => {{
                    game.undo();
                    board.position(game.fen());
                }}, 900);
            }}
        }}

        const board = Chessboard('board', {{
            draggable: true,
            position: startFen,
            orientation: 'white',
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{{piece}}.png',
            onDragStart: onDragStart,
            onDrop: onDrop
        }});

        function resetPuzzle() {{
            game.load(startFen);
            board.position(startFen);
            statusEl.className = '';
            statusEl.innerHTML = 'Black to move: Drag a Black piece to deliver checkmate!';
        }}
    </script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    webbrowser.open("file://" + os.path.abspath(output_html))

if __name__ == "__main__":
    pdf_file = find_pdf_file()
    print(f"Found PDF: {pdf_file}")

    board_img = extract_chessboard_from_page(pdf_file, page_number=48)

    frame_19 = {
        "frame": 19,
        "pdf_page": 48,
        "fen": "8/8/2P5/3K1k2/2R3p1/2q5/8/8 b - - 0 1",
        "prompt": "Draw an arrow to show how I (Black) mated Paul Keres in this position."
    }
    launch_interactive_puzzle(frame_19, board_img)