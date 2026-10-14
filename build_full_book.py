import os
import re
import glob
import json
import webbrowser
import fitz  # PyMuPDF
import cv2
import numpy as np
import chess

# ==============================================================================
# 1. REFERENCE CALIBRATION BOARDS FROM THE BOOK
# ==============================================================================
REFERENCE_BOARDS = [
    (20, "7k/8/8/7p/1nb4P/2P5/8/K7 w - - 0 1"),
    (34, "7k/6pQ/8/5B2/8/8/8/4K3 b - - 0 1"),
    (36, "6Rk/7p/8/5N2/8/8/8/6K1 b - - 0 1"),
    (42, "7k/7p/8/3B4/3B4/6K1/8/8 b - - 0 1"),
    (43, "4k2q/2pp1Q2/8/1b2N3/p7/8/3P1PP1/4K3 b - - 0 1"),
    (46, "rk6/1pP5/1P2Q3/8/8/8/8/7K b - - 0 1"),
    (47, "5Rk1/6pp/3Q4/8/8/8/6K1/8 b - - 0 1"),
    (48, "8/8/2P5/3K1k2/2R3p1/2q5/8/8 b - - 0 1"),
    (51, "3r1k2/1q1P2b1/7Q/p3p2p/1p6/1B3P2/PPP5/1K1R4 w - - 0 1"),
]

PIECE_CLASSES = ["1", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]

def find_pdf_file() -> str:
    for pattern in ["*Fischer*.pdf", "*.pdf", "../*Fischer*.pdf", "../*.pdf"]:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError("Could not find the PDF file in the current directory.")

def render_page(doc, page_num: int, dpi: int = 200) -> np.ndarray:
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)

def detect_exercise_board(img: np.ndarray, allow_multiple: bool = False):
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h
        if 0.90 <= aspect <= 1.10 and (0.32 * w_img <= w <= 0.80 * w_img):
            cy = y + h / 2.0
            if cy > 0.22 * h_img:
                candidates.append((x, y, w, h))

    if not candidates:
        return None

    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    unique_boards = []
    for box in candidates:
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        if not any(abs(cx - (ux + uw / 2.0)) < uw * 0.25 and abs(cy - (uy + uh / 2.0)) < uh * 0.25
                   for (ux, uy, uw, uh) in unique_boards):
            unique_boards.append(box)

    if not allow_multiple and len(unique_boards) > 1:
        largest_area = unique_boards[0][2] * unique_boards[0][3]
        second_area = unique_boards[1][2] * unique_boards[1][3]
        if second_area > 0.65 * largest_area:
            return None

    x, y, w, h = unique_boards[0]
    margin_x, margin_y = int(w * 0.016), int(h * 0.016)
    cropped = img[y + margin_y : y + h - margin_y, x + margin_x : x + w - margin_x]
    board_512 = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_AREA)
    return board_512, (x, y, w, h)

def crop_answer_box(next_page_img: np.ndarray) -> np.ndarray:
    h_img, w_img = next_page_img.shape[:2]
    return next_page_img[0 : int(h_img * 0.36), int(w_img * 0.04) : int(w_img * 0.96)]


# ==============================================================================
# 2. SELF-CALIBRATING OPENCV TEMPLATE CLASSIFIER
# ==============================================================================
class BookPieceClassifier:
    def __init__(self, doc):
        self.templates = {0: {c: [] for c in PIECE_CLASSES}, 1: {c: [] for c in PIECE_CLASSES}}
        self._calibrate(doc)

    def _calibrate(self, doc):
        print("Calibrating piece recognizer directly from book diagrams...")
        for pdf_page, fen in REFERENCE_BOARDS:
            img = render_page(doc, pdf_page, dpi=200)
            res = detect_exercise_board(img, allow_multiple=(pdf_page == 20))
            if res is None:
                continue
            board_512, _ = res
            gray = cv2.cvtColor(board_512, cv2.COLOR_BGR2GRAY)
            board_part = fen.split()[0]
            rows = board_part.split("/")
            for r_idx, row_str in enumerate(rows):
                expanded = ""
                for ch in row_str:
                    expanded += "1" * int(ch) if ch.isdigit() else ch
                for f_idx, piece_ch in enumerate(expanded):
                    sq_color = (r_idx + f_idx) % 2
                    y0, x0 = r_idx * 64, f_idx * 64
                    patch = gray[y0 + 10 : y0 + 54, x0 + 10 : x0 + 54]
                    if len(self.templates[sq_color][piece_ch]) < 6:
                        self.templates[sq_color][piece_ch].append(patch)

        empty_light = self.templates[0]["1"][0]
        empty_dark = self.templates[1]["1"][0]
        for piece_ch in PIECE_CLASSES:
            for sq_color, other_color, bg_patch in [(0, 1, empty_light), (1, 0, empty_dark)]:
                if not self.templates[sq_color][piece_ch] and self.templates[other_color][piece_ch]:
                    src_patch = self.templates[other_color][piece_ch][0]
                    synth = bg_patch.copy()
                    if piece_ch.islower():
                        mask = src_patch < 85
                        synth[mask] = src_patch[mask]
                    else:
                        mask = (src_patch < 100) | (src_patch > 230)
                        synth[mask] = src_patch[mask]
                    self.templates[sq_color][piece_ch].append(synth)

    def classify_board(self, board_512: np.ndarray) -> str:
        gray = cv2.cvtColor(board_512, cv2.COLOR_BGR2GRAY)
        grid = [[None for _ in range(8)] for _ in range(8)]
        king_scores = {"K": [], "k": []}

        for r in range(8):
            for f in range(8):
                sq_color = (r + f) % 2
                y0, x0 = r * 64, f * 64
                sq_img = gray[y0 : y0 + 64, x0 : x0 + 64]
                inner = sq_img[14:50, 14:50]

                if sq_color == 0 and np.mean(inner < 165) < 0.045:
                    grid[r][f] = "1"
                    continue

                best_piece = "1"
                best_score = -1.0

                for piece_ch, patch_list in self.templates[sq_color].items():
                    if piece_ch in ("P", "p") and r in (0, 7):
                        continue
                    for tmpl in patch_list:
                        res = cv2.matchTemplate(sq_img, tmpl, cv2.TM_CCOEFF_NORMED)
                        score = float(np.max(res))
                        if score > best_score:
                            best_score = score
                            best_piece = piece_ch
                        if piece_ch in ("K", "k"):
                            king_scores[piece_ch].append((score, r, f))

                grid[r][f] = best_piece

        for k_char in ("K", "k"):
            positions = [(r, f) for r in range(8) for f in range(8) if grid[r][f] == k_char]
            if len(positions) == 0 and king_scores[k_char]:
                king_scores[k_char].sort(reverse=True)
                _, kr, kf = king_scores[k_char][0]
                grid[kr][kf] = k_char
            elif len(positions) > 1:
                best_pos = max(positions, key=lambda pos: max(
                     s for (s, r, f) in king_scores[k_char] if (r, f) == pos
                ))
                for (r, f) in positions:
                    if (r, f) != best_pos:
                        grid[r][f] = "Q" if k_char == "K" else "q"

        fen_rows = []
        for r in range(8):
            row_str = ""
            empty_cnt = 0
            for f in range(8):
                ch = grid[r][f]
                if ch == "1":
                    empty_cnt += 1
                else:
                    if empty_cnt > 0:
                        row_str += str(empty_cnt)
                        empty_cnt = 0
                    row_str += ch
            if empty_cnt > 0:
                row_str += str(empty_cnt)
            fen_rows.append(row_str)

        return "/".join(fen_rows)


def determine_active_turn(fen_board: str, prompt_text: str) -> str:
    w_board = chess.Board(f"{fen_board} w - - 0 1")
    b_board = chess.Board(f"{fen_board} b - - 0 1")
    if b_board.is_check() and not w_board.is_check():
        return "b"
    if w_board.is_check() and not b_board.is_check():
        return "w"

    text_lower = prompt_text.lower()
    black_cues = [
        "i (black)", "how black", "can black mate", "black's defense",
        "black's saving", "black saves", "black's correct defense",
        "black's best move", "it's black's move", "were black",
        "what does black play", "how does black defend"
    ]
    if any(cue in text_lower for cue in black_cues):
        return "b"
    return "w"


# ==============================================================================
# 3. SCAN THE WHOLE PDF FOR EXERCISE FRAMES
# ==============================================================================
def extract_all_exercises(pdf_path: str, out_dir: str = "book_output") -> list:
    boards_dir = os.path.join(out_dir, "boards")
    answers_dir = os.path.join(out_dir, "answers")
    os.makedirs(boards_dir, exist_ok=True)
    os.makedirs(answers_dir, exist_ok=True)

    cache_path = os.path.join(out_dir, "exercises.json")
    if os.path.exists(cache_path):
        print(f"Loading cached exercises from {cache_path}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    doc = fitz.open(pdf_path)
    classifier = BookPieceClassifier(doc)
    exercises = []
    ref_fen_map = {p: fen for p, fen in REFERENCE_BOARDS}

    start_page, end_page = 34, min(342, len(doc) - 1)
    print(f"Scanning PDF pages {start_page} to {end_page} for exercise frames...")

    for page_num in range(start_page, end_page + 1):
        page_text = doc[page_num - 1].get_text("text")
        next_text = doc[page_num].get_text("text")

        if "NO ANSWER REQUIRED" in next_text.upper():
            continue
        if any(skip_word in page_text.upper() for skip_word in [
            "THIS IS THE END OF CHAPTER", "SUMMARY", "CHAPTER ", "CRITERION POSITIONS"
        ]):
            continue

        img = render_page(doc, page_num, dpi=200)
        detected = detect_exercise_board(img, allow_multiple=False)
        if detected is None:
            continue

        board_512, _ = detected
        nums = [int(n) for n in re.findall(r"\b(\d{1,3})\b", page_text) if 6 <= int(n) <= 275]
        expected_frame = max(6, int((page_num - 34) * 0.88) + 6)
        frame_num = min(nums, key=lambda n: abs(n - expected_frame)) if nums else page_num - 28

        lines = [ln.strip() for ln in page_text.splitlines() if len(ln.strip()) > 2]
        clean_prompt = " ".join([ln for ln in lines if not ln.isdigit()][-6:])
        if not clean_prompt:
            clean_prompt = "Find the best move or determine if mate is possible."

        if page_num in ref_fen_map:
            full_fen = ref_fen_map[page_num]
        else:
            fen_board = classifier.classify_board(board_512)
            turn = determine_active_turn(fen_board, clean_prompt)
            full_fen = f"{fen_board} {turn} - - 0 1"

        board_rel = f"boards/page_{page_num}_frame_{frame_num}.png"
        ans_rel = f"answers/page_{page_num}_frame_{frame_num}_ans.png"
        cv2.imwrite(os.path.join(out_dir, board_rel), board_512)

        next_img = render_page(doc, page_num + 1, dpi=150)
        ans_crop = crop_answer_box(next_img)
        cv2.imwrite(os.path.join(out_dir, ans_rel), ans_crop)

        exercises.append({
            "id": len(exercises) + 1,
            "frame": frame_num,
            "pdf_page": page_num,
            "fen": full_fen,
            "prompt": clean_prompt,
            "board_img": f"book_output/{board_rel}",
            "ans_img": f"book_output/{ans_rel}"
        })
        print(f"[{len(exercises):03d}] Page {page_num:3d} (Frame {frame_num:3d}) -> {full_fen}")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(exercises, f, indent=2)

    return exercises


# ==============================================================================
# 4. GENERATE THE INTERACTIVE VERTICAL-STACK HTML APP
# ==============================================================================
def build_html_app(exercises: list, output_html: str = "fischer_interactive_book.html"):
    exercises_json = json.dumps(exercises)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Bobby Fischer Teaches Chess - Interactive Exercises</title>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 560px; margin: 20px auto; text-align: center; background: #f4f4f2; color: #222; padding: 0 12px; }}
        .nav-bar {{ background: white; padding: 14px 16px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 18px; display: flex; gap: 8px; justify-content: space-between; align-items: center; flex-wrap: wrap; }}
        select {{ padding: 8px 10px; border-radius: 6px; border: 1px solid #ccc; font-weight: 600; flex: 1; min-width: 180px; }}
        .container {{ display: flex; flex-direction: column; gap: 20px; align-items: center; }}
        .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); width: 100%; box-sizing: border-box; }}
        .card h3 {{ margin-top: 0; margin-bottom: 10px; }}
        #board {{ width: 380px; max-width: 100%; margin: 12px auto; }}
        .scan-img {{ width: 380px; max-width: 100%; height: 380px; object-fit: contain; border: 2px solid #222; margin: 10px auto; display: block; background: #fff; }}
        .ans-img {{ width: 100%; max-width: 440px; border: 1px solid #bbb; border-radius: 6px; margin-top: 10px; display: none; }}
        #status {{ font-weight: bold; margin-top: 12px; padding: 10px; border-radius: 6px; background: #eee; }}
        .success {{ background: #d4edda !important; color: #155724; }}
        .error {{ background: #f8d7da !important; color: #721c24; }}
        .info {{ background: #cce5ff !important; color: #004085; }}
        .btn-row {{ display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-top: 12px; }}
        button {{ padding: 8px 14px; cursor: pointer; border-radius: 6px; border: 1px solid #ccc; font-weight: 600; background: #fff; }}
        button:hover {{ background: #f0f0f0; }}
        .prompt-box {{ background: #fff; padding: 14px 18px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 18px; text-align: left; font-size: 0.95em; line-height: 1.4; }}
        .fen-edit {{ display: none; margin-top: 10px; }}
        .fen-edit input {{ width: 72%; padding: 6px; border: 1px solid #ccc; border-radius: 4px; font-family: monospace; }}
    </style>
</head>
<body>
    <h2>Bobby Fischer Teaches Chess</h2>

    <div class="nav-bar">
        <button onclick="prevExercise()">◀ Prev</button>
        <select id="frameSelect" onchange="jumpToExercise(this.value)"></select>
        <button onclick="nextExercise()">Next ▶</button>
    </div>

    <div class="prompt-box">
        <div id="frameTitle" style="font-weight: 700; font-size: 1.1em; margin-bottom: 4px;"></div>
        <div id="promptText"></div>
    </div>

    <div class="container">
        <div class="card">
            <h3>1. Extracted Book Diagram</h3>
            <img id="scanImg" class="scan-img" src="" alt="Extracted Book Diagram">
            <div class="btn-row">
                <button onclick="toggleAnswer()">📖 Show/Hide Book Answer Box</button>
            </div>
            <img id="ansImg" class="ans-img" src="" alt="Book Answer Box">
        </div>

        <div class="card">
            <h3>2. Interactive Board</h3>
            <div id="board"></div>
            <div id="status"></div>
            <div class="btn-row">
                <button onclick="resetPuzzle()">🔄 Reset Position</button>
                <button onclick="flipTurn()">🔁 Switch Turn (W/B)</button>
                <button onclick="claimNoMate()">🚫 No Mate Possible</button>
                <button onclick="toggleFenEdit()">✏️ Edit FEN</button>
                <button onclick="exportExercisesJson()">💾 Save All Edits to JSON</button>
            </div>
            <div id="fenEditBox" class="fen-edit">
                <input type="text" id="fenInput">
                <button onclick="applyCustomFen()">Apply</button>
            </div>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
    <script>
        const exercises = {exercises_json};
        let currentIdx = 0;
        let startFen = "";
        let game = new Chess();
        let board = null;
        let isDefensePuzzle = false;
        let solvedMap = JSON.parse(localStorage.getItem('fischer_solved') || '{{}}');

        const statusEl = document.getElementById('status');
        const selectEl = document.getElementById('frameSelect');

        function canForceMate(chessInstance, maxPly, isAttackerTurn) {{
            if (chessInstance.in_checkmate()) return !isAttackerTurn;
            if (maxPly <= 0 || chessInstance.game_over()) return false;

            const moves = chessInstance.moves({{ verbose: true }});
            if (isAttackerTurn) {{
                for (let m of moves) {{
                    chessInstance.move(m);
                    const givesCheck = chessInstance.in_check();
                    const ok = (maxPly === 1 ? chessInstance.in_checkmate() :
                               (givesCheck && canForceMate(chessInstance, maxPly - 1, false)));
                    chessInstance.undo();
                    if (ok) return true;
                }}
                return false;
            }} else {{
                for (let m of moves) {{
                    chessInstance.move(m);
                    const stillMated = canForceMate(chessInstance, maxPly - 1, true);
                    chessInstance.undo();
                    if (!stillMated) return false;
                }}
                return true;
            }}
        }}

        function findBestDefenderReply(chessInstance) {{
            const replies = chessInstance.moves({{ verbose: true }});
            if (replies.length === 0) return null;
            for (let r of replies) {{
                chessInstance.move(r);
                const matedIn1 = canForceMate(chessInstance, 1, true);
                chessInstance.undo();
                if (!matedIn1) return r;
            }}
            return replies[0];
        }}

        function populateSelect() {{
            selectEl.innerHTML = '';
            exercises.forEach((ex, idx) => {{
                const opt = document.createElement('option');
                opt.value = idx;
                const mark = solvedMap[ex.pdf_page] ? '✅ ' : '';
                const edited = localStorage.getItem('fen_override_' + ex.pdf_page) ? '✏️ ' : '';
                opt.textContent = `${{mark}}${{edited}}Frame ${{ex.frame}} (PDF p.${{ex.pdf_page}})`;
                selectEl.appendChild(opt);
            }});
            selectEl.value = currentIdx;
        }}

        function markSolved() {{
            const ex = exercises[currentIdx];
            solvedMap[ex.pdf_page] = true;
            localStorage.setItem('fischer_solved', JSON.stringify(solvedMap));
            populateSelect();
        }}

        function loadExercise(idx) {{
            currentIdx = Math.max(0, Math.min(exercises.length - 1, parseInt(idx)));
            const ex = exercises[currentIdx];
            startFen = localStorage.getItem('fen_override_' + ex.pdf_page) || ex.fen;

            document.getElementById('frameTitle').textContent =
                `Frame ${{ex.frame}} (Book Page ${{ex.pdf_page - 10}} / PDF Page ${{ex.pdf_page}})`;
            document.getElementById('promptText').textContent = ex.prompt;
            document.getElementById('scanImg').src = ex.board_img;
            document.getElementById('ansImg').src = ex.ans_img;
            document.getElementById('ansImg').style.display = 'none';
            document.getElementById('fenInput').value = startFen;

            game.load(startFen);
            isDefensePuzzle = game.in_check();
            board.position(startFen, false);
            updateInitialStatus();
            populateSelect();
        }}

        function updateInitialStatus() {{
            const side = game.turn() === 'w' ? 'White' : 'Black';
            statusEl.className = '';
            if (isDefensePuzzle) {{
                statusEl.innerHTML = `🛡️ ${{side}} is in check! Drag a ${{side}} piece to defend/escape.`;
            }} else {{
                statusEl.innerHTML = `⚔️ ${{side}} to move: Drag a ${{side}} piece to find the mating combination!`;
            }}
        }}

        function onDragStart(source, piece) {{
            if (game.game_over()) return false;
            if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
                (game.turn() === 'b' && piece.search(/^w/) !== -1)) {{
                return false;
            }}
        }}

        function onDrop(source, target) {{
            const move = game.move({{ from: source, to: target, promotion: 'q' }});
            if (move === null) return 'snapback';

            if (isDefensePuzzle) {{
                if (!canForceMate(game, 1, true)) {{
                    statusEl.className = 'success';
                    statusEl.innerHTML = `✅ Correct defense (${{move.san}})! You escaped checkmate.`;
                    markSolved();
                }} else {{
                    statusEl.className = 'error';
                    statusEl.innerHTML = `❌ ${{move.san}} is a useless defense — opponent still mates in 1!`;
                    setTimeout(() => {{ game.undo(); board.position(game.fen()); }}, 1000);
                }}
                return;
            }}

            if (game.in_checkmate()) {{
                statusEl.className = 'success';
                statusEl.innerHTML = `✅ Checkmate! (${{move.san}}) — Puzzle Solved!`;
                markSolved();
                return;
            }}

            if (game.in_check() && (canForceMate(game, 2, false) || canForceMate(game, 4, false))) {{
                const reply = findBestDefenderReply(game);
                if (reply) {{
                    statusEl.className = 'info';
                    statusEl.innerHTML = `👍 Good check (${{move.san}})! Opponent replies ${{reply.san}}... finish the mate!`;
                    setTimeout(() => {{
                        game.move(reply);
                        board.position(game.fen());
                    }}, 450);
                    return;
                }}
            }}

            statusEl.className = 'error';
            statusEl.innerHTML = `❌ ${{move.san}} does not force mate. Try again!`;
            setTimeout(() => {{
                game.undo();
                board.position(game.fen());
            }}, 900);
        }}

        function claimNoMate() {{
            const temp = new Chess(game.fen());
            if (!canForceMate(temp, 1, true) && !canForceMate(temp, 3, true)) {{
                statusEl.className = 'success';
                statusEl.innerHTML = `✅ Correct! There is no forced mate in this position.`;
                markSolved();
            }} else {{
                statusEl.className = 'error';
                statusEl.innerHTML = `❌ Careful — there IS a forced mate here! Keep looking.`;
            }}
        }}

        function flipTurn() {{
            let parts = startFen.split(' ');
            parts[1] = parts[1] === 'w' ? 'b' : 'w';
            startFen = parts.join(' ');
            document.getElementById('fenInput').value = startFen;
            localStorage.setItem('fen_override_' + exercises[currentIdx].pdf_page, startFen);
            resetPuzzle();
            populateSelect();
        }}

        function toggleAnswer() {{
            const el = document.getElementById('ansImg');
            el.style.display = el.style.display === 'block' ? 'none' : 'block';
        }}

        function toggleFenEdit() {{
            const el = document.getElementById('fenEditBox');
            el.style.display = el.style.display === 'block' ? 'none' : 'block';
        }}

        function applyCustomFen() {{
            const custom = document.getElementById('fenInput').value.trim();
            const test = new Chess();
            if (test.load(custom)) {{
                startFen = custom;
                localStorage.setItem('fen_override_' + exercises[currentIdx].pdf_page, custom);
                resetPuzzle();
                populateSelect();
                statusEl.className = 'info';
                statusEl.innerHTML = '💾 FEN saved to browser memory!';
            }} else {{
                alert("Invalid FEN string!");
            }}
        }}

        function exportExercisesJson() {{
            const updated = exercises.map(ex => {{
                const override = localStorage.getItem('fen_override_' + ex.pdf_page);
                return override ? {{ ...ex, fen: override }} : ex;
            }});
            const blob = new Blob([JSON.stringify(updated, null, 2)], {{ type: 'application/json' }});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'exercises.json';
            a.click();
        }}

        function resetPuzzle() {{
            game.load(startFen);
            isDefensePuzzle = game.in_check();
            board.position(startFen);
            updateInitialStatus();
        }}

        function prevExercise() {{ loadExercise(currentIdx - 1); }}
        function nextExercise() {{ loadExercise(currentIdx + 1); }}
        function jumpToExercise(val) {{ loadExercise(val); }}

        document.addEventListener('keydown', (e) => {{
            if (e.target.tagName === 'INPUT') return;
            if (e.key === 'ArrowLeft') prevExercise();
            if (e.key === 'ArrowRight') nextExercise();
        }});

        board = Chessboard('board', {{
            draggable: true,
            position: 'start',
            orientation: 'white',
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{{piece}}.png',
            onDragStart: onDragStart,
            onDrop: onDrop
        }});

        loadExercise(0);
    </script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n🎉 Generated full interactive book: {os.path.abspath(output_html)}")
    webbrowser.open("file://" + os.path.abspath(output_html))

if __name__ == "__main__":
    pdf_file = find_pdf_file()
    print(f"Found PDF: {pdf_file}")
    all_exercises = extract_all_exercises(pdf_file)
    print(f"Total Exercise Frames Extracted: {len(all_exercises)}")
    build_html_app(all_exercises)