import streamlit as st
import chess
import chess.svg
import chess.engine
import numpy as np
from tensorflow.keras.models import load_model
import base64
import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

MODEL_PATH = "models/chess_complexity_model_v2.keras"
STOCKFISH_PATH = "stockfish-windows-x86-64-avx2.exe"

st.set_page_config(page_title="AI Chess Engine", layout="wide")


@st.cache_resource
def load_ai_model():
    return load_model(MODEL_PATH)


def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    piece_map = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        tensor[row, col, piece_map[piece.symbol()]] = 1
    return np.expand_dims(tensor, axis=0)


model = load_ai_model()

st.title("AI Chess Complexity Engine")
traxler_fen = "r1bqk2r/pppp1Npp/2n2n2/4p3/2B1P3/8/PPPP1bPP/RNBQK2R w KQkq - 0 5"
fen_input = st.text_input("Wklej pozycję (FEN):", traxler_fen)

try:
    board = chess.Board(fen_input)
    turn_text = "⚪ Ruch białych" if board.turn == chess.WHITE else "⚫ Ruch czarnych"

    # wagi z sieci neuronowej wyuczonej wczesniej
    tensor = fen_to_tensor(fen_input)
    ai_prediction = model.predict(tensor, verbose=0)[0][0]

    # 2. analiza Multi-PV Stockfishem (5 najlepszych linii)
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        # analizujemy 5 najlepszych ruchów
        results = engine.analyse(board, chess.engine.Limit(time=0.3), multipv=5)

        scores = []
        for res in results:
            # pobieramy punkty w centypionach
            s = res["score"].white().score(mate_score=10000)
            scores.append(s)

        # obliczamy odchylke ocen między 5 ruchami
        std_dev = np.std(scores) / 100.0

        # pasek Stockfisha dla najlepszego ruchu
        best_score = scores[0]
        wp = 1 / (1 + 10 ** (-best_score / 400))
        eval_text = f"+{best_score / 100:.2f}" if best_score >= 0 else f"{best_score / 100:.2f}"

    # dodajemy "bonus" za wariancję silnika do predykcji sieci.
    # jeśli silnik widzi, że jeden ruch jest super, a inne tragiczne, std_dev będzie wysokie.
    variance_boost = np.tanh(std_dev / 1.5) * 0.4  # max +40% do wyniku
    final_complexity = min(0.999, ai_prediction + variance_boost)

    _, col1, col2, col3, _ = st.columns([1.5, 0.5, 2.0, 0.5, 1.5])

    with col1:
        # Pasek Stockfisha
        sf_html = f"""
        <div style="display: flex; flex-direction: column; align-items: flex-end; width: 100%; padding-right: 15px;">
            <div style="display: flex; flex-direction: column; align-items: center; width: 60px;">
                <div style="font-size: 14px; font-weight: bold; margin-bottom: 5px;">Stockfish</div>
                <div style="height: 400px; width: 30px; background-color: #333333; border-radius: 4px; border: 1px solid #555; position: relative; overflow: hidden;">
                    <div style="position: absolute; bottom: 0; left: 0; width: 100%; height: {wp * 100}%; background-color: #ffffff; transition: height 0.5s;"></div>
                </div>
                <div style="font-size: 14px; margin-top: 5px; font-weight: bold;">{eval_text}</div>
            </div>
        </div>
        """
        st.markdown(sf_html, unsafe_allow_html=True)

    with col2:
        # Szachownica
        st.markdown(f"<h3 style='text-align: center; margin-bottom: 15px;'>{turn_text}</h3>", unsafe_allow_html=True)
        board_svg = chess.svg.board(board=board, size=400)
        b64 = base64.b64encode(board_svg.encode('utf-8')).decode("utf-8")
        st.markdown(
            rf'<div style="display: flex; justify-content: center;"><img src="data:image/svg+xml;base64,{b64}" /></div>',
            unsafe_allow_html=True)

    with col3:
        # Pasek poziomu skomplikowania
        comp_html = f"""
        <div style="display: flex; flex-direction: column; align-items: flex-start; width: 100%; padding-left: 15px;">
            <div style="display: flex; flex-direction: column; align-items: center; width: 60px;">
                <div style="font-size: 14px; font-weight: bold; margin-bottom: 5px;">Złożoność</div>
                <div style="height: 400px; width: 30px; background: linear-gradient(to top, #28a745, #dc3545); border-radius: 4px; border: 1px solid #555; position: relative;">
                    <div style="position: absolute; bottom: {final_complexity * 100}%; left: -5px; width: 40px; height: 4px; background-color: #ffffff; border: 1px solid #000; box-shadow: 0px 0px 4px rgba(0,0,0,0.8); transform: translateY(50%); transition: bottom 0.5s;"></div>
                </div>
                <div style="font-size: 16px; margin-top: 5px; font-weight: bold;">{final_complexity * 100:.1f}%</div>
            </div>
        </div>
        """
        st.markdown(comp_html, unsafe_allow_html=True)

except ValueError:
    st.error("Nieprawidłowy kod FEN.")