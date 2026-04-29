import chess
import chess.engine
import chess.pgn
import io
import math

"""
Automated Data Labeling Script.
Uses Stockfish (Multi-PV) to evaluate historical positions and labels them
with a 'complexity score' based on engine variance and forced moves.
Generates the target variables for the neural network training.
"""

STOCKFISH_PATH = "stockfish-windows-x86-64-avx2.exe"


def parse_score_to_wp(score_obj):
    # jeśli jest mat, szansa wynosi (1.0) dla wygrywającego i (0.0) dla przegrywającego
    if score_obj.is_mate():
        return 1.0 if score_obj.mate() > 0 else 0.0

    cp = score_obj.score()
    cp = max(-2000, min(2000, cp))

    # sigmoid: przelicza centypiony na szansę wygranej
    # wzór: 1 / (1 + e^(-0.004 * cp))
    wp = 1 / (1 + math.exp(-0.004 * cp))
    return wp


def analyze_game(game_pgn, engine_path, limit_sec=0.1, threshold=0.2):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    game = chess.pgn.read_game(io.StringIO(game_pgn))
    board = game.board()

    dataset = []

    info = engine.analyse(board, chess.engine.Limit(time=limit_sec))
    prev_wp = parse_score_to_wp(info["score"].white())

    for move in game.mainline_moves():
        is_white_turn = board.turn
        fen_before = board.fen()

        board.push(move)

        if board.is_checkmate():
            # jeśli jest mat, gracz który właśnie wykonał ruch ma (1.0) szans
            current_wp = 1.0 if is_white_turn else 0.0
        else:
            # w przeciwnym razie pytamy Stockfisha
            info = engine.analyse(board, chess.engine.Limit(time=limit_sec))
            current_wp = parse_score_to_wp(info["score"].white())

        # obliczamy spadek szans
        if is_white_turn:
            wp_drop = prev_wp - current_wp
        else:
            wp_drop = current_wp - prev_wp

        target = 1 if wp_drop >= threshold else 0

        dataset.append({
            "fen": fen_before,
            "move_played": move.uci(),
            "wp_drop": round(wp_drop * 100, 1),
            "is_blunder": target
        })

        prev_wp = current_wp

    engine.quit()
    return dataset
