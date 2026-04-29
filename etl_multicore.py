import zstandard as zstd
import chess.pgn
import io
import pandas as pd
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from blunder_detector import analyze_game, STOCKFISH_PATH

"""
ETL Pipeline for Chess Data (Multiprocessing Version).
Extracts chess positions from large PGN files, filters games by player Elo (>2000), 
and converts board states into numerical features for machine learning.
Optimized using Python's multiprocessing module for large datasets.
"""

INPUT_FILE = "data/lichess_db_standard_rated_2019-03.pgn.zst"
OUTPUT_DIR = "data/processed/"
GAMES_PER_CHUNK = 50
TOTAL_GAMES_TO_PROCESS = 10000

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def worker_task(pgn_strings, chunk_id):
    chunk_results = []
    for pgn_str in pgn_strings:
        results = analyze_game(pgn_str, STOCKFISH_PATH)
        chunk_results.extend(results)

    if chunk_results:
        df = pd.DataFrame(chunk_results)
        file_name = f"chunk_{chunk_id}.parquet"
        df.to_parquet(os.path.join(OUTPUT_DIR, file_name), engine='pyarrow')
    return chunk_id


if __name__ == "__main__":
    # ustawiamy połowę mocy procesora
    num_workers = max(1, multiprocessing.cpu_count() // 2)

    tasks = []
    current_chunk_pgns = []
    games_filtered = 0
    chunk_count = 0

    with open(INPUT_FILE, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')

            while games_filtered < TOTAL_GAMES_TO_PROCESS:
                headers = chess.pgn.read_headers(text_stream)
                if headers is None: break

                w_elo = int(headers.get("WhiteElo", 0) or 0)
                b_elo = int(headers.get("BlackElo", 0) or 0)

                if 2000 <= w_elo <= 2400 and 2000 <= b_elo <= 2400:
                    game = chess.pgn.read_game(text_stream)
                    if game is None: break

                    current_chunk_pgns.append(str(game))
                    games_filtered += 1

                    if len(current_chunk_pgns) >= GAMES_PER_CHUNK:
                        tasks.append((current_chunk_pgns, chunk_count))
                        current_chunk_pgns = []
                        chunk_count += 1

            if current_chunk_pgns:
                tasks.append((current_chunk_pgns, chunk_count))

    completed = 0
    total_tasks = len(tasks)


    def progress_callback(future):
        global completed
        completed += 1

    # przetwarzanie wielowątkowe
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for pgn_list, c_id in tasks:
            future = executor.submit(worker_task, pgn_list, c_id)
            future.add_done_callback(progress_callback)
