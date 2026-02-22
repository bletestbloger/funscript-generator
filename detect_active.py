import cv2
import numpy as np
from scipy.fft import fft
from collections import deque
import csv
import os
import json

# === 設定（デフォルト値） ===
DEFAULT_OUTPUT_CSV = 'object_trajectory_ms_with_periodicity_xy.csv'
DEFAULT_OUTPUT_FUNSCRIPT = 'generated.funscript'
DEFAULT_OUTPUT_FUNSCRIPT_CSV = 'funscript_actions.csv'

DEFAULT_OUTPUT_INTERVAL_SEC = 0.1
DEFAULT_MIN_CONTOUR_AREA = 500
DEFAULT_HISTORY_MAXLEN = 1000
DEFAULT_FREQUENCY_THRESHOLD = 0.5
DEFAULT_MIN_HISTORY_FOR_FFT = 10

# Funscript関連設定
DEFAULT_POS_COLUMN = 'Y座標(px)'  # 'Y座標(px)', 'X座標(px)', '移動量(px)', '原点からの距離(px)'
DEFAULT_MIN_POS = 0
DEFAULT_MAX_POS = 100


def process_video(
    video_path: str,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    output_funscript: str = DEFAULT_OUTPUT_FUNSCRIPT,
    output_funscript_csv: str = DEFAULT_OUTPUT_FUNSCRIPT_CSV,
    output_interval_sec: float = DEFAULT_OUTPUT_INTERVAL_SEC,
    min_contour_area: int = DEFAULT_MIN_CONTOUR_AREA,
    history_maxlen: int = DEFAULT_HISTORY_MAXLEN,
    frequency_threshold: float = DEFAULT_FREQUENCY_THRESHOLD,
    min_history_for_fft: int = DEFAULT_MIN_HISTORY_FOR_FFT,
    pos_column: str = DEFAULT_POS_COLUMN,
    min_pos: int = DEFAULT_MIN_POS,
    max_pos: int = DEFAULT_MAX_POS
) -> str:
    """
    動画を解析してFunscriptを生成するメイン関数
    Returns:
        生成されたfunscriptファイルのフルパス
    Raises:
        ValueError: 動画が開けない場合など
        Exception: その他の処理エラー
    """
    if not os.path.isfile(video_path):
        raise ValueError(f"動画ファイルが見つかりません: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"動画ファイルを開けませんでした: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_time_sec = 1.0 / fps

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True
    )

    position_history = deque(maxlen=history_maxlen)
    trajectory_data = []
    last_pos = None
    last_output_ms = -int(output_interval_sec * 1000)
    current_time_sec = 0.0

    print("動画解析開始...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)

        fg_mask = cv2.medianBlur(fg_mask, 5)
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.erode(fg_mask, None, iterations=2)
        fg_mask = cv2.dilate(fg_mask, None, iterations=4)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_pos = None

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > min_contour_area:
                x, y, w, h = cv2.boundingRect(largest)
                center_x = x + w // 2
                center_y = y + h // 2
                current_pos = (center_x, center_y)
                position_history.append(current_pos)

        # 周期検出（履歴が十分溜まっている場合のみ）
        is_periodic = detect_periodicity(list(position_history), frequency_threshold, min_history_for_fft)

        current_ms = int(current_time_sec * 1000)
        if current_ms >= last_output_ms + int(output_interval_sec * 1000) and is_periodic:
            if current_pos is not None:
                distance_moved = 0.0
                if last_pos is not None:
                    dx = current_pos[0] - last_pos[0]
                    dy = current_pos[1] - last_pos[1]
                    distance_moved = np.sqrt(dx**2 + dy**2)

                distance_from_origin = np.sqrt(current_pos[0]**2 + current_pos[1]**2)

                trajectory_data.append([
                    current_ms,
                    current_pos[0],
                    current_pos[1],
                    round(distance_moved, 2),
                    round(distance_from_origin, 2)
                ])

                last_pos = current_pos
                last_output_ms = current_ms

        current_time_sec += frame_time_sec

    cap.release()
    print(f"動画解析完了。{len(trajectory_data)}件のデータを取得しました。")

    if not trajectory_data:
        raise ValueError("有効な軌跡データが取得できませんでした")

    # CSV保存（軌跡データ）
    output_dir = os.path.dirname(output_funscript) or "."
    os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['時間(ms)', 'X座標(px)', 'Y座標(px)', '移動量(px)', '原点からの距離(px)'])
        writer.writerows(trajectory_data)

    # Funscript生成
    times_ms = [row[0] for row in trajectory_data]
    pos_values_raw = [
        row[2] if pos_column == 'Y座標(px)' else
        row[1] if pos_column == 'X座標(px)' else
        row[3] if pos_column == '移動量(px)' else
        row[4] for row in trajectory_data
    ]

    col_min = min(pos_values_raw)
    col_max = max(pos_values_raw)

    if col_min == col_max:
        print("選択列に変化がありません。posを50に固定します。")
        normalized_pos = [50] * len(pos_values_raw)
    else:
        normalized_pos = [
            min_pos + (v - col_min) / (col_max - col_min) * (max_pos - min_pos)
            for v in pos_values_raw
        ]

    actions = []
    funscript_csv_rows = []
    for i in range(len(times_ms)):
        at_time = int(times_ms[i])
        pos_value = int(round(normalized_pos[i]))
        actions.append({"at": at_time, "pos": pos_value})
        funscript_csv_rows.append([at_time, pos_value])

    # Funscript JSON出力
    funscript_data = {
        "version": "1.0",
        "inverted": False,
        "range": 90,
        "actions": actions
    }

    with open(output_funscript, 'w', encoding='utf-8') as f:
        json.dump(funscript_data, f, indent=4)

    # Funscript用CSV出力
    with open(output_funscript_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['at (ms)', 'pos (0-100)'])
        writer.writerows(funscript_csv_rows)

    print(f"Funscriptファイルを生成しました → {os.path.abspath(output_funscript)}")
    print(f"生成アクション数: {len(actions)}")

    return os.path.abspath(output_funscript)


def detect_periodicity(positions, frequency_threshold=DEFAULT_FREQUENCY_THRESHOLD,
                       min_history_for_fft=DEFAULT_MIN_HISTORY_FOR_FFT):
    if len(positions) < min_history_for_fft:
        return False

    x_positions = np.array([pos[0] for pos in positions])
    fft_x = fft(x_positions)
    freqs_x = np.abs(fft_x)[1:len(fft_x)//2]
    periodic_x = np.max(freqs_x) > frequency_threshold * np.mean(freqs_x)

    y_positions = np.array([pos[1] for pos in positions])
    fft_y = fft(y_positions)
    freqs_y = np.abs(fft_y)[1:len(fft_y)//2]
    periodic_y = np.max(freqs_y) > frequency_threshold * np.mean(freqs_y)

    return periodic_x or periodic_y


# 単体テスト用（コマンドラインで直接実行した場合のみ）
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("使い方: python detect_active.py <動画ファイルパス>")
        sys.exit(1)

    test_video = sys.argv[1]
    try:
        generated_file = process_video(test_video)
        print(f"生成完了: {generated_file}")
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)