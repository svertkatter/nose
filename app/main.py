import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
import os
import sys

# ─────────────────────────────────────────────────────────────────
# 設定：各モードの切り替え時間や音量などの定数
# ─────────────────────────────────────────────────────────────────

# 一人モードで鼻変形開始までの遅延（秒）
ONE_PERSON_DELAY = 3.0

# ２人以上での反転サイクル（秒）
INVERT_CYCLE = 90.0

# 鼻の最小・最大スケール（倍率）。1.0 = 元サイズ
NOSE_MIN_SCALE = 1.0
NOSE_MAX_SCALE = 3.0

# モード切り替え用の人数閾値
MODE_ONE_PERSON = 1
MODE_TWO_PERSON = 2
MODE_GROUP = 3  # 3人以上

# 音声ファイルのパス（相対パス）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LAUGH_FILES = {
    "giggle": os.path.join(ASSETS_DIR, "laugh_giggle.wav"),
    "chuckle": os.path.join(ASSETS_DIR, "laugh_chuckle.wav"),
    "big": os.path.join(ASSETS_DIR, "laugh_big.wav"),
}

# 笑顔強度の閾値（smile_ratio がこの値以上で「笑っている」とみなすか）
SMILE_THRESHOLD = 0.1

# ─────────────────────────────────────────────────────────────────
# 初期化: MediaPipe FaceMesh, OpenCV, Pygame 音響
# ─────────────────────────────────────────────────────────────────

# MediaPipe の顔メッシュモデルを準備
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=8,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# OpenCV ウィンドウ名
WINDOW_NAME = "Interactive Nose Mirror"

# Pygame ミキサー初期化（音響用）
pygame.mixer.init()

# 笑い声のチャンネルをロード・プレイ（ループ再生）
laugh_channels = {}
for layer, path in LAUGH_FILES.items():
    if not os.path.isfile(path):
        print(f"[ERROR] Missing sound file: {path}")
        sys.exit(1)
    sound = pygame.mixer.Sound(path)
    channel = sound.play(loops=-1)  # ループ再生
    channel.set_volume(0.0)         # 最初は音量0
    laugh_channels[layer] = (sound, channel)

# 鼻画像をロードして透過マスクを作成
nose_img_path = os.path.join(ASSETS_DIR, "nose.png")
if not os.path.isfile(nose_img_path):
    print(f"[ERROR] Missing nose image: {nose_img_path}")
    sys.exit(1)
nose_img = cv2.imread(nose_img_path, cv2.IMREAD_UNCHANGED)  # RGBA 透過PNG

# 透過PNG からアルファチャネルマスクを分離
if nose_img.shape[2] == 4:
    nose_alpha = nose_img[:, :, 3] / 255.0
    nose_rgb = nose_img[:, :, :3]
else:
    # 透過がない場合はそのまま BGR として扱い、アルファはすべて1とする
    nose_alpha = np.ones(nose_img.shape[:2], dtype=float)
    nose_rgb = nose_img

# マスクの反転（鼻以外を黒くするマスク）
nose_mask = nose_alpha
inverse_nose_mask = 1.0 - nose_mask

# ─────────────────────────────────────────────────────────────────
# ヘルパー関数
# ─────────────────────────────────────────────────────────────────

def overlay_transparent(background, overlay, x, y, scale=1.0):
    """
    background 上に透過画像 overlay を (x,y) の位置に scale 倍のサイズで重ねる。
    overlay は BGR または BGRA (透過含む) 画像を想定し、
    透過チャンネルがある場合はアルファブレンドする。
    """
    h, w = overlay.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w <= 0 or new_h <= 0:
        return background

    overlay_resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if overlay_resized.shape[2] == 4:
        alpha = overlay_resized[:, :, 3] / 255.0
        color = overlay_resized[:, :, :3]
    else:
        alpha = np.ones((new_h, new_w), dtype=float)
        color = overlay_resized

    h_bg, w_bg = background.shape[:2]
    if x < 0 or y < 0 or x + new_w > w_bg or y + new_h > h_bg:
        # 画面外にはみ出すときはクリッピング
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + new_w, w_bg)
        y2 = min(y + new_h, h_bg)

        overlay_x1 = max(-x, 0)
        overlay_y1 = max(-y, 0)
        overlay_x2 = overlay_x1 + (x2 - x1)
        overlay_y2 = overlay_y1 + (y2 - y1)

        bg_roi = background[y1:y2, x1:x2]
        ol_roi_color = color[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        ol_roi_alpha = alpha[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

        for c in range(3):
            bg_roi[:, :, c] = (ol_roi_alpha * ol_roi_color[:, :, c] +
                               (1.0 - ol_roi_alpha) * bg_roi[:, :, c])
        background[y1:y2, x1:x2] = bg_roi
        return background

    # 通常のケース
    for c in range(3):
        background[y:y + new_h, x:x + new_w, c] = (
            alpha * color[:, :, c] +
            (1.0 - alpha) * background[y:y + new_h, x:x + new_w, c]
        )
    return background


def calculate_smile_ratio(landmarks, img_w, img_h):
    """
    MediaPipe FaceMesh の landmarks から "笑顔強度" を数値化する簡易的指標を返す。
    - 口の左右端 (lm 61, 291) の距離を計算し、 
    - 顔幅（左右の頬付近、lm 234 と lm 454 など）の距離で正規化して返す。
    """
    # landmark 61 (左口角), 291 (右口角)
    lmk_left = landmarks[61]
    lmk_right = landmarks[291]
    x_left, y_left = int(lmk_left.x * img_w), int(lmk_left.y * img_h)
    x_right, y_right = int(lmk_right.x * img_w), int(lmk_right.y * img_h)
    mouth_width = np.hypot(x_right - x_left, y_right - y_left)

    # 顔幅の指標として頬付近を使う（例：lm 234 と lm 454）
    lmk_cheek_l = landmarks[234]
    lmk_cheek_r = landmarks[454]
    x_cl, y_cl = int(lmk_cheek_l.x * img_w), int(lmk_cheek_l.y * img_h)
    x_cr, y_cr = int(lmk_cheek_r.x * img_w), int(lmk_cheek_r.y * img_h)
    face_width = np.hypot(x_cr - x_cl, y_cr - y_cl)

    if face_width == 0:
        return 0.0
    return mouth_width / face_width


# ─────────────────────────────────────────────────────────────────
# メインループ
# ─────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Mac では通常 VideoCapture(0) で OK
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    start_time = time.time()
    last_invert = start_time

    # モードと状態の初期化
    mode = MODE_ONE_PERSON
    one_person_start = None   # 一人モードのタイマー開始時刻
    nose_scale_person = {}    # {face_id: current_scale} for 1人 or 2人
    invert_phase = False      # 反転中かどうか

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # FaceMesh で顔検出・ランドマーク取得
        results = face_mesh.process(rgb_frame)
        faces = []
        if results.multi_face_landmarks:
            for idx, face_lm in enumerate(results.multi_face_landmarks):
                # 顔ごとのランドマークを一旦配列化
                landmarks = face_lm.landmark
                # 顔のバウンディングボックス（ランドマークの min/max で近似）
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                box = (int(x_min * img_w), int(y_min * img_h),
                       int((x_max - x_min) * img_w), int((y_max - y_min) * img_h))
                faces.append((idx, landmarks, box))

        # 検出した顔の人数でモードを切り替え
        n_faces = len(faces)
        previous_mode = mode
        if n_faces == 0:
            mode = MODE_ONE_PERSON
        elif n_faces == 1:
            mode = MODE_ONE_PERSON
        elif n_faces == 2:
            mode = MODE_TWO_PERSON
        else:
            mode = MODE_GROUP

        current_time = time.time()
        elapsed_since_invert = current_time - last_invert
        # 90秒たったら反転フラグをオン／オフトグル
        if elapsed_since_invert >= INVERT_CYCLE:
            invert_phase = not invert_phase
            last_invert = current_time
            # 反転時に各顔の鼻スケールをリセットしておく
            nose_scale_person.clear()
            one_person_start = None

        # 各モードごとの処理
        if mode == MODE_ONE_PERSON:
            # １人モード
            if len(faces) == 1:
                idx, landmarks, box = faces[0]
                # 一人モードに入った瞬間にタイマー開始
                if one_person_start is None:
                    one_person_start = current_time

                elapsed = current_time - one_person_start
                # 3秒経過後、鼻のスケールを徐々に増加させる
                if elapsed >= ONE_PERSON_DELAY:
                    t = min((elapsed - ONE_PERSON_DELAY) / (INVERT_CYCLE), 1.0)
                    scale = NOSE_MIN_SCALE + t * (NOSE_MAX_SCALE - NOSE_MIN_SCALE)
                else:
                    scale = NOSE_MIN_SCALE

                nose_scale_person[idx] = scale

                # 音響：常に微かな笑い声を小ボリュームで再生
                laugh_channels["giggle"][1].set_volume(0.2)
                laugh_channels["chuckle"][1].set_volume(0.0)
                laugh_channels["big"][1].set_volume(0.0)

                # 表情反応
                smile_ratio = calculate_smile_ratio(landmarks, img_w, img_h)
                if smile_ratio > SMILE_THRESHOLD:
                    # 笑顔：鼻の変形を加速し、giggle の音量を減少
                    scale = min(scale + 0.02, NOSE_MAX_SCALE)
                    nose_scale_person[idx] = scale
                    vol = max(0.2 - (smile_ratio - SMILE_THRESHOLD), 0.0)
                    laugh_channels["giggle"][1].set_volume(vol)
                else:
                    # 困惑・悲しみ・怒り？（無感情も含む）
                    if scale > NOSE_MIN_SCALE:
                        # scale は維持。giggle はそのまま
                        pass
                    # 無感情ならそのまま

                # 鼻の重ね描画
                # 鼻の基準は landmark 1（鼻先）を使う
                nose_lm = landmarks[1]
                x_nose = int(nose_lm.x * img_w - (nose_img.shape[1] * scale) / 2)
                y_nose = int(nose_lm.y * img_h - (nose_img.shape[0] * scale) / 2)
                frame = overlay_transparent(frame, nose_img, x_nose, y_nose, scale)

        elif mode == MODE_TWO_PERSON:
            # ２人モード
            # まず顔を前後でソート（バウンディングボックスの面積が大きいほうを「前」にする）
            faces_sorted = sorted(faces, key=lambda f: f[2][2] * f[2][3], reverse=True)
            front_idx, front_lm, front_box = faces_sorted[0]
            back_idx, back_lm, back_box = faces_sorted[1]

            # 前の人の鼻スケールを保持。最初は最小値
            if front_idx not in nose_scale_person:
                nose_scale_person[front_idx] = NOSE_MIN_SCALE
            scale_front = nose_scale_person[front_idx]

            # 後ろの人の笑顔強度を取得
            smile_back = calculate_smile_ratio(back_lm, img_w, img_h)

            # インタラクション：後ろの人が笑うほど、前の人の鼻が変形
            if smile_back > SMILE_THRESHOLD:
                scale_front = min(scale_front + smile_back * 0.1, NOSE_MAX_SCALE)
                nose_scale_person[front_idx] = scale_front

                # 音響：後ろの人の笑顔強度に応じて多層的にボリューム調整
                laugh_channels["giggle"][1].set_volume(0.0)
                laugh_channels["chuckle"][1].set_volume(min(smile_back * 0.5, 1.0))
                laugh_channels["big"][1].set_volume(max((smile_back - 0.5) * 2, 0.0))
            else:
                # 後ろの人が真面目：笑い声は減少（全体）
                for layer in laugh_channels:
                    cur_vol = laugh_channels[layer][1].get_volume()
                    laugh_channels[layer][1].set_volume(max(cur_vol - 0.01, 0.0))

            # 前の人の鼻を重ね描画
            nose_lm = front_lm[1]
            x_nose = int(nose_lm.x * img_w - (nose_img.shape[1] * scale_front) / 2)
            y_nose = int(nose_lm.y * img_h - (nose_img.shape[0] * scale_front) / 2)
            frame = overlay_transparent(frame, nose_img, x_nose, y_nose, scale_front)

            # 後ろの人は変形なしで映すだけ（何もしない）

        else:
            # MODE_GROUP：3人以上
            # 最大面積を持つ顔を「被笑われ者」として、それ以外を「笑う側」とみなす
            faces_sorted = sorted(faces, key=lambda f: f[2][2] * f[2][3], reverse=True)
            target_idx, target_lm, target_box = faces_sorted[0]
            laugher_faces = faces_sorted[1:]

            # 被笑われ者のスケールを保持。最初は最小
            if target_idx not in nose_scale_person:
                nose_scale_person[target_idx] = NOSE_MIN_SCALE
            scale_target = nose_scale_person[target_idx]

            # 笑う側の笑顔強度を合算
            sum_smile = 0.0
            for idx_l, lm_l, box_l in laugher_faces:
                sum_smile += calculate_smile_ratio(lm_l, img_w, img_h)

            # 合算結果に応じて鼻を変形
            if sum_smile > SMILE_THRESHOLD:
                delta = (sum_smile / (len(laugher_faces))) * 0.15
                scale_target = min(scale_target + delta, NOSE_MAX_SCALE)
                nose_scale_person[target_idx] = scale_target

                # 音響：笑い声を各レイヤー適当にミックス
                avg_smile = sum_smile / len(laugher_faces)
                laugh_channels["giggle"][1].set_volume(0.0)
                laugh_channels["chuckle"][1].set_volume(min(avg_smile * 0.5, 1.0))
                laugh_channels["big"][1].set_volume(max((avg_smile - 0.5) * 2, 0.0))
            else:
                # 笑う人がいない or 真面目モード：音は減少
                for layer in laugh_channels:
                    cur_vol = laugh_channels[layer][1].get_volume()
                    laugh_channels[layer][1].set_volume(max(cur_vol - 0.01, 0.0))

            # 被笑われ者の鼻を重ね描画
            nose_lm = target_lm[1]
            x_nose = int(nose_lm.x * img_w - (nose_img.shape[1] * scale_target) / 2)
            y_nose = int(nose_lm.y * img_h - (nose_img.shape[0] * scale_target) / 2)
            frame = overlay_transparent(frame, nose_img, x_nose, y_nose, scale_target)

            # 他の人は通常の姿で映す（何もしない）

        # ウィンドウに描画＆キー制御
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC キーで終了
            break

    # クリーンアップ
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
