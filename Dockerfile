# my_project/Dockerfile

# ─────────────────────────────────────────────────
# 1) ベースイメージに「python:3.10-slim」を指定
FROM python:3.10-slim

# ─────────────────────────────────────────────────
# 2) OS 依存ライブラリをインストール
#    - build-essential, cmake: Mediapipe のビルド依存として
#    - libglib2.0-0, libsm6, libxrender1, libxext6: OpenCV の動画/GUI
#    - libsdl2-*         : pygame の依存
#    - libavformat-dev, libavcodec-dev, libswscale-dev: OpenCV の動画 I/O
#    - libtiff-dev, libjpeg-dev: 画像入出力用
#    - x11-apps          : X11 クライアント（GUI 表示用）
#    - git, curl         : ソース取得・補助ツール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    libtiff-dev \
    libjpeg-dev \
    x11-apps \
  && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────
# 3) 作業ディレクトリを /app に設定し、ソース一式をコピー
WORKDIR /app
COPY . /app

# ─────────────────────────────────────────────────
# 4) Python ライブラリをインストール
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────
# 5) デフォルトコマンド: app/main.py を実行
CMD ["python", "app/main.py"]
