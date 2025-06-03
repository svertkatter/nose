# my_project/Dockerfile

# 1) ベースイメージに Python 3.10-slim を指定
FROM python:3.10-slim

# 2) 必要な OS 依存ライブラリをインストール
#    - libgl1-mesa-glx を追加して、libGL.so.1 を提供
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
    libgl1-mesa-glx \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# 3) 作業ディレクトリを /app にして、ソース一式をコピー
WORKDIR /app
COPY . /app

# 4) Python ライブラリをインストール
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5) デフォルトコマンド: app/main.py を実行
CMD ["python", "app/main.py"]
