# honkai-ocr-3rd

崩壊3rdの実況動画から、会話ウィンドウを検出してダイアログを文字起こしする実験用ツールです。

## 開発環境

Nix flakes と `flake-parts` を使っています。

```sh
nix develop
```

シェルに入ると、Python 3.11、uv、OpenCV、ffmpeg、yt-dlp、Tesseract（日本語データ）が利用できます。Python依存関係はuvが`.venv`に同期します。

## 動画の取得

動画全体は保存せず、冒頭5分だけを取得する例です。

```sh
mkdir -p data/raw
yt-dlp \
  --download-sections '*0-300' \
  --force-keyframes-at-cuts \
  --no-playlist \
  -f 298 \
  -o 'data/raw/honkai-opening.%(ext)s' \
  'https://youtu.be/HoeDSGX81FU?si=Uu64gw-R76zcje4L'
```

## 実行

1枚の画像で、検出枠・OCR切り出し・OCR結果を確認できます。

```sh
uv run honkai-ocr image data/frames/sample-005.jpg \
  --output-dir data/output/image-005
```

動画はサンプル間隔を指定して処理します。出力先にはイベントごとの代表フレーム、`name.png`、`body.png`、`events.json`、確認用の`review.html`が作られます。

```sh
uv run honkai-ocr video data/raw/honkai-opening.mp4 \
  --output-dir data/output/opening \
  --sample-fps 2 \
  --max-seconds 300
```

`--sample-fps 2` は0.5秒ごとのサンプリングです。台詞の取りこぼしを減らしたい場合は、この値を使用してください。

会話枠を検出できないフレームはOCRしません。`events.json`の`needs_review`が付いたイベントは、保存された切り出し画像を見ながら手動修正する想定です。自動結果は`text`、検出信頼度や元フレームは別フィールドに残ります。

## 現在の検出方針

- 画面下部の淡色パネルの上端・下端を横方向の境界として検出
- 会話枠の高さがほぼ一定という事前知識を利用
- 右寄り・全体に広がる枠を同じ矩形形式で扱う
- 立ち絵が左側に重なる場合は、濃い画素の列分布から本文開始位置を推定
- 本文右下の進行用の逆三角を小さな暗色成分として検出し、OCR前にマスク
- 複数の前処理画像をTesseractに渡し、最もよい候補を採用

この段階では機械学習モデルは使っていません。検出に失敗する場面をラベル化してから、必要に応じてセグメンテーションまたは物体検出モデルを追加します。

## テスト

```sh
uv run pytest
```

動画と生成物はリポジトリに含めず、`data/raw`、`data/frames`、`data/output`で管理します。
