# Document Cropper API

FastAPI で PDF / 画像から複数の書類を自動検出し、トリミング＋台形補正した画像を data URI として返す API です。Cloud Run を想定した Dockerfile を同梱しています。

## 必要環境
- Python 3.11
- (Docker 利用時) Docker 20 以降

## ローカルでの実行手順
1. 依存関係をインストールします。
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Uvicorn でサーバーを起動します。
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
3. `http://127.0.0.1:8000/docs` を開くと Swagger UI で API を試せます。

## エンドポイント仕様
- メソッド: `POST`
- パス: `/crop-documents`
- Content-Type: `multipart/form-data`
- フィールド:
  - `file`: 必須。PDF または JPG/PNG 画像。
  - `image_format`: 任意。`png` (デフォルト) または `jpeg`。返却画像のフォーマットを指定。

### 成功レスポンス例
```json
{
  "documents": [
    {
      "page_index": 0,
      "doc_index": 0,
      "mime_type": "image/png",
      "width": 1240,
      "height": 1754,
      "data_uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```

### エラーレスポンス例
```json
{"error": "Unsupported file type. Only PDF and image are allowed."}
```

## cURL での利用例
```bash
curl -X POST \
  -F "file=@./sample.pdf" \
  -F "image_format=png" \
  http://127.0.0.1:8000/crop-documents
```

## Docker ビルドと実行
1. イメージをビルドします。
   ```bash
   docker build -t document-cropper .
   ```
2. コンテナを起動します (Cloud Run と同様に `PORT` 環境変数を使用)。
   ```bash
   docker run -it --rm -p 8080:8080 -e PORT=8080 document-cropper
   ```
3. `http://127.0.0.1:8080/docs` にアクセスすると API を試せます。

## Cloud Run デプロイのポイント
- コンテナイメージを Container Registry / Artifact Registry にプッシュ後、Cloud Run で `PORT` 環境変数を指定してデプロイしてください。
- `Dockerfile` は必要なネイティブライブラリを含むマルチステージ構成で、実行時は `uvicorn main:app --host 0.0.0.0 --port $PORT` を使います。
