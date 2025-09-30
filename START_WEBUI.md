# 🚀 LLM比較WebUI 起動手順

## ✅ 完了した実装

### 🎯 **オプションB: FastAPI + Streamlit** を完全実装

**アーキテクチャ**:
```
┌─────────────────────────────────────────┐
│         Streamlit Frontend              │
│         (localhost:8501)                │
├─────────────────────────────────────────┤
│          FastAPI Backend               │
│         (localhost:8000)                │
├─────────────────────────────────────────┤
│    ベースモデル    │ ファインチューニング │
│  TinyLlama-1.1B   │      済みモデル     │
│                   │      (LoRA)        │
└─────────────────────────────────────────┘
```

## 🚀 起動方法（推奨）

### 方法1: 自動起動スクリプト
```bash
cd /home/ubuntu/Documents/projects/llm_finetuning
source webui_env/bin/activate
python run_webui.py
```

### 方法2: 手動起動（デバッグ用）

#### 1. FastAPIサーバー起動
```bash
# ターミナル1
cd /home/ubuntu/Documents/projects/llm_finetuning
source webui_env/bin/activate
python -m uvicorn webui.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Streamlitアプリ起動
```bash
# ターミナル2
cd /home/ubuntu/Documents/projects/llm_finetuning
source webui_env/bin/activate
streamlit run streamlit_app.py --server.port 8501
```

## 🌐 アクセスURL

- **🖥️ メインUI**: http://localhost:8501
- **📚 API ドキュメント**: http://localhost:8000/api/docs
- **💓 ヘルスチェック**: http://localhost:8000/api/health

## 📁 実装されたファイル構成

```
webui/
├── __init__.py
├── app.py                     # Streamlitメインアプリ（モジュール版）
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPIメインファイル
│   ├── models.py             # モデル管理サービス
│   └── inference.py          # 推論サービス
├── components/
│   ├── __init__.py
│   ├── styles.py             # モダンUIスタイル
│   └── comparison.py         # 比較表示コンポーネント
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # 評価指標計算
│   └── api_client.py         # FastAPIクライアント
└── config/
    ├── __init__.py
    └── settings.py           # 設定管理

# ルートファイル
├── streamlit_app.py          # Streamlitスタンドアロン版
├── run_webui.py              # 統合起動スクリプト
├── webui_requirements.txt    # 依存関係
├── WEBUI_README.md           # 詳細ドキュメント
└── START_WEBUI.md            # 起動手順（このファイル）
```

## ✨ 実装された機能

### 🎨 **モダンで洗練されたUI**
- グラデーション背景とカードデザイン
- レスポンシブレイアウト
- アニメーション効果
- ダークモード対応

### ⚡ **高性能アーキテクチャ**
- FastAPI非同期処理
- 並列モデル推論
- メモリ効率化
- エラーハンドリング

### 📊 **包括的な分析機能**
- リアルタイム比較
- 日本語特化メトリクス
- インタラクティブチャート
- 履歴管理

### 🔧 **システム機能**
- ヘルスチェック
- モデル状態管理
- パラメータ調整
- プリセット質問

## 🎛️ 使用方法

### 1. **初回セットアップ**
1. WebUIにアクセス: http://localhost:8501
2. サイドバーの「📥 モデル読み込み」をクリック
3. 読み込み完了を待機（初回は5-10分）

### 2. **比較実行**
1. プリセット質問選択 または カスタム質問入力
2. 推論パラメータ調整（オプション）
3. 「🚀 比較実行」ボタンクリック

### 3. **結果分析**
- **📝 モデル出力比較**: 並列表示
- **⚡ パフォーマンス指標**: 速度・文字数
- **📊 詳細メトリクス**: 日本語品質分析
- **📈 可視化チャート**: インタラクティブグラフ

## 🔍 トラブルシューティング

### API接続エラー
```bash
# プロセス確認
ps aux | grep uvicorn

# 手動でAPIサーバー起動
python -m uvicorn webui.api.main:app --host 0.0.0.0 --port 8000
```

### モデル読み込みエラー
```bash
# モデルファイル確認
ls -la japanese_finetuned_model/

# メモリ確認
free -h
```

### Streamlitエラー
```bash
# キャッシュクリア
streamlit cache clear

# 手動起動
streamlit run streamlit_app.py
```

## 🎯 次のステップ

### 1. **基本動作確認**
- [ ] WebUI起動成功
- [ ] API接続確認
- [ ] モデル読み込み成功
- [ ] 比較実行成功

### 2. **機能テスト**
- [ ] プリセット質問テスト
- [ ] カスタム質問テスト
- [ ] パラメータ調整テスト
- [ ] 結果分析確認

### 3. **パフォーマンス最適化**
- [ ] 推論速度測定
- [ ] メモリ使用量確認
- [ ] UI応答性チェック

## 🎉 完成！

**FastAPI + Streamlit による洗練されたLLM比較WebUIの実装が完了しました！**

- ✅ **モダンなデザイン**: シンプルで美しいUI
- ✅ **高性能**: 非同期並列処理
- ✅ **包括的分析**: 詳細な評価指標
- ✅ **使いやすさ**: 直感的な操作

**お疲れ様でした！🚀**
