# 🤖 LLM ファインチューニング & 比較システム

**日本語特化LLMファインチューニングと洗練されたWebUI比較システム**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.118.0-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🎯 プロジェクト概要

このプロジェクトは、**TinyLlama-1.1B-Chat-v1.0**を三陽商会ドキュメントでファインチューニングし、ベースモデルとの比較を美しいWebUIで行うシステムです。

### ✨ 主な特徴

- **🔬 LoRAファインチューニング**: 効率的な日本語特化学習
- **🎨 モダンWebUI**: FastAPI + Streamlit による洗練されたインターフェース
- **⚡ 並列推論**: ベースモデルとファインチューニング済みモデルの同時比較
- **📊 詳細分析**: 日本語特化の評価指標とインタラクティブ可視化
- **📚 包括的ドキュメント**: 実装記録から結果分析まで完全網羅

## 🚀 クイックスタート

### 1. 環境セットアップ
```bash
# リポジトリクローン
git clone <repository-url>
cd llm_finetuning

# 仮想環境作成（WebUI用）
python3 -m venv webui_env
source webui_env/bin/activate
pip install -r webui_requirements.txt
```

### 2. WebUI起動
```bash
# 統合起動（推奨）
export PYTHONPATH=/path/to/llm_finetuning:$PYTHONPATH
python run_webui.py
```

### 3. アクセス
- **🖥️ メインUI**: http://localhost:8501
- **📚 API ドキュメント**: http://localhost:8000/api/docs
- **💓 ヘルスチェック**: http://localhost:8000/api/health

## 📁 プロジェクト構成

```
llm_finetuning/
├── 📊 dataset/                          # データセット
│   ├── japanese_qa_dataset.jsonl       # 日本語QAデータ（133サンプル）
│   ├── evaluation_questions.json       # 評価用質問
│   └── raw/                            # 元データ（三陽商会ドキュメント）
├── 🤖 japanese_finetuned_model/         # ファインチューニング済みモデル
├── 📈 results/                          # 評価結果
│   ├── baseline_evaluation.csv         # ベースライン評価
│   ├── comparison_evaluation.csv       # 比較評価
│   └── training_results.json           # 学習結果
├── 🔧 src/                             # コアスクリプト
│   ├── japanese_finetuning.py          # ファインチューニング実行
│   ├── baseline_evaluation.py          # ベースライン評価
│   ├── comparison_evaluation.py        # 比較評価
│   └── visualization.py               # 可視化
├── 🌐 webui/                           # WebUIシステム
│   ├── api/                           # FastAPIバックエンド
│   │   ├── main.py                    # メインAPI
│   │   ├── models.py                  # モデル管理
│   │   └── inference.py               # 推論サービス
│   ├── components/                    # UIコンポーネント
│   │   ├── styles.py                  # モダンスタイル
│   │   └── comparison.py              # 比較表示
│   ├── utils/                         # ユーティリティ
│   │   ├── metrics.py                 # 評価指標
│   │   └── api_client.py              # APIクライアント
│   └── config/                        # 設定管理
│       └── settings.py                # システム設定
├── 📱 streamlit_app.py                 # Streamlitメインアプリ
├── 🚀 run_webui.py                     # 統合起動スクリプト
├── 📋 demo_script.py                   # 統合デモスクリプト
└── 📚 ドキュメント/
    ├── LLM_ファインチューニング_実装記録.md
    ├── LLM_ファインチューニング_結果分析レポート.md
    ├── WEBUI_README.md                 # WebUI詳細ドキュメント
    └── START_WEBUI.md                  # 起動手順
```

## 🎯 使用方法

### WebUI操作手順

1. **🌐 アクセス**: http://localhost:8501
2. **📥 モデル読み込み**: サイドバーの「📥 モデル読み込み」をクリック
3. **💬 質問入力**: プリセット質問選択 または カスタム質問入力
4. **🚀 比較実行**: 「🚀 比較実行」ボタンで並列推論開始
5. **📊 結果分析**: 並列表示・メトリクス・チャートで詳細確認

### コマンドライン実行

```bash
# ファインチューニング実行
python src/japanese_finetuning.py

# ベースライン評価
python src/baseline_evaluation.py

# 比較評価
python src/comparison_evaluation.py

# 統合デモ
python demo_script.py
```

## 📊 技術仕様

### プロジェクト統計
- **総ファイル数**: 28,872ファイル
- **Pythonファイル**: 多数のスクリプト・モジュール
- **ドキュメント**: 36個のMarkdownファイル
- **設定・データ**: 140個のJSONファイル
- **プロジェクトサイズ**: 16GB
- **WebUIコード**: 272KB
- **結果データ**: 472KB
- **モデルサイズ**: 1.6GB

### モデル構成
- **ベースモデル**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **ファインチューニング手法**: LoRA (Low-Rank Adaptation)
- **データセット**: 三陽商会ドキュメントベース日本語QA（133サンプル）
- **学習時間**: 約17分（1033秒、CPU環境）
- **学習可能パラメータ**: 4,505,600（全体の0.4079%）
- **モデル読み込み時間**: 約6秒

### WebUIアーキテクチャ
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

### 評価指標
- **基本統計**: 文字数・単語数・文数・平均文長
- **日本語特性**: ひらがな・カタカナ・漢字使用率、文字種多様性
- **言語品質**: 敬語・丁寧語使用率、語彙豊富さ
- **パフォーマンス**: 推論時間、メモリ使用量

## 🎨 WebUI特徴

### モダンデザイン
- **🌈 グラデーション背景**: 美しいカラーリング
- **📱 レスポンシブ**: あらゆるデバイスに対応
- **✨ アニメーション**: スムーズな操作感
- **🌙 ダークモード対応**: 目に優しい表示

### 高性能機能
- **⚡ 非同期処理**: 並列モデル推論
- **📊 リアルタイム可視化**: Plotlyインタラクティブチャート
- **📚 履歴管理**: 過去の比較結果保存
- **🎛️ パラメータ調整**: Temperature・Top-p・Top-k制御

## 📈 実行結果

### ファインチューニング成果
- **学習時間**: 1033.33秒（約17分）
- **最終損失**: 1.739
- **評価損失**: 0.966
- **推論速度**: 3-4秒（並列処理）

### 比較結果例（実際の出力）
```json
{
  "status": "success",
  "prompt": "三陽商会について教えてください",
  "results": {
    "base_model": {
      "text": "3. Kumamoto-Shi (旭市)",
      "inference_time": 4.038146257400513,
      "character_count": 20,
      "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    },
    "finetuned_model": {
      "text": "四川大学商会です。<|end|>",
      "inference_time": 3.8927292823791504,
      "character_count": 16,
      "model_path": "./japanese_finetuned_model"
    }
  },
  "comparison": {
    "speed_improvement": 3.601082421293279,
    "length_difference": -4,
    "total_time": 4.038146257400513
  },
  "metrics": {
    "base_metrics": {
      "basic_stats": {
        "character_count": 20,
        "word_count": 3,
        "sentence_count": 1,
        "avg_sentence_length": 20.0
      },
      "japanese_characteristics": {
        "hiragana_count": 0,
        "katakana_count": 0,
        "kanji_count": 4,
        "character_diversity": 20.0
      }
    }
  }
}
```

### 動作実績
- **✅ 成功した推論**: 多数の比較推論が正常実行
- **⚡ 平均推論時間**: 3-4秒（並列処理）
- **🔄 自動リロード**: FastAPIの変更を自動検出
- **📊 リアルタイム分析**: 日本語特性・品質指標を即座に計算

## 🛠️ 開発・カスタマイズ

### 依存関係
```bash
# コア依存関係
torch>=2.8.0
transformers>=4.56.0
peft>=0.17.0
datasets>=4.1.0

# WebUI依存関係
fastapi>=0.118.0
streamlit>=1.50.0
plotly>=6.3.0
```

### 設定カスタマイズ
```python
# webui/config/settings.py
class Settings:
    BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    FINETUNED_MODEL_PATH = "./japanese_finetuned_model"
    MAX_LENGTH = 512
    TEMPERATURE = 0.7
    # ... その他設定
```

## 🔍 トラブルシューティング

### よくある問題

#### API接続エラー
```bash
# プロセス確認
ps aux | grep -E "(uvicorn|streamlit)"

# 手動起動
python -m uvicorn webui.api.main:app --host 0.0.0.0 --port 8000
```

#### モデル読み込みエラー
```bash
# モデルファイル確認
ls -la japanese_finetuned_model/

# メモリ確認
free -h
```

#### インポートエラー
```bash
# PYTHONPATHの設定
export PYTHONPATH=/path/to/llm_finetuning:$PYTHONPATH
```

## 📚 ドキュメント

- **[WebUI詳細ガイド](WEBUI_README.md)**: WebUIの詳細機能説明
- **[起動手順](START_WEBUI.md)**: ステップバイステップ起動ガイド
- **[実装記録](LLM_ファインチューニング_実装記録.md)**: 技術的実装詳細
- **[結果分析](LLM_ファインチューニング_結果分析レポート.md)**: 評価結果の詳細分析

## 🤝 コントリビューション

プルリクエストやイシュー報告を歓迎します！

### 開発ガイドライン
1. コードスタイル: Black + isort
2. テスト: pytest
3. ドキュメント: 日本語での詳細記述

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🎉 謝辞

- **Hugging Face**: Transformersライブラリ
- **Microsoft**: LoRA手法
- **Streamlit**: 美しいWebUIフレームワーク
- **FastAPI**: 高速APIフレームワーク

---

**🚀 Happy Fine-tuning & Comparing! 🤖**

*最終更新: 2025年9月30日*