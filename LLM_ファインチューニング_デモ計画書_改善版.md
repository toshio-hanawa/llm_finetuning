# 日本語LLMファインチューニング デモ計画書（改善版）

## 1. プロジェクト概要

### 目的
- 同僚に日本語LLMのファインチューニングの概念と実装方法を説明する
- 日本語データでのファインチューニングを実際に体験し、その効果を実感してもらう
- 日本語特化のオープンソースツールを使用した実践的なデモンストレーション
- 日本語自然言語処理の特殊性と最適化手法の理解

### 対象者
- データサイエンス・データエンジニアリングチームの同僚
- LLMの基礎知識はあるが、日本語ファインチューニングの実装経験がない方
- 日本語NLPプロジェクトに関心のある方

## 2. 環境仕様

### 現在の環境
- **OS**: Ubuntu 24.04 LTS
- **CPU**: Intel Xeon Platinum 8488C (8コア)
- **メモリ**: 30GB RAM
- **ストレージ**: 116GB (64GB利用可能)
- **GPU**: なし（CPUベースで実行）

### 推奨環境
- Python 3.9以上
- 仮想環境（uv使用）
- 十分なメモリ（最低16GB推奨）

## 3. 使用する日本語対応オープンソースLLM

### 日本語特化モデル候補
1. **OpenCALM-7B** (7Bパラメータ)
   - サイバーエージェント製の日本語特化モデル
   - 日本語での高品質な出力
   - 商用利用可能
   - CPU実行は重いが、デモ用に調整可能

2. **TinyLlama-1.1B** (1.1Bパラメータ)
   - 軽量でCPUでも実行可能
   - 多言語対応（日本語含む）
   - デモ用に最適
   - 学習時間が短い

3. **Sarashina-7B** (7Bパラメータ)
   - ソフトバンク製の日本語特化モデル
   - 高品質な日本語出力
   - 商用利用可能

### 最終選択
**TinyLlama-1.1B** を選択（日本語対応版）
- 最も軽量でデモに適している
- 学習時間が短い（30分-1時間程度）
- メモリ使用量が少ない
- 日本語トークナイザーでの前処理が可能

### 日本語処理の考慮点
- **トークナイザー**: 日本語に最適化されたトークナイザーの使用
- **文字エンコーディング**: UTF-8での適切な処理
- **形態素解析**: 必要に応じてMeCabやSudachiPyの活用

## 4. 日本語デモ用データセット

### データセット選択
1. **日本語QAデータセット**
   - 日本の歴史・文化に関する質問と回答
   - 100-200サンプル程度
   - シンプルで理解しやすい内容
   - 日本語の自然な表現を使用

2. **日本語チャット形式データセット**
   - 対話形式のデータ
   - 敬語・丁寧語の適切な使用
   - 日本語特有の表現パターン

3. **カスタムデータセット例**
   ```
   質問: 江戸時代の将軍は誰ですか？
   回答: 江戸時代の将軍は徳川家康から始まり、徳川慶喜まで15代続きました。
   
   質問: 明治維新はいつ起こりましたか？
   回答: 明治維新は1868年に起こり、江戸幕府が倒され明治政府が成立しました。
   
   質問: 日本の四季について教えてください。
   回答: 日本には春、夏、秋、冬の四季があり、それぞれ独特の美しさがあります。
   ```

### 日本語データの特徴
- **文字種**: ひらがな、カタカナ、漢字、英数字の混在
- **敬語**: 丁寧語、尊敬語、謙譲語の適切な使用
- **助詞**: 日本語特有の助詞の正確な使用
- **語順**: 日本語の自然な語順

### データ形式
- JSON Lines形式
- 質問と回答のペア
- 日本語での説明が可能
- 文字エンコーディング: UTF-8
- 改行コード: LF

### データ前処理
- **正規化**: 全角・半角の統一
- **形態素解析**: MeCabやSudachiPyでの分かち書き
- **トークン化**: 日本語に最適化されたトークナイザー

## 5. 日本語特化技術スタック

### 使用ライブラリ
- **Transformers**: Hugging Faceのライブラリ
- **Datasets**: データセット管理
- **Accelerate**: 分散学習サポート
- **PEFT**: Parameter Efficient Fine-Tuning
- **LoRA**: Low-Rank Adaptation
- **MeCab**: 日本語形態素解析
- **SudachiPy**: 高精度日本語形態素解析
- **jaconv**: 日本語文字変換

### 日本語処理ライブラリ
- **MeCab**: 形態素解析エンジン
- **SudachiPy**: 高精度形態素解析
- **jaconv**: ひらがな・カタカナ・ローマ字変換
- **mojimoji**: 全角・半角変換

### 学習手法
- **LoRA (Low-Rank Adaptation)**
  - パラメータ効率的なファインチューニング
  - メモリ使用量を大幅に削減
  - 学習時間の短縮
  - 日本語特有の表現パターンの学習

### 日本語特化設定
- **トークナイザー**: 日本語に最適化された設定
- **特殊トークン**: 日本語特有の特殊トークンの追加
- **文字エンコーディング**: UTF-8での統一処理
- **前処理**: 日本語の正規化とクリーンアップ

## 6. 実装計画

### Phase 1: 日本語環境構築 (30分)
1. Python仮想環境の作成
2. 日本語処理ライブラリのインストール
3. 日本語データセットの準備
4. 日本語トークナイザーの設定

### Phase 2: 日本語ベースラインモデルの評価 (15分)
1. **Before評価の実施**
   - 標準化された日本語質問セット（10問）での回答生成
   - 回答品質の定量的評価（BLEU、ROUGE、BERTScore）
   - 日本語表現の自然さ評価（5段階評価）
   - 敬語・丁寧語の使用頻度測定
   - 回答時間の記録

2. **評価データの記録**
   - 各質問に対する回答をJSON形式で保存
   - 評価指標の数値記録
   - スクリーンショットでの視覚的記録

### Phase 3: 日本語ファインチューニング実行 (45-60分)
1. 日本語特化LoRA設定の調整
2. 日本語データでの学習ループの実行
3. 学習進捗の可視化
4. 日本語表現の学習状況の監視
5. **リアルタイム評価**
   - 学習中の損失値の記録
   - 中間評価（学習途中での回答品質チェック）

### Phase 4: 日本語結果比較 (15分)
1. **After評価の実施**
   - 同じ標準化質問セット（10問）での回答生成
   - 同様の評価指標での測定
   - 学習前と同じ条件での評価

2. **Before/After比較の実施**
   - 回答品質の数値比較（改善率の計算）
   - 回答内容の並列表示
   - 改善点の具体的な指摘
   - 視覚的な比較チャートの作成

3. **詳細分析**
   - 敬語・丁寧語の使用改善度合い
   - 日本語表現の自然さ向上
   - ドメイン知識の習得度
   - 回答の一貫性向上

## 7. 評価プロトコル

### 7.1 評価用質問セット（10問）
```python
evaluation_questions = [
    "江戸時代の将軍について教えてください",
    "日本の四季の特徴を説明してください", 
    "ビジネスメールの書き方を教えてください",
    "日本の伝統文化について説明してください",
    "敬語の使い分けについて教えてください",
    "日本の地理的特徴を説明してください",
    "日本の教育制度について教えてください",
    "日本の経済について説明してください",
    "日本の科学技術について教えてください",
    "日本の社会問題について教えてください"
]
```

### 7.2 評価指標の定義

#### 定量的評価指標
- **BLEU Score**: 翻訳品質の評価（0-1、高いほど良い）
- **ROUGE Score**: 要約品質の評価（0-1、高いほど良い）
- **BERTScore**: 意味的類似度の評価（0-1、高いほど良い）
- **回答時間**: 質問から回答までの時間（秒）

#### 定性的評価指標
- **日本語自然さ**: 1-5段階評価（5が最も自然）
- **敬語使用頻度**: 0-1（1が最も適切）
- **丁寧語使用頻度**: 0-1（1が最も適切）
- **助詞の正確性**: 1-5段階評価（5が最も正確）
- **語順の自然さ**: 1-5段階評価（5が最も自然）

### 7.3 評価データの保存形式
```json
{
  "evaluation_id": "demo_20241201_001",
  "model_name": "TinyLlama-1.1B",
  "phase": "before", // or "after"
  "timestamp": "2024-12-01T10:00:00Z",
  "evaluations": [
    {
      "question_id": 1,
      "question": "江戸時代の将軍について教えてください",
      "answer": "江戸時代の将軍は...",
      "metrics": {
        "bleu_score": 0.65,
        "rouge_score": 0.72,
        "bert_score": 0.78,
        "response_time": 2.3,
        "naturalness": 3.2,
        "keigo_frequency": 0.3,
        "teineigo_frequency": 0.7,
        "particle_accuracy": 4.1,
        "word_order": 3.8
      }
    }
  ],
  "summary_metrics": {
    "avg_bleu": 0.65,
    "avg_rouge": 0.72,
    "avg_bert": 0.78,
    "avg_response_time": 2.3,
    "avg_naturalness": 3.2
  }
}
```

## 8. 比較可視化手法

### 8.1 数値比較チャート
- **回答品質改善グラフ**: BLEU/ROUGE/BERTScoreのBefore/After比較
- **日本語表現改善グラフ**: 自然さ評価の向上
- **敬語使用頻度変化**: 学習前後の比較
- **回答時間の変化**: 効率性の向上

### 8.2 回答内容の並列表示
```
質問: 江戸時代の将軍について教えてください

【Before】
回答: 江戸時代の将軍は徳川家康から始まりました。

【After】  
回答: 江戸時代の将軍は、徳川家康から始まり、徳川慶喜まで15代続きました。
       初代将軍の徳川家康は1603年に征夷大将軍に任じられ、江戸幕府を開きました。
```

### 8.3 改善点のハイライト
- **改善された表現**: 緑色でハイライト
- **新しく追加された情報**: 青色でハイライト
- **敬語・丁寧語の改善**: 黄色でハイライト

## 9. デモンストレーション内容

### 説明ポイント
1. **日本語ファインチューニングとは**
   - 事前学習済みモデルを日本語タスクに特化させる手法
   - ゼロから学習する必要がない
   - 少ない日本語データで効果的な学習が可能
   - 日本語特有の表現パターンの学習

2. **日本語LoRAの仕組み**
   - パラメータの大部分を凍結
   - 低ランク行列で効率的に学習
   - メモリ効率と学習効率の両立
   - 日本語表現の微調整

3. **日本語実装の流れ**
   - 日本語データの前処理（形態素解析、正規化）
   - 日本語特化モデルの設定
   - 日本語データでの学習の実行
   - 日本語品質の評価

4. **日本語特有の考慮点**
   - 文字種の混在（ひらがな、カタカナ、漢字）
   - 敬語・丁寧語の適切な使用
   - 助詞の正確な使用
   - 自然な日本語の語順

### 可視化要素
- 学習損失のグラフ
- 学習前後の日本語回答品質比較
- メモリ使用量の変化
- 学習時間の記録
- 日本語表現の改善度合い
- 敬語・丁寧語の使用頻度変化
- 文字種別の使用パターン分析

## 10. 期待される成果

### 学習効果
- 日本語の質問に対する回答精度の向上
- ドメイン特化した日本語知識の獲得
- より自然な日本語での回答生成
- 敬語・丁寧語の適切な使用
- 日本語特有の表現パターンの習得

### 技術的学習
- 日本語ファインチューニングの実装方法
- 日本語LoRAの効果的な使用方法
- リソース制約下での最適化手法
- 日本語形態素解析の活用
- 日本語トークナイザーの最適化

## 11. リスクと対策

### リスク
- **メモリ不足**: 30GB RAMでも大規模モデルは困難
- **学習時間**: CPU実行のため時間がかかる
- **品質**: 軽量モデルのため出力品質に限界
- **日本語品質**: 日本語特有の表現の学習が不十分
- **文字エンコーディング**: 日本語文字の処理エラー

### 対策
- 軽量モデル（TinyLlama）の使用
- LoRAによる効率化
- 適切なデータセットサイズの調整
- 事前のテスト実行
- 日本語形態素解析の活用
- UTF-8エンコーディングの統一
- 日本語データの品質チェック

## 12. スケジュール

### 準備期間 (1-2日)
- 日本語環境構築
- 日本語データセット準備
- 日本語処理ライブラリのテスト
- 事前テスト

### デモ当日 (2時間)
- 日本語ファインチューニング説明 (30分)
- 日本語実装 (60分)
- 日本語結果確認 (30分)

## 13. 必要なリソース

### ハードウェア
- CPU: 8コア以上
- メモリ: 16GB以上
- ストレージ: 10GB以上の空き容量

### ソフトウェア
- Python 3.9+
- Git
- 必要なPythonライブラリ
- 日本語処理ライブラリ（MeCab、SudachiPy）
- 日本語形態素解析辞書

## 14. 成功指標

### 技術指標
- 学習損失の収束
- メモリ使用量の最適化
- 学習時間の短縮
- 日本語品質の向上
- 日本語表現の自然さ

### 教育指標
- 参加者の理解度
- 実装の再現可能性
- 今後の応用可能性
- 日本語NLPプロジェクトへの応用

---

## 付録: 実装コード例

### 日本語環境構築
```bash
# 仮想環境の作成
uv venv japanese-llm-finetuning-demo
source japanese-llm-finetuning-demo/bin/activate

# 必要なライブラリのインストール
uv pip install transformers datasets accelerate peft torch
uv pip install mecab-python3 sudachipy jaconv mojimoji

# MeCab辞書のインストール（Ubuntu）
sudo apt-get install mecab mecab-ipadic-utf8
```

### 日本語特化学習コード
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch
import mecab
import jaconv

# 日本語形態素解析器の初期化
mecab_tagger = mecab.MeCab()

# モデルとトークナイザーの読み込み
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 日本語特殊トークンの追加
tokenizer.add_tokens(["<|japanese|>", "<|question|>", "<|answer|>"])

# LoRA設定（日本語特化）
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRAモデルの作成
model = get_peft_model(model, lora_config)

# 日本語データの前処理関数
def preprocess_japanese_text(text):
    # 全角・半角の統一
    text = jaconv.h2z(text)
    # 形態素解析による分かち書き
    tokens = mecab_tagger.parse(text).strip()
    return tokens
```

### 評価コード例
```python
import json
from datetime import datetime

def evaluate_model(model, tokenizer, questions, phase="before"):
    """モデルの評価を実行"""
    results = {
        "evaluation_id": f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "model_name": "TinyLlama-1.1B",
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "evaluations": []
    }
    
    for i, question in enumerate(questions):
        # 回答生成
        start_time = time.time()
        answer = generate_answer(model, tokenizer, question)
        response_time = time.time() - start_time
        
        # 評価指標の計算
        metrics = calculate_metrics(question, answer)
        metrics["response_time"] = response_time
        
        # 結果の記録
        results["evaluations"].append({
            "question_id": i + 1,
            "question": question,
            "answer": answer,
            "metrics": metrics
        })
    
    # サマリー指標の計算
    results["summary_metrics"] = calculate_summary_metrics(results["evaluations"])
    
    return results

def compare_before_after(before_results, after_results):
    """Before/After比較の実行"""
    comparison = {
        "improvement_rates": {},
        "detailed_comparison": []
    }
    
    # 改善率の計算
    for metric in ["bleu_score", "rouge_score", "bert_score", "naturalness"]:
        before_avg = before_results["summary_metrics"][f"avg_{metric}"]
        after_avg = after_results["summary_metrics"][f"avg_{metric}"]
        improvement = ((after_avg - before_avg) / before_avg) * 100
        comparison["improvement_rates"][metric] = improvement
    
    # 詳細比較
    for i in range(len(before_results["evaluations"])):
        before_eval = before_results["evaluations"][i]
        after_eval = after_results["evaluations"][i]
        
        comparison["detailed_comparison"].append({
            "question_id": i + 1,
            "question": before_eval["question"],
            "before_answer": before_eval["answer"],
            "after_answer": after_eval["answer"],
            "before_metrics": before_eval["metrics"],
            "after_metrics": after_eval["metrics"]
        })
    
    return comparison
```

この計画書に基づいて、実際のファインチューニングデモを実装していきましょう。
