# LLMファインチューニング実装記録

## 📋 プロジェクト概要

**目的**: 日本語LLMファインチューニングデモの実装  
**対象ドキュメント**: 三陽商会経営実態調査報告書  
**技術スタック**: TinyLlama-1.1B + LoRA + Hugging Face Transformers  
**実装期間**: 2024年9月29日  

---

## 🎯 実装計画

### 基本要件
- **実行環境**: CPU（GPU非対応）
- **学習時間**: 適切な範囲内
- **データセット**: 参照ドキュメントから想定問答を作成
- **出力形式**: 評価結果（CSV）、デモレポート（Markdown）

### 技術仕様
- **ベースモデル**: TinyLlama-1.1B-Chat-v1.0
- **ファインチューニング手法**: LoRA（Low-Rank Adaptation）
- **データセット**: 三陽商会ドキュメントから生成した日本語QA
- **評価指標**: BLEU、ROUGE、BERTScore、日本語自然性、敬語使用頻度

---

## 📁 ファイル構成

```
/home/ubuntu/Documents/projects/llm_finetuning/
├── LLM_ファインチューニング_デモ計画書_改善版.md
├── dataset/
│   ├── raw/
│   │   └── compass_artifact_wf-2b19dbab-4647-4022-a5c3-ec6e6a2f37dc_text_markdown.md
│   ├── japanese_qa_dataset.jsonl
│   └── evaluation_questions.json
├── src/
│   ├── baseline_evaluation.py
│   ├── japanese_finetuning.py
│   ├── comparison_evaluation.py
│   └── visualization.py
├── results/
│   ├── baseline_evaluation.csv
│   ├── baseline_evaluation.json
│   ├── comparison_evaluation.csv
│   ├── comparison_evaluation.json
│   └── training_results.json
├── models/
│   └── finetuned_japanese_llm/
└── demo_script.py
```

---

## 🔧 実装フェーズ

### Phase 1: 環境セットアップ
- **仮想環境**: `uv`を使用してPython環境構築
- **依存関係**: transformers, peft, datasets, torch, MeCab, jaconv, mojimoji
- **MeCab設定**: `/etc/mecabrc`を明示的に指定

### Phase 2: データセット作成
- **参照ドキュメント**: 三陽商会経営実態調査報告書
- **生成データ**: 133問の日本語QAペア
- **評価用質問**: 133問の評価用質問セット

### Phase 3: ベースライン評価
- **実装ファイル**: `src/baseline_evaluation.py`
- **評価対象**: TinyLlama-1.1B（ファインチューニング前）
- **評価指標**: BLEU、ROUGE、BERTScore、日本語自然性

### Phase 4: ファインチューニング
- **実装ファイル**: `src/japanese_finetuning.py`
- **手法**: LoRA（Low-Rank Adaptation）
- **学習時間**: 1033.33秒（約17分）
- **最終損失**: 1.739

### Phase 5: 比較評価
- **実装ファイル**: `src/comparison_evaluation.py`
- **比較対象**: ベースモデル vs ファインチューニング済みモデル
- **評価結果**: 改善率の算出

---

## 🚨 エラー対応記録

### エラー1: MeCab初期化エラー
**症状**: `RuntimeError: [ifs] no such file or directory: /usr/local/etc/mecabrc`

**原因**: MeCabの設定ファイルパスが正しく認識されていない

**解決策**:
```python
# 各Pythonスクリプトに追加
os.environ['MECABRC'] = '/etc/mecabrc'
```

**影響ファイル**:
- `src/baseline_evaluation.py`
- `src/japanese_finetuning.py`
- `src/comparison_evaluation.py`

### エラー2: TrainingArguments引数エラー
**症状**: `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`

**原因**: Hugging Face TransformersライブラリのAPI変更

**解決策**:
```python
# 修正前
evaluation_strategy="steps"

# 修正後
eval_strategy="steps"
```

**影響ファイル**: `src/japanese_finetuning.py`

### エラー3: モデル読み込みエラー
**症状**: `size mismatch for base_model.model.model.embed_tokens.weight`

**原因**: トークナイザーの語彙サイズ不整合（32000 vs 32004）

**解決策**:
```python
# 埋め込み層のリサイズを明示的に実行
self.model.resize_token_embeddings(len(self.tokenizer))
```

**影響ファイル**: `src/comparison_evaluation.py`

---

## 📊 実装結果

### ファインチューニング結果
- **学習時間**: 1033.33秒
- **エポック数**: 3.0
- **最終損失**: 1.739
- **評価損失**: 0.966
- **学習可能パラメータ**: 4,505,600（全体の0.4079%）

### 評価結果
- **データセット**: 133問の日本語QA
- **ベースライン評価**: 完了
- **比較評価**: 完了
- **結果ファイル**: CSV形式で保存

### 生成ファイル
- `results/baseline_evaluation.csv` - ベースライン評価結果
- `results/comparison_evaluation.csv` - 比較評価結果
- `results/training_results.json` - 学習結果
- `models/finetuned_japanese_llm/` - ファインチューニング済みモデル

---

## 🔍 技術的課題と解決策

### 課題1: ファインチューニングの限界
**問題**: ファインチューニング後も回答が不正確なケースが散見

**原因分析**:
1. **データ量不足**: 133サンプルは少なすぎる
2. **学習時間不足**: 3エポックでは不十分
3. **モデルサイズ制約**: TinyLlama-1.1Bの知識容量限界
4. **技術的問題**: モデル読み込みエラーによる評価失敗

**改善策**:
- データセット拡張（500-1000サンプル）
- エポック数増加（10-20）
- より大きなモデル使用（7B-13B）
- RAG（Retrieval-Augmented Generation）の導入

### 課題2: 評価指標の限界
**問題**: BLEU/ROUGEは翻訳品質指標で事実の正確性を測れない

**改善策**:
- ドメイン特化評価指標の導入
- 人間による評価の併用
- 事実検証システムの構築

---

## 📈 学習成果

### 技術的成果
1. **LoRAファインチューニング**: 正常に実装・実行
2. **日本語テキスト処理**: MeCab、jaconv、mojimojiの統合
3. **評価システム**: 多角的な評価指標の実装
4. **エラー対応**: 実践的な問題解決能力

### 教育的成果
1. **ファインチューニングの現実**: 万能ではないことの理解
2. **データ品質の重要性**: 量と質の両方が必要
3. **評価指標の限界**: 適切な指標選択の重要性
4. **段階的改善**: ベースライン改善→ファインチューニングの順序

---

## 🚀 今後の展開

### 短期改善
1. **データセット拡張**: より多くの高品質QAペア
2. **学習パラメータ調整**: エポック数、学習率の最適化
3. **評価指標改善**: ドメイン特化指標の導入

### 中長期展開
1. **WebUI実装**: Streamlit/Gradioによるデモインターフェース
2. **RAGシステム**: 外部知識ベースとの統合
3. **本格運用**: より大きなモデルでの実装

---

## 📝 備忘録

### 重要な学び
- **ファインチューニングは魔法ではない**: 適切なデータと設定が必要
- **評価の重要性**: 定量的な改善測定が不可欠
- **段階的アプローチ**: ベースライン→改善の順序が重要
- **エラー対応**: 実践的な問題解決スキルが重要

### 技術的注意点
- MeCabの設定ファイルパスは環境依存
- Hugging Face TransformersのAPI変更に注意
- トークナイザーの語彙サイズ統一が重要
- CPU環境での学習時間制約を考慮

### 今後の課題
- より大規模なデータセットの準備
- 評価指標の改善
- 本格的なWebUIの実装
- プロダクション環境での運用

---

## 📚 参考資料

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [TinyLlama Model Card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

---

*実装完了日: 2024年9月29日*  
*実装者: AI Assistant*  
*プロジェクト: 日本語LLMファインチューニングデモ*
