#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本語LLMファインチューニング ベースライン評価システム
三陽商会ドキュメントを基にした日本語QA評価
"""

import json
import time
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import MeCab
import jaconv
import mojimoji
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np

# MeCabの設定
os.environ['MECABRC'] = '/etc/mecabrc'

class JapaneseTextProcessor:
    """日本語テキスト処理クラス"""
    
    def __init__(self):
        self.mecab = MeCab.Tagger()
    
    def normalize_text(self, text: str) -> str:
        """日本語テキストの正規化"""
        # 全角・半角の統一
        text = jaconv.h2z(text)
        # 英数字の半角化
        text = mojimoji.zen_to_han(text, kana=False)
        return text.strip()
    
    def tokenize_japanese(self, text: str) -> List[str]:
        """日本語形態素解析"""
        try:
            parsed = self.mecab.parse(text)
            tokens = []
            for line in parsed.split('\n'):
                if line and line != 'EOS':
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        tokens.append(parts[0])
            return tokens
        except Exception as e:
            print(f"MeCab parsing error: {e}")
            return text.split()

class BaselineEvaluator:
    """ベースラインモデル評価クラス"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_processor = JapaneseTextProcessor()
        
        print(f"使用デバイス: {self.device}")
        print(f"モデル読み込み中: {model_name}")
        
        # モデルとトークナイザーの読み込み
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # パディングトークンの設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 日本語特殊トークンの追加
        self.tokenizer.add_tokens(["<|japanese|>", "<|question|>", "<|answer|>"])
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        print("モデル読み込み完了")
    
    def generate_answer(self, question: str, max_length: int = 512) -> str:
        """質問に対する回答を生成"""
        try:
            # プロンプトの構築
            prompt = f"<|japanese|><|question|>{question}<|answer|>"
            
            # トークン化
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # 回答生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 回答の抽出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text.split("<|answer|>")[-1].strip()
            
            return answer if answer else "回答を生成できませんでした。"
            
        except Exception as e:
            print(f"回答生成エラー: {e}")
            return "エラーが発生しました。"
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """BLEUスコアの計算（簡易版）"""
        try:
            ref_tokens = self.text_processor.tokenize_japanese(reference)
            cand_tokens = self.text_processor.tokenize_japanese(candidate)
            
            if not ref_tokens or not cand_tokens:
                return 0.0
            
            # 1-gram BLEUの簡易計算
            ref_set = set(ref_tokens)
            cand_set = set(cand_tokens)
            
            if not ref_set:
                return 0.0
            
            precision = len(ref_set.intersection(cand_set)) / len(cand_set)
            return precision
            
        except Exception as e:
            print(f"BLEU計算エラー: {e}")
            return 0.0
    
    def calculate_rouge_score(self, reference: str, candidate: str) -> float:
        """ROUGEスコアの計算"""
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
            scores = scorer.score(reference, candidate)
            return scores['rouge1'].fmeasure
        except Exception as e:
            print(f"ROUGE計算エラー: {e}")
            return 0.0
    
    def calculate_bert_score(self, reference: str, candidate: str) -> float:
        """BERTScoreの計算"""
        try:
            P, R, F1 = bert_score([candidate], [reference], lang="ja", verbose=False)
            return F1.item()
        except Exception as e:
            print(f"BERTScore計算エラー: {e}")
            return 0.0
    
    def evaluate_japanese_quality(self, text: str) -> Dict[str, float]:
        """日本語品質の評価"""
        try:
            tokens = self.text_processor.tokenize_japanese(text)
            
            # 敬語・丁寧語の使用頻度
            keigo_words = ['です', 'ます', 'でした', 'ました', 'でございます', 'いたします']
            teineigo_words = ['です', 'ます', 'でした', 'ました']
            
            keigo_count = sum(1 for token in tokens if any(word in token for word in keigo_words))
            teineigo_count = sum(1 for token in tokens if any(word in token for word in teineigo_words))
            
            keigo_frequency = keigo_count / len(tokens) if tokens else 0.0
            teineigo_frequency = teineigo_count / len(tokens) if tokens else 0.0
            
            # 文字種の多様性
            hiragana_count = sum(1 for char in text if '\u3040' <= char <= '\u309F')
            katakana_count = sum(1 for char in text if '\u30A0' <= char <= '\u30FF')
            kanji_count = sum(1 for char in text if '\u4E00' <= char <= '\u9FAF')
            
            total_chars = len(text)
            char_diversity = (hiragana_count + katakana_count + kanji_count) / total_chars if total_chars > 0 else 0.0
            
            return {
                'keigo_frequency': keigo_frequency,
                'teineigo_frequency': teineigo_frequency,
                'char_diversity': char_diversity,
                'token_count': len(tokens)
            }
            
        except Exception as e:
            print(f"日本語品質評価エラー: {e}")
            return {
                'keigo_frequency': 0.0,
                'teineigo_frequency': 0.0,
                'char_diversity': 0.0,
                'token_count': 0
            }
    
    def evaluate_model(self, questions: List[str], phase: str = "before") -> Dict[str, Any]:
        """モデルの評価を実行"""
        print(f"\n=== {phase.upper()}評価開始 ===")
        
        results = {
            "evaluation_id": f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_name": self.model_name,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "evaluations": []
        }
        
        total_questions = len(questions)
        
        for i, question in enumerate(questions):
            print(f"評価中: {i+1}/{total_questions} - {question[:50]}...")
            
            # 回答生成
            start_time = time.time()
            answer = self.generate_answer(question)
            response_time = time.time() - start_time
            
            # 評価指標の計算
            japanese_quality = self.evaluate_japanese_quality(answer)
            
            # 結果の記録
            evaluation_result = {
                "question_id": i + 1,
                "question": question,
                "answer": answer,
                "response_time": response_time,
                "japanese_quality": japanese_quality
            }
            
            results["evaluations"].append(evaluation_result)
            
            # 進捗表示
            if (i + 1) % 10 == 0:
                print(f"進捗: {i+1}/{total_questions} 完了")
        
        # サマリー指標の計算
        results["summary_metrics"] = self.calculate_summary_metrics(results["evaluations"])
        
        print(f"\n=== {phase.upper()}評価完了 ===")
        print(f"総質問数: {total_questions}")
        print(f"平均回答時間: {results['summary_metrics']['avg_response_time']:.2f}秒")
        print(f"平均敬語使用頻度: {results['summary_metrics']['avg_keigo_frequency']:.3f}")
        
        return results
    
    def calculate_summary_metrics(self, evaluations: List[Dict]) -> Dict[str, float]:
        """サマリー指標の計算"""
        if not evaluations:
            return {}
        
        response_times = [e["response_time"] for e in evaluations]
        keigo_frequencies = [e["japanese_quality"]["keigo_frequency"] for e in evaluations]
        teineigo_frequencies = [e["japanese_quality"]["teineigo_frequency"] for e in evaluations]
        char_diversities = [e["japanese_quality"]["char_diversity"] for e in evaluations]
        token_counts = [e["japanese_quality"]["token_count"] for e in evaluations]
        
        return {
            "avg_response_time": np.mean(response_times),
            "avg_keigo_frequency": np.mean(keigo_frequencies),
            "avg_teineigo_frequency": np.mean(teineigo_frequencies),
            "avg_char_diversity": np.mean(char_diversities),
            "avg_token_count": np.mean(token_counts),
            "total_questions": len(evaluations)
        }
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """結果をCSVファイルに保存"""
        try:
            # 評価結果をDataFrameに変換
            data = []
            for eval_result in results["evaluations"]:
                row = {
                    "question_id": eval_result["question_id"],
                    "question": eval_result["question"],
                    "answer": eval_result["answer"],
                    "response_time": eval_result["response_time"],
                    "keigo_frequency": eval_result["japanese_quality"]["keigo_frequency"],
                    "teineigo_frequency": eval_result["japanese_quality"]["teineigo_frequency"],
                    "char_diversity": eval_result["japanese_quality"]["char_diversity"],
                    "token_count": eval_result["japanese_quality"]["token_count"]
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"結果を保存しました: {filename}")
            
        except Exception as e:
            print(f"結果保存エラー: {e}")

def main():
    """メイン実行関数"""
    print("=== 日本語LLMファインチューニング ベースライン評価 ===")
    
    # 評価用質問の読み込み
    try:
        with open('dataset/evaluation_questions.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = data['evaluation_questions']
        print(f"評価用質問を読み込みました: {len(questions)}問")
    except Exception as e:
        print(f"質問読み込みエラー: {e}")
        return
    
    # 評価器の初期化
    evaluator = BaselineEvaluator()
    
    # ベースライン評価の実行
    baseline_results = evaluator.evaluate_model(questions, "before")
    
    # 結果の保存
    evaluator.save_results(baseline_results, "results/baseline_evaluation.csv")
    
    # JSON結果の保存
    with open("results/baseline_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, ensure_ascii=False, indent=2)
    
    print("\n=== ベースライン評価完了 ===")
    print("結果ファイル:")
    print("- results/baseline_evaluation.csv")
    print("- results/baseline_evaluation.json")

if __name__ == "__main__":
    main()
