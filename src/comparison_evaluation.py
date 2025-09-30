#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本語LLMファインチューニング Before/After比較評価システム
三陽商会ドキュメントを基にした日本語QA評価の比較分析
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import MeCab
import jaconv
import mojimoji

# MeCabの設定
os.environ['MECABRC'] = '/etc/mecabrc'

class JapaneseTextProcessor:
    """日本語テキスト処理クラス"""
    
    def __init__(self):
        self.mecab = MeCab.Tagger()
    
    def normalize_text(self, text: str) -> str:
        """日本語テキストの正規化"""
        text = jaconv.h2z(text)
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

class ComparisonEvaluator:
    """Before/After比較評価クラス"""
    
    def __init__(self, base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.base_model_name = base_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_processor = JapaneseTextProcessor()
        
        print(f"使用デバイス: {self.device}")
        
        # ベースモデルの読み込み
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        # ファインチューニング済みモデルの読み込み
        try:
            # まずベースモデルを再読み込み（語彙サイズを統一）
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # トークナイザーも再読み込み
            finetuned_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            finetuned_tokenizer.add_tokens(["<|japanese|>", "<|question|>", "<|answer|>", "<|end|>"])
            base_model.resize_token_embeddings(len(finetuned_tokenizer))
            
            # PEFTモデルとして読み込み
            self.finetuned_model = PeftModel.from_pretrained(
                base_model, 
                "./japanese_finetuned_model"
            )
            self.finetuned_tokenizer = finetuned_tokenizer
            print("ファインチューニング済みモデル読み込み完了")
        except Exception as e:
            print(f"ファインチューニング済みモデル読み込みエラー: {e}")
            self.finetuned_model = None
            self.finetuned_tokenizer = None
    
    def generate_answer(self, model, tokenizer, question: str, max_length: int = 512) -> str:
        """質問に対する回答を生成"""
        try:
            prompt = f"<|japanese|><|question|>{question}<|answer|>"
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text.split("<|answer|>")[-1].strip()
            
            if "<|end|>" in answer:
                answer = answer.split("<|end|>")[0].strip()
            
            return answer if answer else "回答を生成できませんでした。"
            
        except Exception as e:
            print(f"回答生成エラー: {e}")
            return "エラーが発生しました。"
    
    def calculate_metrics(self, reference: str, candidate: str) -> Dict[str, float]:
        """評価指標の計算"""
        try:
            # BLEUスコア（簡易版）
            ref_tokens = self.text_processor.tokenize_japanese(reference)
            cand_tokens = self.text_processor.tokenize_japanese(candidate)
            
            if not ref_tokens or not cand_tokens:
                bleu_score = 0.0
            else:
                ref_set = set(ref_tokens)
                cand_set = set(cand_tokens)
                if not ref_set:
                    bleu_score = 0.0
                else:
                    precision = len(ref_set.intersection(cand_set)) / len(cand_set)
                    bleu_score = precision
            
            # ROUGEスコア
            try:
                scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
                rouge_scores = scorer.score(reference, candidate)
                rouge_score = rouge_scores['rouge1'].fmeasure
            except:
                rouge_score = 0.0
            
            # BERTScore
            try:
                P, R, F1 = bert_score([candidate], [reference], lang="ja", verbose=False)
                bert_score = F1.item()
            except:
                bert_score = 0.0
            
            return {
                'bleu_score': bleu_score,
                'rouge_score': rouge_score,
                'bert_score': bert_score
            }
            
        except Exception as e:
            print(f"指標計算エラー: {e}")
            return {
                'bleu_score': 0.0,
                'rouge_score': 0.0,
                'bert_score': 0.0
            }
    
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
    
    def compare_models(self, questions: List[str]) -> Dict[str, Any]:
        """モデルの比較評価"""
        print("\n=== Before/After比較評価開始 ===")
        
        comparison_results = {
            "comparison_id": f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "before_results": [],
            "after_results": [],
            "improvement_analysis": {}
        }
        
        total_questions = len(questions)
        
        for i, question in enumerate(questions):
            print(f"比較評価中: {i+1}/{total_questions} - {question[:50]}...")
            
            # Before評価（ベースモデル）
            before_answer = self.generate_answer(
                self.base_model, 
                self.base_tokenizer, 
                question
            )
            before_metrics = self.calculate_metrics(question, before_answer)
            before_quality = self.evaluate_japanese_quality(before_answer)
            
            # After評価（ファインチューニング済みモデル）
            if self.finetuned_model and self.finetuned_tokenizer:
                after_answer = self.generate_answer(
                    self.finetuned_model, 
                    self.finetuned_tokenizer, 
                    question
                )
                after_metrics = self.calculate_metrics(question, after_answer)
                after_quality = self.evaluate_japanese_quality(after_answer)
            else:
                after_answer = "ファインチューニング済みモデルが利用できません"
                after_metrics = {'bleu_score': 0.0, 'rouge_score': 0.0, 'bert_score': 0.0}
                after_quality = {'keigo_frequency': 0.0, 'teineigo_frequency': 0.0, 'char_diversity': 0.0, 'token_count': 0}
            
            # 結果の記録
            before_result = {
                "question_id": i + 1,
                "question": question,
                "answer": before_answer,
                "metrics": before_metrics,
                "japanese_quality": before_quality
            }
            
            after_result = {
                "question_id": i + 1,
                "question": question,
                "answer": after_answer,
                "metrics": after_metrics,
                "japanese_quality": after_quality
            }
            
            comparison_results["before_results"].append(before_result)
            comparison_results["after_results"].append(after_result)
            
            # 進捗表示
            if (i + 1) % 10 == 0:
                print(f"進捗: {i+1}/{total_questions} 完了")
        
        # 改善分析の計算
        comparison_results["improvement_analysis"] = self.calculate_improvement_analysis(
            comparison_results["before_results"],
            comparison_results["after_results"]
        )
        
        print("\n=== Before/After比較評価完了 ===")
        return comparison_results
    
    def calculate_improvement_analysis(self, before_results: List[Dict], after_results: List[Dict]) -> Dict[str, Any]:
        """改善分析の計算"""
        if not before_results or not after_results:
            return {}
        
        # 平均値の計算
        before_avg_bleu = np.mean([r["metrics"]["bleu_score"] for r in before_results])
        after_avg_bleu = np.mean([r["metrics"]["bleu_score"] for r in after_results])
        
        before_avg_rouge = np.mean([r["metrics"]["rouge_score"] for r in before_results])
        after_avg_rouge = np.mean([r["metrics"]["rouge_score"] for r in after_results])
        
        before_avg_bert = np.mean([r["metrics"]["bert_score"] for r in before_results])
        after_avg_bert = np.mean([r["metrics"]["bert_score"] for r in after_results])
        
        before_avg_keigo = np.mean([r["japanese_quality"]["keigo_frequency"] for r in before_results])
        after_avg_keigo = np.mean([r["japanese_quality"]["keigo_frequency"] for r in after_results])
        
        before_avg_teineigo = np.mean([r["japanese_quality"]["teineigo_frequency"] for r in before_results])
        after_avg_teineigo = np.mean([r["japanese_quality"]["teineigo_frequency"] for r in after_results])
        
        before_avg_diversity = np.mean([r["japanese_quality"]["char_diversity"] for r in before_results])
        after_avg_diversity = np.mean([r["japanese_quality"]["char_diversity"] for r in after_results])
        
        # 改善率の計算
        bleu_improvement = ((after_avg_bleu - before_avg_bleu) / before_avg_bleu * 100) if before_avg_bleu > 0 else 0
        rouge_improvement = ((after_avg_rouge - before_avg_rouge) / before_avg_rouge * 100) if before_avg_rouge > 0 else 0
        bert_improvement = ((after_avg_bert - before_avg_bert) / before_avg_bert * 100) if before_avg_bert > 0 else 0
        keigo_improvement = ((after_avg_keigo - before_avg_keigo) / before_avg_keigo * 100) if before_avg_keigo > 0 else 0
        teineigo_improvement = ((after_avg_teineigo - before_avg_teineigo) / before_avg_teineigo * 100) if before_avg_teineigo > 0 else 0
        diversity_improvement = ((after_avg_diversity - before_avg_diversity) / before_avg_diversity * 100) if before_avg_diversity > 0 else 0
        
        return {
            "before_averages": {
                "bleu_score": before_avg_bleu,
                "rouge_score": before_avg_rouge,
                "bert_score": before_avg_bert,
                "keigo_frequency": before_avg_keigo,
                "teineigo_frequency": before_avg_teineigo,
                "char_diversity": before_avg_diversity
            },
            "after_averages": {
                "bleu_score": after_avg_bleu,
                "rouge_score": after_avg_rouge,
                "bert_score": after_avg_bert,
                "keigo_frequency": after_avg_keigo,
                "teineigo_frequency": after_avg_teineigo,
                "char_diversity": after_avg_diversity
            },
            "improvement_rates": {
                "bleu_score": bleu_improvement,
                "rouge_score": rouge_improvement,
                "bert_score": bert_improvement,
                "keigo_frequency": keigo_improvement,
                "teineigo_frequency": teineigo_improvement,
                "char_diversity": diversity_improvement
            }
        }
    
    def save_comparison_results(self, results: Dict[str, Any], filename: str):
        """比較結果をCSVファイルに保存"""
        try:
            # Before結果のDataFrame作成
            before_data = []
            for result in results["before_results"]:
                row = {
                    "phase": "before",
                    "question_id": result["question_id"],
                    "question": result["question"],
                    "answer": result["answer"],
                    "bleu_score": result["metrics"]["bleu_score"],
                    "rouge_score": result["metrics"]["rouge_score"],
                    "bert_score": result["metrics"]["bert_score"],
                    "keigo_frequency": result["japanese_quality"]["keigo_frequency"],
                    "teineigo_frequency": result["japanese_quality"]["teineigo_frequency"],
                    "char_diversity": result["japanese_quality"]["char_diversity"],
                    "token_count": result["japanese_quality"]["token_count"]
                }
                before_data.append(row)
            
            # After結果のDataFrame作成
            after_data = []
            for result in results["after_results"]:
                row = {
                    "phase": "after",
                    "question_id": result["question_id"],
                    "question": result["question"],
                    "answer": result["answer"],
                    "bleu_score": result["metrics"]["bleu_score"],
                    "rouge_score": result["metrics"]["rouge_score"],
                    "bert_score": result["metrics"]["bert_score"],
                    "keigo_frequency": result["japanese_quality"]["keigo_frequency"],
                    "teineigo_frequency": result["japanese_quality"]["teineigo_frequency"],
                    "char_diversity": result["japanese_quality"]["char_diversity"],
                    "token_count": result["japanese_quality"]["token_count"]
                }
                after_data.append(row)
            
            # 統合DataFrameの作成
            all_data = before_data + after_data
            df = pd.DataFrame(all_data)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"比較結果を保存しました: {filename}")
            
        except Exception as e:
            print(f"結果保存エラー: {e}")

def main():
    """メイン実行関数"""
    print("=== 日本語LLMファインチューニング Before/After比較評価 ===")
    
    # 評価用質問の読み込み
    try:
        with open('dataset/evaluation_questions.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = data['evaluation_questions']
        print(f"評価用質問を読み込みました: {len(questions)}問")
    except Exception as e:
        print(f"質問読み込みエラー: {e}")
        return
    
    # 比較評価器の初期化
    evaluator = ComparisonEvaluator()
    
    # 比較評価の実行
    comparison_results = evaluator.compare_models(questions)
    
    # 結果の保存
    evaluator.save_comparison_results(comparison_results, "results/comparison_evaluation.csv")
    
    # JSON結果の保存
    with open("results/comparison_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    # 改善分析の表示
    improvement = comparison_results["improvement_analysis"]
    print("\n=== 改善分析結果 ===")
    print(f"BLEUスコア改善率: {improvement['improvement_rates']['bleu_score']:.2f}%")
    print(f"ROUGEスコア改善率: {improvement['improvement_rates']['rouge_score']:.2f}%")
    print(f"BERTスコア改善率: {improvement['improvement_rates']['bert_score']:.2f}%")
    print(f"敬語使用頻度改善率: {improvement['improvement_rates']['keigo_frequency']:.2f}%")
    print(f"丁寧語使用頻度改善率: {improvement['improvement_rates']['teineigo_frequency']:.2f}%")
    print(f"文字多様性改善率: {improvement['improvement_rates']['char_diversity']:.2f}%")
    
    print("\n=== 比較評価完了 ===")
    print("結果ファイル:")
    print("- results/comparison_evaluation.csv")
    print("- results/comparison_evaluation.json")

if __name__ == "__main__":
    main()
