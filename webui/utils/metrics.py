#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
評価メトリクス計算ユーティリティ
"""

import re
import logging
from typing import Dict, Any, List
from collections import Counter

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """評価指標計算クラス"""
    
    def __init__(self):
        # 日本語文字パターン
        self.hiragana_pattern = re.compile(r'[ひ-ん]')
        self.katakana_pattern = re.compile(r'[ア-ン]')
        self.kanji_pattern = re.compile(r'[一-龯]')
        self.keigo_pattern = re.compile(r'(です|ます|ございます|いたします|させていただき|恐れ入り|申し上げ|いただき|くださ)')
        self.teineigo_pattern = re.compile(r'(である|だ|する|なる)(?![ぁ-ん])')
    
    def calculate_text_metrics(self, text: str) -> Dict[str, Any]:
        """テキストの基本メトリクスを計算"""
        try:
            if not text or len(text.strip()) == 0:
                return self._empty_metrics()
            
            # 基本統計
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = len([s for s in re.split(r'[。！？]', text) if s.strip()])
            
            # 日本語特性分析
            hiragana_count = len(self.hiragana_pattern.findall(text))
            katakana_count = len(self.katakana_pattern.findall(text))
            kanji_count = len(self.kanji_pattern.findall(text))
            
            # 文字種多様性
            total_jp_chars = hiragana_count + katakana_count + kanji_count
            char_diversity = total_jp_chars / char_count * 100 if char_count > 0 else 0
            
            # 敬語・丁寧語使用率
            keigo_matches = len(self.keigo_pattern.findall(text))
            teineigo_matches = len(self.teineigo_pattern.findall(text))
            
            keigo_rate = keigo_matches / sentence_count * 100 if sentence_count > 0 else 0
            teineigo_rate = teineigo_matches / sentence_count * 100 if sentence_count > 0 else 0
            
            # 読みやすさ指標（文の長さの分散）
            sentences = [s.strip() for s in re.split(r'[。！？]', text) if s.strip()]
            sentence_lengths = [len(s) for s in sentences]
            avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
            
            # 語彙の豊富さ（ユニークな単語の割合）
            words = text.split()
            unique_words = set(words)
            vocabulary_richness = len(unique_words) / len(words) * 100 if words else 0
            
            return {
                "basic_stats": {
                    "character_count": char_count,
                    "word_count": word_count,
                    "sentence_count": sentence_count,
                    "avg_sentence_length": round(avg_sentence_length, 2)
                },
                "japanese_characteristics": {
                    "hiragana_count": hiragana_count,
                    "katakana_count": katakana_count,
                    "kanji_count": kanji_count,
                    "character_diversity": round(char_diversity, 2)
                },
                "linguistic_quality": {
                    "keigo_usage_rate": round(keigo_rate, 2),
                    "teineigo_usage_rate": round(teineigo_rate, 2),
                    "vocabulary_richness": round(vocabulary_richness, 2)
                },
                "readability": {
                    "avg_sentence_length": round(avg_sentence_length, 2),
                    "sentence_length_variance": round(self._calculate_variance(sentence_lengths), 2)
                }
            }
            
        except Exception as e:
            logger.error(f"メトリクス計算エラー: {e}")
            return self._empty_metrics()
    
    def compare_texts(self, base_text: str, finetuned_text: str) -> Dict[str, Any]:
        """2つのテキストを比較"""
        try:
            base_metrics = self.calculate_text_metrics(base_text)
            finetuned_metrics = self.calculate_text_metrics(finetuned_text)
            
            # 改善率の計算
            improvements = {}
            
            # 基本統計の比較
            improvements["character_improvement"] = self._calculate_improvement(
                base_metrics["basic_stats"]["character_count"],
                finetuned_metrics["basic_stats"]["character_count"]
            )
            
            # 日本語特性の改善
            improvements["diversity_improvement"] = self._calculate_improvement(
                base_metrics["japanese_characteristics"]["character_diversity"],
                finetuned_metrics["japanese_characteristics"]["character_diversity"]
            )
            
            # 言語品質の改善
            improvements["keigo_improvement"] = self._calculate_improvement(
                base_metrics["linguistic_quality"]["keigo_usage_rate"],
                finetuned_metrics["linguistic_quality"]["keigo_usage_rate"]
            )
            
            improvements["vocabulary_improvement"] = self._calculate_improvement(
                base_metrics["linguistic_quality"]["vocabulary_richness"],
                finetuned_metrics["linguistic_quality"]["vocabulary_richness"]
            )
            
            # 類似度計算（単純な文字レベル）
            similarity = self._calculate_similarity(base_text, finetuned_text)
            
            return {
                "base_metrics": base_metrics,
                "finetuned_metrics": finetuned_metrics,
                "improvements": improvements,
                "similarity": similarity,
                "overall_improvement": round(
                    sum(improvements.values()) / len(improvements), 2
                )
            }
            
        except Exception as e:
            logger.error(f"テキスト比較エラー: {e}")
            return {"error": str(e)}
    
    def _calculate_improvement(self, base_value: float, finetuned_value: float) -> float:
        """改善率を計算"""
        if base_value == 0:
            return 100.0 if finetuned_value > 0 else 0.0
        return round((finetuned_value - base_value) / base_value * 100, 2)
    
    def _calculate_variance(self, values: List[float]) -> float:
        """分散を計算"""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """簡単な類似度計算（Jaccard係数）"""
        try:
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            if not words1 and not words2:
                return 100.0
            if not words1 or not words2:
                return 0.0
                
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return round(intersection / union * 100, 2) if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """空のメトリクスを返す"""
        return {
            "basic_stats": {
                "character_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "avg_sentence_length": 0
            },
            "japanese_characteristics": {
                "hiragana_count": 0,
                "katakana_count": 0,
                "kanji_count": 0,
                "character_diversity": 0
            },
            "linguistic_quality": {
                "keigo_usage_rate": 0,
                "teineigo_usage_rate": 0,
                "vocabulary_richness": 0
            },
            "readability": {
                "avg_sentence_length": 0,
                "sentence_length_variance": 0
            }
        }

# グローバルメトリクス計算インスタンス
metrics_calculator = MetricsCalculator()
