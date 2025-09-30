#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本語LLMファインチューニング システム
三陽商会ドキュメントを基にした日本語特化LoRAファインチューニング
"""

import json
import time
import torch
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import MeCab
import jaconv
import mojimoji
import numpy as np
from tqdm import tqdm

# MeCabの設定
os.environ['MECABRC'] = '/etc/mecabrc'

class JapaneseDataProcessor:
    """日本語データ処理クラス"""
    
    def __init__(self):
        self.mecab = MeCab.Tagger()
    
    def normalize_text(self, text: str) -> str:
        """日本語テキストの正規化"""
        # 全角・半角の統一
        text = jaconv.h2z(text)
        # 英数字の半角化
        text = mojimoji.zen_to_han(text, kana=False)
        return text.strip()
    
    def create_prompt(self, question: str, answer: str) -> str:
        """プロンプトの作成"""
        prompt = f"<|japanese|><|question|>{question}<|answer|>{answer}<|end|>"
        return self.normalize_text(prompt)
    
    def load_qa_dataset(self, filepath: str) -> List[Dict[str, str]]:
        """QAデータセットの読み込み"""
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        data.append(item)
            print(f"データセット読み込み完了: {len(data)}件")
            return data
        except Exception as e:
            print(f"データセット読み込みエラー: {e}")
            return []
    
    def prepare_training_data(self, qa_data: List[Dict[str, str]]) -> List[str]:
        """学習用データの準備"""
        training_texts = []
        
        for item in tqdm(qa_data, desc="データ前処理"):
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            if question and answer:
                prompt = self.create_prompt(question, answer)
                training_texts.append(prompt)
        
        print(f"学習用テキスト準備完了: {len(training_texts)}件")
        return training_texts

class JapaneseLoRAFineTuner:
    """日本語特化LoRAファインチューニングクラス"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_processor = JapaneseDataProcessor()
        
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
        self.tokenizer.add_tokens(["<|japanese|>", "<|question|>", "<|answer|>", "<|end|>"])
        
        # 埋め込み層のリサイズ（重要！）
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        print(f"語彙サイズ: {len(self.tokenizer)}")
        print(f"埋め込み層サイズ: {self.model.get_input_embeddings().num_embeddings}")
        
        print("モデル読み込み完了")
    
    def setup_lora_config(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1) -> LoraConfig:
        """LoRA設定の構築"""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        return lora_config
    
    def prepare_dataset(self, training_texts: List[str], max_length: int = 512) -> Dataset:
        """データセットの準備"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # データセットの作成
        dataset = Dataset.from_dict({"text": training_texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def fine_tune(
        self,
        training_texts: List[str],
        output_dir: str = "./japanese_finetuned_model",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_length: int = 512,
        save_steps: int = 100,
        eval_steps: int = 100,
        logging_steps: int = 10
    ) -> Dict[str, Any]:
        """LoRAファインチューニングの実行"""
        
        print("\n=== 日本語LoRAファインチューニング開始 ===")
        
        # LoRA設定
        lora_config = self.setup_lora_config()
        
        # LoRAモデルの作成
        model = get_peft_model(self.model, lora_config)
        model.print_trainable_parameters()
        
        # データセットの準備
        dataset = self.prepare_dataset(training_texts, max_length)
        
        # データコレーターの設定
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 学習引数の設定
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # wandb等のログ出力を無効化
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=self.device == "cuda",
            gradient_accumulation_steps=4,
            max_grad_norm=1.0,
        )
        
        # トレーナーの初期化
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 学習の実行
        print("学習開始...")
        start_time = time.time()
        
        try:
            trainer.train()
            training_time = time.time() - start_time
            
            # モデルの保存
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            print(f"学習完了: {training_time:.2f}秒")
            
            # 学習結果の記録
            training_results = {
                "model_name": self.model_name,
                "training_time": training_time,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "total_samples": len(training_texts),
                "output_dir": output_dir,
                "timestamp": datetime.now().isoformat()
            }
            
            return training_results
            
        except Exception as e:
            print(f"学習エラー: {e}")
            return {"error": str(e)}
    
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
            
            # <|end|>トークンで終了
            if "<|end|>" in answer:
                answer = answer.split("<|end|>")[0].strip()
            
            return answer if answer else "回答を生成できませんでした。"
            
        except Exception as e:
            print(f"回答生成エラー: {e}")
            return "エラーが発生しました。"

def main():
    """メイン実行関数"""
    print("=== 日本語LLMファインチューニング システム ===")
    
    # データセットの読み込み
    data_processor = JapaneseDataProcessor()
    qa_data = data_processor.load_qa_dataset('dataset/japanese_qa_dataset.jsonl')
    
    if not qa_data:
        print("データセットの読み込みに失敗しました。")
        return
    
    # 学習用データの準備
    training_texts = data_processor.prepare_training_data(qa_data)
    
    if not training_texts:
        print("学習用データの準備に失敗しました。")
        return
    
    # ファインチューニングの実行
    fine_tuner = JapaneseLoRAFineTuner()
    
    training_results = fine_tuner.fine_tune(
        training_texts=training_texts,
        output_dir="./japanese_finetuned_model",
        num_epochs=3,
        batch_size=2,  # CPU実行のため小さく設定
        learning_rate=2e-4,
        max_length=512,
        save_steps=50,
        eval_steps=50,
        logging_steps=5
    )
    
    # 結果の保存
    with open("results/training_results.json", "w", encoding="utf-8") as f:
        json.dump(training_results, f, ensure_ascii=False, indent=2)
    
    print("\n=== ファインチューニング完了 ===")
    print("結果ファイル:")
    print("- results/training_results.json")
    print("- japanese_finetuned_model/ (モデルファイル)")

if __name__ == "__main__":
    main()
