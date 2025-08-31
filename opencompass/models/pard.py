import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from typing import List, Dict, Optional, Union, Tuple

# 假设 opencompass.models.base 和自定义的 Dream 模型在 python 路径中
from ..base import BaseModel
from ..model4draft.modeling_dream import DreamModel, DreamGenerationConfig


def _rewind_kv_cache(kv_cache: DynamicCache, num_tokens_to_keep: int) -> DynamicCache:
    """辅助函数，用于将 DynamicCache 对象回退到指定的长度。"""
    if kv_cache is None or len(kv_cache.key_cache) == 0:
        return None
    new_cache = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        key, value = kv_cache[layer_idx]
        new_key = key[:, :, :num_tokens_to_keep, :]
        new_value = value[:, :, :num_tokens_to_keep, :]
        new_cache.key_cache.append(new_key)
        new_cache.value_cache.append(new_value)
    new_cache.seen_tokens = num_tokens_to_keep
    return new_cache


class PardDreamModel(BaseModel):
    """
    一个为 OpenCompass 设计的模型包装器，集成了标准生成（基准）和
    Pard-Dream 推测性解码。

    Args:
        path (str): 目标模型的路径。
        draft_model_path (str, optional): 草稿模型的路径。如果提供，则启用
            推测性解码。否则，以基准模式运行。
        max_parallel_draft (int): 并行草稿的词元数量。默认为 64。
        draft_temperature (float): 草稿模型的采样温度。默认为 0.0。
        draft_steps (int): Dream 草稿模型的扩散步数。默认为 1。
        acceptance_top_k (int): 接受标准的 Top-K 值。默认为 10。
        acceptance_threshold (float): 接受标准的概率阈值。默认为 0.1。
    """

    def __init__(self,
                 path: str,
                 draft_model_path: Optional[str] = None,
                 max_seq_len: int = 4096,
                 meta_template: Optional[Dict] = None,
                 max_parallel_draft: int = 64,
                 draft_temperature: float = 0.0,
                 draft_steps: int = 1,
                 acceptance_top_k: int = 10,
                 acceptance_threshold: float = 0.1,
                 **kwargs):
        
        super().__init__(path=path, max_seq_len=max_seq_len, meta_template=meta_template, **kwargs)
        
        self.speculative = draft_model_path is not None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        print(f"Initializing model in {'Speculative' if self.speculative else 'Baseline'} mode.")

        # 加载目标模型 (总是需要)
        self.target_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=self.dtype, trust_remote_code=True
        ).to(self.device).eval()

        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token

        # 加载草稿模型和推测参数 (仅在启用时)
        if self.speculative:
            self.draft_model_path = draft_model_path
            self.draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_path, trust_remote_code=True)
            self.draft_model = DreamModel.from_pretrained(
                draft_model_path, torch_dtype=self.dtype, trust_remote_code=True
            ).to(self.device).eval()
            
            # 存储推测性解码的超参数
            self.max_parallel_draft = max_parallel_draft
            self.draft_temperature = draft_temperature
            self.draft_steps = draft_steps
            self.acceptance_top_k = acceptance_top_k
            self.acceptance_threshold = acceptance_threshold
            self.mask_id = self.draft_model.config.mask_token_id
            print(f"Speculative decoding params: max_draft={self.max_parallel_draft}, temp={self.draft_temperature}, top_k={self.acceptance_top_k}")


    def get_token_len(self, prompt: str) -> int:
        """获取 tokenized 字符串的长度。"""
        return len(self.target_tokenizer.encode(prompt))

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """
        根据给定的输入列表生成结果。根据初始化状态在基准模式和
        推测性解码模式之间切换。
        """
        outputs = []
        for prompt in inputs:
            if self.speculative:
                # 使用 Pard-Dream 推测性解码
                output_text = self._generate_speculative(prompt, max_out_len)
            else:
                # 使用标准基准生成
                output_text = self._generate_baseline(prompt, max_out_len)
            outputs.append(output_text)
        return outputs

    def _generate_baseline(self, prompt: str, max_out_len: int) -> str:
        """标准自回归生成。"""
        input_ids = self.target_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        outputs = self.target_model.generate(
            input_ids,
            max_new_tokens=max_out_len,
            pad_token_id=self.target_tokenizer.eos_token_id,
            do_sample=False # 在评估中确保确定性输出
        )
        
        # 只解码新生成的 tokens
        generated_ids = outputs[0, input_ids.shape[1]:]
        return self.target_tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _generate_speculative(self, prompt: str, max_out_len: int) -> str:
        """推测性解码生成逻辑，改编自 pard_dream_core。"""
        input_ids = self.target_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # 使用 prompt 进行初始前向传播
        target_outputs = self.target_model(input_ids, use_cache=True)
        target_kv_cache = target_outputs.past_key_values
        all_generated_tokens = input_ids
        num_generated = 0

        while num_generated < max_out_len:
            current_length = all_generated_tokens.shape[1]
            draft_len = min(self.max_parallel_draft, max_out_len - num_generated)
            if draft_len <= 0:
                break

            # 1. 草稿生成
            draft_gen_config = DreamGenerationConfig(
                max_new_tokens=draft_len, steps=self.draft_steps, alg='maskgit_plus',
                temperature=self.draft_temperature, mask_token_id=self.mask_id
            )
            draft_sequences = self.draft_model.diffusion_generate(
                inputs=all_generated_tokens, generation_config=draft_gen_config
            )
            draft_tokens = draft_sequences[:, current_length:]

            # 2. Token 映射与验证
            draft_text = self.draft_tokenizer.decode(draft_tokens[0], skip_special_tokens=False)
            mapped_draft_tokens = self.target_tokenizer(draft_text, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)
            draft_verify_len = mapped_draft_tokens.shape[1]

            if draft_verify_len > 0:
                verification_output = self.target_model(mapped_draft_tokens, past_key_values=target_kv_cache, use_cache=True)
                
                # 3. 接受逻辑
                first_draft_logit = target_outputs.logits[:, -1:, :]
                remaining_draft_logits = verification_output.logits[:, :-1, :]
                target_logits_for_match = torch.cat([first_draft_logit, remaining_draft_logits], dim=1)
                
                target_probs = F.softmax(target_logits_for_match, dim=-1)
                _, top_k_indices = torch.topk(target_probs, k=self.acceptance_top_k, dim=-1)
                
                in_top_k = (mapped_draft_tokens.unsqueeze(-1) == top_k_indices).any(dim=-1)
                draft_token_probs = torch.gather(target_probs, -1, mapped_draft_tokens.unsqueeze(-1)).squeeze(-1)
                above_threshold = draft_token_probs > self.acceptance_threshold
                matches = in_top_k & above_threshold
                
                cum_matches = torch.cumprod(matches.int(), dim=1)
                num_accepted = torch.sum(cum_matches, dim=1).item()
                
                accepted_tokens = mapped_draft_tokens[0, :num_accepted]
                
                # 4. 修正步骤
                if num_accepted < draft_verify_len:
                    correction_token = target_logits_for_match.argmax(dim=-1)[0, num_accepted].unsqueeze(0)
                else:
                    correction_token = verification_output.logits[:, -1, :].argmax(dim=-1, keepdim=True).view(-1)
                
                newly_generated = torch.cat([accepted_tokens, correction_token]).unsqueeze(0)
                full_kv_cache_after_verify = verification_output.past_key_values

            else:  # 如果草稿为空，则回退到标准生成一步
                correction_token = target_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                newly_generated = correction_token
                fallback_output = self.target_model(newly_generated, past_key_values=target_kv_cache, use_cache=True)
                full_kv_cache_after_verify = fallback_output.past_key_values

            # 5. 更新状态以进行下一次迭代
            all_generated_tokens = torch.cat([all_generated_tokens, newly_generated], dim=1)
            num_generated += newly_generated.shape[1]
            
            # 将 KV 缓存回退到新的总长度
            target_kv_cache = _rewind_kv_cache(full_kv_cache_after_verify, all_generated_tokens.shape[1])
            
            # 准备下一次草稿-验证循环
            next_input_ids = all_generated_tokens[:, -1:]
            target_outputs = self.target_model(next_input_ids, past_key_values=target_kv_cache, use_cache=True)
            target_kv_cache = target_outputs.past_key_values

            if self.target_tokenizer.eos_token_id in newly_generated:
                break
        
        # 只解码新生成的 tokens
        generated_ids = all_generated_tokens[0, input_ids.shape[1]:]
        return self.target_tokenizer.decode(generated_ids, skip_special_tokens=True)

    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        """此生成模型未实现困惑度计算。"""
        raise NotImplementedError(f'{self.__class__.__name__} does not support ppl-based evaluation.')
