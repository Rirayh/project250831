from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from opencompass.models.base import BaseModel
from opencompass.utils.logging import get_logger

# --- Custom Dream Model Imports ---
try:
    from opencompass.models.model4draft.modeling_dream import (
        DreamGenerationConfig, DreamModel)
except ImportError:
    print('WARNING: Could not import DreamModel. Using placeholder class.')

    class DreamModel:

        @staticmethod
        def from_pretrained(name, **kwargs):
            raise NotImplementedError(
                'DreamModel is not available. '
                "Please install the required 'model4draft' library.")

    class DreamGenerationConfig:
        pass


# ===================================================================
# 辅助函数 (源自 HuggingFacewithChatTemplate)
# ===================================================================


def _convert_chat_messages(inputs: List[Union[str, List[Dict]]],
                           skip_empty_prompt: bool = True) -> List[List[Dict]]:
    """
    将 OpenCompass 内部的 prompt 格式 (使用 HUMAN/BOT 角色) 转换为
    Hugging Face `apply_chat_template` 所需的标准格式 (使用 user/assistant/system 角色)。
    """
    outputs = []
    for _input in inputs:
        messages = []
        if isinstance(_input, str):
            messages.append({'role': 'user', 'content': _input})
        else:
            for item in _input:
                # 在 Pard-Dream 的上下文中，prompt 是一个 Pydantic 对象，所以我们用 .get()
                prompt_content = item.get('prompt', '') or item.get(
                    'content', '')
                if skip_empty_prompt and not prompt_content:
                    continue
                # 角色映射
                role = {
                    'HUMAN': 'user',
                    'BOT': 'assistant',
                    'SYSTEM': 'system',
                    # 兼容 Hugging Face 格式
                    'user': 'user',
                    'assistant': 'assistant',
                    'system': 'system',
                }[item['role']]
                messages.append({'role': role, 'content': prompt_content})

        outputs.append(messages)
    return outputs


# ===================================================================
# 核心引擎: 这部分代码已与 pard_dream_core.py 对齐
# ===================================================================


def _rewind_kv_cache(kv_cache: DynamicCache,
                     num_tokens_to_keep: int) -> DynamicCache:
    if kv_cache is None or len(kv_cache.key_cache) == 0:
        return None
    new_cache = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        key, value = kv_cache[layer_idx]
        new_cache.key_cache.append(key[:, :, :num_tokens_to_keep, :])
        new_cache.value_cache.append(value[:, :, :num_tokens_to_keep, :])
    return new_cache


class PardDreamEngine:
    """
    Pard-Dream 推测解码的核心逻辑引擎。
    """

    def __init__(self, config: Dict, eos_token_id: int):
        self.logger = get_logger()
        self.logger.info('Initializing PardDreamEngine...')
        self.config = config
        self.target_model_name = config['target_model_name']
        self.draft_model_name = config['draft_model_name']
        self.eos_token_id = eos_token_id

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if torch.cuda.is_available(
        ) and torch.cuda.is_bf16_supported() else torch.float32

        self._load_models_and_tokenizers()
        self.mask_id = self.draft_model.config.mask_token_id
        self.logger.info('PardDreamEngine initialized successfully.')

    def _load_models_and_tokenizers(self):
        self.logger.info(f"Loading target model: {self.target_model_name}")
        self.target_tokenizer = AutoTokenizer.from_pretrained(
            self.target_model_name, trust_remote_code=True)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True).to(self.device).eval()

        self.logger.info(f"Loading draft model: {self.draft_model_name}")
        self.draft_tokenizer = AutoTokenizer.from_pretrained(
            self.draft_model_name, trust_remote_code=True)
        self.draft_model = DreamModel.from_pretrained(
            self.draft_model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True).to(self.device).eval()

        if self.eos_token_id is None:
            self.eos_token_id = self.target_tokenizer.eos_token_id
            self.logger.warning(
                f"eos_token_id not found in meta_template, using tokenizer's default: {self.eos_token_id}"
            )

    @torch.no_grad()
    def generate(self, prompt: List[Dict],
                 max_gen_toks: int) -> Tuple[str, Dict]:

        # --- 重构核心 ---
        # 步骤 1: 将 OpenCompass 的 prompt 格式转换为 Hugging Face 标准格式。
        # _convert_chat_messages 期望一个批次的输入，所以我们传入 [prompt] 并取第一个结果。
        hf_messages = _convert_chat_messages([prompt])[0]

        # 步骤 2: 使用 target_tokenizer 的聊天模板来生成最终的输入字符串。
        final_prompt_str = self.target_tokenizer.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True)

        # 步骤 3: 对最终字符串进行编码。
        input_ids = self.target_tokenizer(final_prompt_str,
                                          return_tensors='pt').input_ids.to(
                                              self.device)

        # --- 推测解码逻辑保持不变 ---
        target_outputs = self.target_model(input_ids, use_cache=True)
        target_kv_cache = target_outputs.past_key_values
        all_generated_tokens = input_ids
        num_generated, iteration = 0, 0

        request_accepted = 0
        request_drafted = 0

        while num_generated < max_gen_toks:
            current_length = all_generated_tokens.shape[1]
            draft_len = min(self.config['max_parallel_draft'],
                            max_gen_toks - num_generated)
            if draft_len <= 0: break
            iteration += 1

            draft_gen_config = DreamGenerationConfig(
                max_new_tokens=draft_len,
                steps=self.config['draft_steps'],
                alg='maskgit_plus',
                temperature=self.config['draft_temperature'],
                mask_token_id=self.mask_id)
            draft_sequences = self.draft_model.diffusion_generate(
                inputs=all_generated_tokens,
                generation_config=draft_gen_config)

            draft_tokens = draft_sequences[:, current_length:]
            draft_text = self.draft_tokenizer.decode(draft_tokens[0],
                                                     skip_special_tokens=False)
            mapped_draft_tokens = self.target_tokenizer(
                draft_text, add_special_tokens=False,
                return_tensors='pt').input_ids.to(self.device)
            draft_verify_len = mapped_draft_tokens.shape[1]

            if draft_verify_len > 0:
                verification_output = self.target_model(
                    mapped_draft_tokens,
                    past_key_values=target_kv_cache,
                    use_cache=True)

                first_draft_logit = target_outputs.logits[:, -1:, :]
                remaining_draft_logits = verification_output.logits[:, :-1, :]
                target_logits_for_match = torch.cat(
                    [first_draft_logit, remaining_draft_logits], dim=1)
                target_probs = F.softmax(target_logits_for_match, dim=-1)
                _, top_k_indices = torch.topk(
                    target_probs, k=self.config['acceptance_top_k'], dim=-1)
                in_top_k = (
                    mapped_draft_tokens.unsqueeze(-1) == top_k_indices).any(
                        dim=-1)
                draft_token_probs = torch.gather(
                    target_probs, -1,
                    mapped_draft_tokens.unsqueeze(-1)).squeeze(-1)
                above_threshold = draft_token_probs > self.config[
                    'acceptance_threshold']
                matches = in_top_k & above_threshold
                cum_matches = torch.cumprod(matches.int(), dim=1)
                num_accepted = torch.sum(cum_matches, dim=1).item()

                request_accepted += num_accepted
                request_drafted += draft_verify_len
                accepted_tokens = mapped_draft_tokens[0, :num_accepted]

                if num_accepted < draft_verify_len:
                    correction_token = target_logits_for_match.argmax(
                        dim=-1)[0, num_accepted].unsqueeze(0)
                else:
                    correction_token = verification_output.logits[:,
                                                                  -1, :].argmax(
                                                                      dim=-1,
                                                                      keepdim=
                                                                      True
                                                                  ).view(-1)
                newly_generated = torch.cat(
                    [accepted_tokens, correction_token]).unsqueeze(0)
                full_kv_cache_after_verify = verification_output.past_key_values
            else:
                correction_token = target_outputs.logits[:, -1, :].argmax(
                    dim=-1, keepdim=True)
                newly_generated = correction_token
                fallback_output = self.target_model(
                    newly_generated,
                    past_key_values=target_kv_cache,
                    use_cache=True)
                full_kv_cache_after_verify = fallback_output.past_key_values

            all_generated_tokens = torch.cat(
                [all_generated_tokens, newly_generated], dim=1)
            num_generated += newly_generated.shape[1]
            target_kv_cache = _rewind_kv_cache(full_kv_cache_after_verify,
                                               all_generated_tokens.shape[1])
            next_input_ids = all_generated_tokens[:, -1:]
            target_outputs = self.target_model(next_input_ids,
                                               past_key_values=target_kv_cache,
                                               use_cache=True)
            target_kv_cache = target_outputs.past_key_values

            if self.eos_token_id in newly_generated:
                break

        output_text = self.target_tokenizer.decode(
            all_generated_tokens[0, input_ids.shape[1]:],
            skip_special_tokens=True)
        metrics = {
            'accepted_tokens': request_accepted,
            'drafted_tokens': request_drafted,
            'draft_attempts': iteration,
        }
        return output_text, metrics


# ===================================================================
# OpenCompass 模型包装器 (已重构)
# ===================================================================


class PardDreamModel(BaseModel):

    def __init__(self,
                 target_model_name: str,
                 draft_model_name: str,
                 max_parallel_draft: int = 256,
                 draft_temperature: float = 0.0,
                 draft_steps: int = 1,
                 acceptance_top_k: int = 1,
                 acceptance_threshold: float = 0.001,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 **kwargs):

        super().__init__(path=target_model_name,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         **kwargs)
        self.logger = get_logger()
        self.logger.info(
            'PardDreamModel is using the modern Hugging Face Chat Template approach.'
        )

        config = {
            'target_model_name': target_model_name,
            'draft_model_name': draft_model_name,
            'max_parallel_draft': max_parallel_draft,
            'draft_temperature': draft_temperature,
            'draft_steps': draft_steps,
            'acceptance_top_k': acceptance_top_k,
            'acceptance_threshold': acceptance_threshold,
        }

        # meta_template 中的 eos_token_id 仍然是有用的
        self.engine = PardDreamEngine(config, eos_token_id=self.eos_token_id)
        # 将 target_tokenizer 暴露给 OpenCompass 框架
        self.tokenizer = self.engine.target_tokenizer

        # 移除了 self.template_parser 和 self._get_meta_template，因为它们不再被使用。

    def generate(self, inputs: List[str], max_out_len: int,
                 **kwargs) -> List[str]:
        outputs = []
        for prompt in inputs:
            output_text, metrics = self.engine.generate(
                prompt=prompt, max_gen_toks=max_out_len)
            self.logger.debug(f"Pard-Dream Metrics: {metrics}")
            outputs.append(output_text)
        return outputs

    def get_token_len(self, prompt: str) -> int:
        # 使用与 generate 方法一致的逻辑来计算 token 长度
        hf_messages = _convert_chat_messages([prompt])[0]
        # 注意: 这里我们不传入 add_generation_prompt=True, 因为通常 get_token_len
        # 是为了计算上下文长度，而不是包含生成提示符的长度。
        # 但为保持一致性，也可以加上。具体取决于框架的要求。
        input_ids = self.tokenizer.apply_chat_template(
            hf_messages, tokenize=True, add_generation_prompt=True)
        return len(input_ids)

    def get_ppl(self, *args, **kwargs):
        # PPL 计算通常需要一个基础模型，这里我们委托给目标模型
        # 注意: get_ppl 的输入格式可能需要适配，这里仅为示例
        return self.engine.target_model.get_ppl(*args, **kwargs)


# # -*- coding: utf-8 -*-
# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers.cache_utils import DynamicCache
# from typing import List, Tuple, Dict, Optional

# from opencompass.models.base import BaseModel
# from opencompass.utils.logging import get_logger

# # --- Custom Dream Model Imports ---
# try:
#     from opencompass.models.model4draft.modeling_dream import DreamModel, DreamGenerationConfig
# except ImportError:
#     print("WARNING: Could not import DreamModel. Using placeholder class.")
#     class DreamModel:
#         @staticmethod
#         def from_pretrained(name, **kwargs):
#             raise NotImplementedError(
#                 "DreamModel is not available. "
#                 "Please install the required 'model4draft' library."
#             )
#     class DreamGenerationConfig:
#         pass

# # ===================================================================
# # 核心引擎: 这部分代码已与 pard_dream_core.py 对齐
# # ===================================================================

# def _rewind_kv_cache(kv_cache: DynamicCache, num_tokens_to_keep: int) -> DynamicCache:
#     if kv_cache is None or len(kv_cache.key_cache) == 0:
#         return None
#     new_cache = DynamicCache()
#     for layer_idx in range(len(kv_cache)):
#         key, value = kv_cache[layer_idx]
#         new_cache.key_cache.append(key[:, :, :num_tokens_to_keep, :])
#         new_cache.value_cache.append(value[:, :, :num_tokens_to_keep, :])
#     return new_cache

# class PardDreamEngine:
#     """
#     Pard-Dream 推测解码的核心逻辑引擎。
#     """
#     def __init__(self, config: Dict, eos_token_id: int):
#         self.logger = get_logger()
#         self.logger.info("Initializing PardDreamEngine...")
#         self.config = config
#         self.target_model_name = config['target_model_name']
#         self.draft_model_name = config['draft_model_name']
#         self.eos_token_id = eos_token_id

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

#         self._load_models_and_tokenizers()
#         # 关键修正: 使用正确的属性名 mask_token_id
#         self.mask_id = self.draft_model.config.mask_token_id
#         self.logger.info("PardDreamEngine initialized successfully.")

#     def _load_models_and_tokenizers(self):
#         self.logger.info(f"Loading target model: {self.target_model_name}")
#         self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_model_name, trust_remote_code=True)
#         self.target_model = AutoModelForCausalLM.from_pretrained(
#             self.target_model_name, torch_dtype=self.dtype, trust_remote_code=True
#         ).to(self.device).eval()

#         self.logger.info(f"Loading draft model: {self.draft_model_name}")
#         self.draft_tokenizer = AutoTokenizer.from_pretrained(self.draft_model_name, trust_remote_code=True)
#         self.draft_model = DreamModel.from_pretrained(
#             self.draft_model_name, torch_dtype=self.dtype, trust_remote_code=True
#         ).to(self.device).eval()

#         if self.eos_token_id is None:
#             self.eos_token_id = self.target_tokenizer.eos_token_id
#             self.logger.warning(f"eos_token_id not found in meta_template, using tokenizer's default: {self.eos_token_id}")

#     @torch.no_grad()
#     def generate(self, prompt: str, max_gen_toks: int) -> Tuple[str, Dict]:
#         input_ids = self.target_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

#         target_outputs = self.target_model(input_ids, use_cache=True)
#         target_kv_cache = target_outputs.past_key_values
#         all_generated_tokens = input_ids
#         num_generated, iteration = 0, 0

#         request_accepted = 0
#         request_drafted = 0

#         while num_generated < max_gen_toks:
#             current_length = all_generated_tokens.shape[1]
#             draft_len = min(self.config['max_parallel_draft'], max_gen_toks - num_generated)
#             if draft_len <= 0: break
#             iteration += 1

#             draft_gen_config = DreamGenerationConfig(max_new_tokens=draft_len, steps=self.config['draft_steps'], alg='maskgit_plus', temperature=self.config['draft_temperature'], mask_token_id=self.mask_id)
#             draft_sequences = self.draft_model.diffusion_generate(inputs=all_generated_tokens, generation_config=draft_gen_config)

#             draft_tokens = draft_sequences[:, current_length:]
#             draft_text = self.draft_tokenizer.decode(draft_tokens[0], skip_special_tokens=False)
#             mapped_draft_tokens = self.target_tokenizer(draft_text, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)
#             draft_verify_len = mapped_draft_tokens.shape[1]

#             if draft_verify_len > 0:
#                 verification_output = self.target_model(mapped_draft_tokens, past_key_values=target_kv_cache, use_cache=True)

#                 first_draft_logit = target_outputs.logits[:, -1:, :]
#                 remaining_draft_logits = verification_output.logits[:, :-1, :]
#                 target_logits_for_match = torch.cat([first_draft_logit, remaining_draft_logits], dim=1)
#                 target_probs = F.softmax(target_logits_for_match, dim=-1)
#                 _, top_k_indices = torch.topk(target_probs, k=self.config['acceptance_top_k'], dim=-1)
#                 in_top_k = (mapped_draft_tokens.unsqueeze(-1) == top_k_indices).any(dim=-1)
#                 draft_token_probs = torch.gather(target_probs, -1, mapped_draft_tokens.unsqueeze(-1)).squeeze(-1)
#                 above_threshold = draft_token_probs > self.config['acceptance_threshold']
#                 matches = in_top_k & above_threshold
#                 cum_matches = torch.cumprod(matches.int(), dim=1)
#                 num_accepted = torch.sum(cum_matches, dim=1).item()

#                 request_accepted += num_accepted
#                 request_drafted += draft_verify_len
#                 accepted_tokens = mapped_draft_tokens[0, :num_accepted]

#                 if num_accepted < draft_verify_len:
#                     correction_token = target_logits_for_match.argmax(dim=-1)[0, num_accepted].unsqueeze(0)
#                 else:
#                     correction_token = verification_output.logits[:, -1, :].argmax(dim=-1, keepdim=True).view(-1)
#                 newly_generated = torch.cat([accepted_tokens, correction_token]).unsqueeze(0)
#                 full_kv_cache_after_verify = verification_output.past_key_values
#             else:
#                 correction_token = target_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
#                 newly_generated = correction_token
#                 fallback_output = self.target_model(newly_generated, past_key_values=target_kv_cache, use_cache=True)
#                 full_kv_cache_after_verify = fallback_output.past_key_values

#             all_generated_tokens = torch.cat([all_generated_tokens, newly_generated], dim=1)
#             num_generated += newly_generated.shape[1]
#             target_kv_cache = _rewind_kv_cache(full_kv_cache_after_verify, all_generated_tokens.shape[1])
#             next_input_ids = all_generated_tokens[:, -1:]
#             target_outputs = self.target_model(next_input_ids, past_key_values=target_kv_cache, use_cache=True)
#             target_kv_cache = target_outputs.past_key_values

#             if self.eos_token_id in newly_generated:
#                 break

#         output_text = self.target_tokenizer.decode(all_generated_tokens[0, input_ids.shape[1]:], skip_special_tokens=True)
#         metrics = {
#             "accepted_tokens": request_accepted,
#             "drafted_tokens": request_drafted,
#             "draft_attempts": iteration,
#         }
#         return output_text, metrics

# # ===================================================================
# # OpenCompass 模型包装器
# # ===================================================================

# class PardDreamModel(BaseModel):
#     """
#     用于 Pard-Dream 推测解码的 OpenCompass 模型包装器。
#     """

#     def __init__(self,
#                  target_model_name: str,
#                  draft_model_name: str,
#                  max_parallel_draft: int = 32,
#                  draft_temperature: float = 0.0,
#                  draft_steps: int = 1,
#                  acceptance_top_k: int = 1,
#                  acceptance_threshold: float = 0.01,
#                  max_seq_len: int = 2048,
#                  meta_template: Optional[Dict] = None,
#                  **kwargs):

#         super().__init__(path=target_model_name,
#                          max_seq_len=max_seq_len,
#                          meta_template=meta_template,
#                          **kwargs)

#         config = {
#             "target_model_name": target_model_name,
#             "draft_model_name": draft_model_name,
#             "max_parallel_draft": max_parallel_draft,
#             "draft_temperature": draft_temperature,
#             "draft_steps": draft_steps,
#             "acceptance_top_k": acceptance_top_k,
#             "acceptance_threshold": acceptance_threshold,
#         }

#         self.engine = PardDreamEngine(config, eos_token_id=self.eos_token_id)
#         self.tokenizer = self.engine.target_tokenizer

#     def generate(self, inputs: List[str], max_out_len: int, **kwargs) -> List[str]:
#         outputs = []
#         for prompt in inputs:
#             output_text, metrics = self.engine.generate(prompt=prompt, max_gen_toks=max_out_len)
#             self.logger.debug(f"Pard-Dream Metrics: {metrics}")
#             outputs.append(output_text)
#         return outputs

#     def get_token_len(self, prompt: str) -> int:
#         return len(self.tokenizer.encode(prompt))

#     def get_ppl(self, *args, **kwargs):
#       return self.engine.target_model.get_ppl(*args, **kwargs)

# # -*- coding: utf-8 -*-
# # # Save this file as pard_dream_model.py
# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers.cache_utils import DynamicCache
# import logging
# from typing import List, Union, Tuple, Dict, Optional

# from opencompass.models.base import BaseModel
# from opencompass.utils.logging import get_logger

# # --- Custom Dream Model Imports ---
# # NOTE: Ensure the 'model4draft' library is installed in your environment.
# try:
#     from opencompass.models.model4draft.modeling_dream import DreamModel, DreamGenerationConfig
# except ImportError:
#     print("WARNING: Could not import DreamModel. Using placeholder class.")
#     class DreamModel:
#         @staticmethod
#         def from_pretrained(name, **kwargs):
#             raise NotImplementedError(
#                 "DreamModel is not available. "
#                 "Please install the required 'model4draft' library."
#             )
#     class DreamGenerationConfig:
#         pass

# # ===================================================================
# # Start: Code migrated from pard_dream_core.py
# # This section contains the core speculative decoding logic,
# # encapsulated here to make the model file self-contained.
# # ===================================================================

# def _rewind_kv_cache(kv_cache: DynamicCache, num_tokens_to_keep: int) -> DynamicCache:
#     """Helper function to truncate the KV cache."""
#     if kv_cache is None or len(kv_cache.key_cache) == 0:
#         return None
#     new_cache = DynamicCache()
#     for layer_idx in range(len(kv_cache)):
#         key, value = kv_cache[layer_idx]
#         new_cache.key_cache.append(key[:, :, :num_tokens_to_keep, :])
#         new_cache.value_cache.append(value[:, :, :num_tokens_to_keep, :])
#     return new_cache

# # from opencompass.models.base import BaseModel, LMTemplateParser

# class PardDreamEngine:
#     """
#     Core logic for speculative decoding using Pard-Dream.
#     This class is instantiated by the OpenCompass model wrapper.
#     """
#     def __init__(self, config: Dict):
#         self.logger = get_logger()
#         self.logger.info("Initializing PardDreamEngine...")
#         self.config = config
#         self.target_model_name = config['target_model_name']
#         self.draft_model_name = config['draft_model_name']

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

#         self._load_models_and_tokenizers()
#         self.mask_id = self.draft_model.config.mask_token_id
#         self.logger.info("PardDreamEngine initialized successfully.")
#         # meta_template = None
#         # self.template_parser = LMTemplateParser(meta_template)
#         # self.eos_token_id = None
#         # if meta_template and 'eos_token_id' in meta_template:
#         #     self.eos_token_id = meta_template['eos_token_id']
#     def _load_models_and_tokenizers(self):
#         """Loads target and draft models and their tokenizers."""
#         self.logger.info(f"Loading target model: {self.target_model_name}")
#         self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_model_name)
#         self.target_model = AutoModelForCausalLM.from_pretrained(
#             self.target_model_name, torch_dtype=self.dtype, trust_remote_code=True
#         ).to(self.device).eval()

#         self.logger.info(f"Loading draft model: {self.draft_model_name}")
#         self.draft_tokenizer = AutoTokenizer.from_pretrained(self.draft_model_name, trust_remote_code=True)
#         self.draft_model = DreamModel.from_pretrained(
#             self.draft_model_name, torch_dtype=self.dtype, trust_remote_code=True
#         ).to(self.device).eval()

#     @torch.no_grad()
#     def generate(self, prompt: str, max_gen_toks: int) -> Tuple[str, Dict]:
#         """
#         Generates text for a single prompt and returns the text and performance metrics.
#         """
#         input_ids = self.target_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

#         target_outputs = self.target_model(input_ids, use_cache=True)
#         target_kv_cache = target_outputs.past_key_values
#         all_generated_tokens = input_ids
#         num_generated, iteration = 0, 0

#         request_accepted = 0
#         request_drafted = 0

#         while num_generated < max_gen_toks:
#             current_length = all_generated_tokens.shape[1]
#             draft_len = min(self.config['max_parallel_draft'], max_gen_toks - num_generated)
#             if draft_len <= 0: break
#             iteration += 1

#             # 1. Draft Generation
#             draft_gen_config = DreamGenerationConfig(max_new_tokens=draft_len, steps=self.config['draft_steps'], alg='maskgit_plus', temperature=self.config['draft_temperature'], mask_token_id=self.mask_id)
#             draft_sequences = self.draft_model.diffusion_generate(inputs=all_generated_tokens, generation_config=draft_gen_config)

#             # 2. Verification and Acceptance
#             draft_tokens = draft_sequences[:, current_length:]
#             draft_text = self.draft_tokenizer.decode(draft_tokens[0], skip_special_tokens=False)
#             mapped_draft_tokens = self.target_tokenizer(draft_text, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)
#             draft_verify_len = mapped_draft_tokens.shape[1]

#             if draft_verify_len > 0:
#                 verification_output = self.target_model(mapped_draft_tokens, past_key_values=target_kv_cache, use_cache=True)

#                 first_draft_logit = target_outputs.logits[:, -1:, :]
#                 remaining_draft_logits = verification_output.logits[:, :-1, :]
#                 target_logits_for_match = torch.cat([first_draft_logit, remaining_draft_logits], dim=1)
#                 target_probs = F.softmax(target_logits_for_match, dim=-1)
#                 _, top_k_indices = torch.topk(target_probs, k=self.config['acceptance_top_k'], dim=-1)
#                 in_top_k = (mapped_draft_tokens.unsqueeze(-1) == top_k_indices).any(dim=-1)
#                 draft_token_probs = torch.gather(target_probs, -1, mapped_draft_tokens.unsqueeze(-1)).squeeze(-1)
#                 above_threshold = draft_token_probs > self.config['acceptance_threshold']
#                 matches = in_top_k & above_threshold
#                 cum_matches = torch.cumprod(matches.int(), dim=1)
#                 num_accepted = torch.sum(cum_matches, dim=1).item()

#                 request_accepted += num_accepted
#                 request_drafted += draft_verify_len
#                 accepted_tokens = mapped_draft_tokens[0, :num_accepted]

#                 if num_accepted < draft_verify_len:
#                     correction_token = target_logits_for_match.argmax(dim=-1)[0, num_accepted].unsqueeze(0)
#                 else:
#                     correction_token = verification_output.logits[:, -1, :].argmax(dim=-1, keepdim=True).view(-1)
#                 newly_generated = torch.cat([accepted_tokens, correction_token]).unsqueeze(0)
#                 full_kv_cache_after_verify = verification_output.past_key_values
#             else: # Fallback for empty draft
#                 correction_token = target_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
#                 newly_generated = correction_token
#                 fallback_output = self.target_model(newly_generated, past_key_values=target_kv_cache, use_cache=True)
#                 full_kv_cache_after_verify = fallback_output.past_key_values

#             # 3. Update state
#             all_generated_tokens = torch.cat([all_generated_tokens, newly_generated], dim=1)
#             num_generated += newly_generated.shape[1]
#             target_kv_cache = _rewind_kv_cache(full_kv_cache_after_verify, all_generated_tokens.shape[1])
#             next_input_ids = all_generated_tokens[:, -1:]
#             target_outputs = self.target_model(next_input_ids, past_key_values=target_kv_cache, use_cache=True)
#             target_kv_cache = target_outputs.past_key_values
#             if self.target_tokenizer.eos_token_id in newly_generated: break

#         output_text = self.target_tokenizer.decode(all_generated_tokens[0, input_ids.shape[1]:], skip_special_tokens=True)
#         metrics = {
#             "accepted_tokens": request_accepted,
#             "drafted_tokens": request_drafted,
#             "draft_attempts": iteration,
#         }
#         return output_text, metrics

# # ===================================================================
# # End: Code migrated from pard_dream_core.py
# # ===================================================================
# from opencompass.registry import MODELS

# @MODELS.register_module()
# class PardDreamModel(BaseModel):
#     """
#     Model wrapper for Pard-Dream speculative decoding in OpenCompass.
#     """

#     def __init__(self,
#                  target_model_name: str,
#                  draft_model_name: str,
#                  max_parallel_draft: int = 32,
#                  draft_temperature: float = 0.0,
#                  draft_steps: int = 1,
#                  acceptance_top_k: int = 1,
#                  acceptance_threshold: float = 0.01,
#                  max_seq_len: int = 2048,
#                  **kwargs):
#         # super().__init__(path=target_model_name, max_seq_len=max_seq_len, model_kwargs=kwargs)
#         from opencompass.models.base import BaseModel, LMTemplateParser
#         meta_template = None
#         self.template_parser = LMTemplateParser(meta_template)
#         self.eos_token_id = None
#         if meta_template and 'eos_token_id' in meta_template:
#             self.eos_token_id = meta_template['eos_token_id']
#         config = {
#             "target_model_name": target_model_name,
#             "draft_model_name": draft_model_name,
#             "max_parallel_draft": max_parallel_draft,
#             "draft_temperature": draft_temperature,
#             "draft_steps": draft_steps,
#             "acceptance_top_k": acceptance_top_k,
#             "acceptance_threshold": acceptance_threshold,
#         }

#         self.engine = PardDreamEngine(config)
#         self.tokenizer = self.engine.target_tokenizer # Expose tokenizer for framework use

#     def generate(self, inputs: List[str], max_out_len: int, **kwargs) -> List[str]:
#         """
#         Generate results given a list of inputs.

#         Args:
#             inputs (List[str]): A list of strings.
#             max_out_len (int): The maximum length of the output.

#         Returns:
#             List[str]: A list of generated strings.
#         """
#         outputs = []
#         for prompt in inputs:
#             # The engine's generate method returns (text, metrics)
#             # We only need the text for the standard generate interface.
#             output_text, metrics = self.engine.generate(prompt=prompt, max_gen_toks=max_out_len)
#             # self.logger.debug(f"Pard-Dream Metrics: {metrics}")
#             outputs.append(output_text)
#         return outputs

#     def get_token_len(self, prompt: str) -> int:
#         """Get lengths of the tokenized strings."""
#         return len(self.tokenizer.encode(prompt))

#     def get_ppl(self,
#                 input_texts: List[str],
#                 mask_length: Optional[List[int]] = None) -> List[float]:
#         """Get perplexity scores."""
#         # Note: Pard-Dream is optimized for generation speed, not direct PPL calculation.
#         # This implementation falls back to the target model for PPL.
#         return self.engine.target_model.get_ppl(input_texts, mask_length)
