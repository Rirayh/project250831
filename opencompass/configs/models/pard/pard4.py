# -*- coding: utf-8 -*-
from opencompass.models import PardDreamModel

models = [
    dict(
        # 指定使用我们刚刚创建的 PardDreamModel 类
        type=PardDreamModel,
        
        # ----- Pard-Dream 核心参数 -----
        # 目标模型 (高质量)
        target_model_name="Qwen/Qwen2.5-7B-Instruct",
        # 草稿模型 (高速度)
        draft_model_name="Dream-org/Dream-v0-Instruct-7B",

        # ----- 推测解码超参数 -----
        max_parallel_draft=256,
        draft_temperature=0.0,
        draft_steps=1,
        acceptance_top_k=100,
        acceptance_threshold=0.01,

        # ----- OpenCompass 标准参数 -----
        max_out_len=256,
        max_seq_len=2048,
        batch_size=1, # PardDreamEngine 当前实现为逐个处理
        run_cfg=dict(num_gpus=1, num_procs=1),
        abbr='pard-qwen2.5-7b-tp100-th001' # 为模型指定一个简洁的别名
    )
]
