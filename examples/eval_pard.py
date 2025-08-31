from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets
    from opencompass.models.pard import \
        PardDreamModel

datasets = gsm8k_datasets
models = PardDreamModel
