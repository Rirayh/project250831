### setup
pip install -r requirements.txt

pip install -e .


### run
opencompass --models hf_qwen2_5_7b_instruct --datasets gsm8k_gen

opencompass --models pard1 --datasets gsm8k_gen

opencompass --models pard2 --datasets gsm8k_gen

opencompass --models pard3 --datasets gsm8k_gen

opencompass --models pard4 --datasets gsm8k_gen

### 服务器环境报dataset错

opencompass --models hf_qwen2_5_7b_instruct --datasets mmlu_pro_gen    

opencompass --models pard1 --datasets mmlu_pro_gen

opencompass --models pard2 --datasets mmlu_pro_gen

opencompass --models pard3 --datasets mmlu_pro_gen

opencompass --models pard4 --datasets mmlu_pro_gen

### 可能可以跑的其他数据集（默认的8batch来评估快些）


  * **MMLU**

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets mmlu_gen
    ```

  * **TriviaQA**

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets triviaqa_gen
    ```

  * **CommonsenseQA**

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets commonsenseqa_gen
    ```

  * **StrategyQA**

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets strategyqa_gen
    ```

  * **PiQA** (Physical Interaction QA)

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets piqa_gen
    ```

  * **OpenBookQA**

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets openbookqa_gen
    ```

  * **AGIEval**

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets agieval_gen
    ```

  * **GSM8K** (Grade School Math 8K)

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets gsm8k_gen
    ```

  * **NQ (Natural Questions)**

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets nq_gen
    ```

  * **RACE**

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets race_gen
    ```

  * **SIQA** (Social IQA)

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets siqa_gen
    ```

  * **MATH**

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets math_gen
    ```

  * **ARC** (AI2 Reasoning Challenge)

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets arc_challenge_gen
    ```

  * **Winogrande**

    ```bash
    opencompass --models hf_qwen2_5_7b_instruct_8batch --datasets winogrande_gen
    ```

-----

### 合并评估命令（想了想算了）


```bash
opencompass --models hf_qwen2_5_7b_instruct_8batch \
--datasets mmlu_gen,triviaqa_gen,commonsenseqa_gen,strategyqa_gen,piqa_gen,openbookqa_gen,agieval_gen,gsm8k_gen,nq_gen,race_gen,siqa_gen,math_gen,arc_challenge_gen,winogrande_gen,tydiqa_goldp_gen
```

-----





### opencompass支持的英文数据集

humaneval, triviaqa, commonsenseqa, tydiqa, strategyqa, cmmlu, lambada, piqa, ceval, math, LCSTS, Xsum, winogrande, openbookqa, AGIEval, gsm8k, nq, race, siqa, mbpp, mmlu, hellaswag, ARC, BBH, xstory_cloze, summedits, GAOKAO-BENCH, OCNLI, cmnli
好的，在您列出的这些数据集中，以下是**以QA（Question Answering，问答）形式为主**并且是**英文**的数据集：

我将它们分为几类，以便您更好地理解它们的侧重点：

#### 1. 综合知识与考试类问答 (General Knowledge & Exam QA)

这类数据集考察模型广泛的知识储备和在标准化考试中的表现。

* **MMLU (Massive Multitask Language Understanding)**: 涵盖57个学科的英文多项选择题，是衡量模型综合能力的核心基准。
* **TriviaQA**: 基于互联网上的小知识问题，需要模型在大量文本证据中寻找答案。
* **NQ (Natural Questions)**: 来自真实的谷歌搜索查询，要求模型从维基百科页面中找出答案。
* **ARC (AI2 Reasoning Challenge)**: 源自美国中小学科学考试的问答题。
* **AGIEval**: 包含了美国SAT、LSAT、GRE等多种英文标准化考试的题目。
* **RACE**: 来自中国初高中的**英文**阅读理解考试题。
* **OpenBookQA**: “开卷考试”形式的科学问答，需要结合给定的科学事实来回答问题。

#### 2. 常识推理问答 (Commonsense Reasoning QA)

这类数据集考察模型是否具备人类的常识并能进行相应的推理。

* **CommonsenseQA**: 需要利用常识才能回答的选择题。
* **PiQA (Physical Interaction QA)**: 关于物理世界互动常识的问答。
* **SIQA (Social Interaction QA)**: 关于社交场景和情商的常识问答。
* **StrategyQA**: 需要进行多步推理才能回答的“是/否”问题，并要求解释原因。
* **Winogrande**: 以选择题的形式考察模型对代词指代的理解，这背后需要常识支持。

#### 3. 数学与逻辑推理问答 (Math & Logical Reasoning QA)

这类数据集专门评测模型的数学计算和逻辑能力。

* **GSM8K**: 小学水平的数学应用题，需要多步推理。
* **MATH**: 竞赛水平的数学难题。

#### 4. 多语言问答 (包含英文)

* **TyDi QA**: 这是一个多语言问答数据集，它**包含了英文**以及其他10种语言，专门用于测试模型的跨语言能力。

---

**总结一下，直接回答您问题的核心列表是：**

* **MMLU**
* **TriviaQA**
* **CommonsenseQA**
* **StrategyQA**
* **PiQA**
* **OpenBookQA**
* **AGIEval**
* **GSM8K**
* **NQ (Natural Questions)**
* **RACE**
* **SIQA**
* **MATH**
* **ARC**
* **Winogrande**
* **TyDi QA** (包含英文)

**非英文的QA数据集**：

* **C-Eval**, **CMMLU**, **GAOKAO-BENCH**: 这些是**中文**的QA数据集。
