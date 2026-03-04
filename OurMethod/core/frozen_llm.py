"""Step 4 — 冻结 LLM 编码器（支持多模型注册与选择）。

五种实现：
  1. StubFrozenLLM          — 无外部依赖，hash(prompt) → 确定性 h_t + 伪 ds_llm
  2. RealFrozenLLM          — HuggingFace Encoder 模型（DistilBERT 等，向后兼容）
  3. GenerativeFrozenLLM    — HuggingFace Causal LM + PyTorch (SmolLM2 等)
  4. MLXGenerativeFrozenLLM — Apple MLX 原生加速 (Apple Silicon 专用，2-3x 更快)
       h_t  = 最后输入 token 的隐藏状态 (L2 归一化)
       ds_llm = 模型生成文本 → 解析 JSON → DsLlm 列表
  5. FrozenLLMRegistry      — 注册多个冻结 LLM，运行时按名选择

在线流程统一调用 encode(prompt, num_arms) → (h_t, ds_llm | None)。
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import os

_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

from typing import Dict, List, Optional, Tuple

import numpy as np

from OurMethod.core.protocol import DsLlm, FrozenLLMEncoder

_HAS_TRANSFORMERS = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM  # type: ignore
    _HAS_TRANSFORMERS = True
except ImportError:
    pass

_HAS_MLX = False
try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    from mlx_lm import load as mlx_load, generate as mlx_generate  # type: ignore
    _HAS_MLX = True
except ImportError:
    pass


# ====================================================================== #
#  1. StubFrozenLLM — 测试/基线用                                         #
# ====================================================================== #

class StubFrozenLLM(FrozenLLMEncoder):
    """Stub 模式：确定性伪特征 + 伪 ds_llm。无任何外部依赖。"""

    def __init__(self, hidden_dim: int = 128, generate_ds_llm: bool = True):
        self._dim = hidden_dim
        self._gen_ds = generate_ds_llm

    def encode(
        self, prompt: str, num_arms: int = 0,
    ) -> Tuple[np.ndarray, Optional[List[DsLlm]]]:
        h_t = self._hash_to_vec(prompt, self._dim)
        ds_llm: Optional[List[DsLlm]] = None
        if self._gen_ds and num_arms > 0:
            ds_llm = self._stub_ds_llm(prompt, num_arms)
        return h_t, ds_llm

    def get_hidden_dim(self) -> int:
        return self._dim

    @property
    def generate_ds_llm(self) -> bool:
        return self._gen_ds

    @generate_ds_llm.setter
    def generate_ds_llm(self, value: bool):
        self._gen_ds = value

    # ---- 内部 ---- #
    @staticmethod
    def _hash_to_vec(text: str, dim: int) -> np.ndarray:
        """确定性：相同 text → 相同向量，L2 归一化。"""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.RandomState(seed)
        v = rng.randn(dim).astype(np.float64)
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v

    @staticmethod
    def _stub_ds_llm(prompt: str, num_arms: int) -> List[DsLlm]:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.RandomState(seed + 1)
        out: List[DsLlm] = []
        for i in range(num_arms):
            out.append(DsLlm(
                arm_id=i,
                predicted_mean=float(np.clip(rng.rand(), 0.05, 0.95)),
                predicted_std=float(np.clip(rng.rand() * 0.3 + 0.05, 0.05, 0.5)),
                reasoning=f"stub prediction for arm {i}",
            ))
        return out


# ====================================================================== #
#  2. RealFrozenLLM — Encoder 模型（向后兼容）                              #
# ====================================================================== #

class RealFrozenLLM(FrozenLLMEncoder):
    """HuggingFace Encoder 模型冻结提取 h_t（如 DistilBERT）。
    如果 transformers 不可用，自动回退到 StubFrozenLLM。"""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: Optional[str] = None,
        max_length: int = 128,
        generate_ds_llm: bool = True,
    ):
        self._gen_ds = generate_ds_llm
        self._fallback: Optional[StubFrozenLLM] = None
        self._dim = 0

        if not _HAS_TRANSFORMERS:
            self._fallback = StubFrozenLLM(hidden_dim=128, generate_ds_llm=generate_ds_llm)
            self._dim = 128
            return

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)
        self._dim = self.model.config.hidden_size

    def encode(
        self, prompt: str, num_arms: int = 0,
    ) -> Tuple[np.ndarray, Optional[List[DsLlm]]]:
        if self._fallback is not None:
            return self._fallback.encode(prompt, num_arms)

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            max_length=self.max_length, truncation=True, padding="max_length",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        h_t = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype(np.float64)
        ds_llm: Optional[List[DsLlm]] = None
        if self._gen_ds and num_arms > 0:
            ds_llm = StubFrozenLLM._stub_ds_llm(prompt, num_arms)
        return h_t, ds_llm

    def get_hidden_dim(self) -> int:
        return self._dim

    @property
    def generate_ds_llm(self) -> bool:
        return self._gen_ds

    @generate_ds_llm.setter
    def generate_ds_llm(self, value: bool):
        self._gen_ds = value


# ====================================================================== #
#  JSON 解析工具（从 LLM 生成文本提取 DsLlm）                               #
# ====================================================================== #

def _parse_ds_llm_from_text(text: str, num_arms: int) -> List[DsLlm]:
    """从 LLM 生成文本中鲁棒提取 DsLlm 列表。

    尝试顺序：
      1. Markdown 代码块内的 JSON 数组
      2. 裸 JSON 数组
      3. 散落的 JSON 对象
      4. 回退：均匀预测 + 高不确定性
    """
    json_str: Optional[str] = None

    # 1) Markdown code block
    md = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if md:
        json_str = md.group(1)

    # 2) Bare JSON array
    if json_str is None:
        arr = re.search(
            r'\[\s*\{[^[\]]*?"arm_id"[^[\]]*?\}\s*(?:,\s*\{[^[\]]*?\}\s*)*\]',
            text, re.DOTALL,
        )
        if arr:
            json_str = arr.group(0)

    # 3) Scattered JSON objects
    if json_str is None:
        objs = re.findall(r'\{[^{}]*?"arm_id"\s*:\s*\d+[^{}]*?\}', text)
        if objs:
            json_str = "[" + ", ".join(objs) + "]"

    # Parse
    if json_str:
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                ds: List[DsLlm] = []
                seen: set = set()
                for item in parsed:
                    if isinstance(item, dict) and "arm_id" in item:
                        aid = int(item["arm_id"])
                        if 0 <= aid < num_arms and aid not in seen:
                            seen.add(aid)
                            ds.append(DsLlm(
                                arm_id=aid,
                                predicted_mean=float(np.clip(
                                    item.get("predicted_mean", 0.5), 0.0, 1.0)),
                                predicted_std=float(np.clip(
                                    item.get("predicted_std", 0.3), 0.01, 1.0)),
                                reasoning=str(item.get("reasoning", "")),
                                raw_text=text,
                            ))
                # Fill missing arms
                for a in range(num_arms):
                    if a not in seen:
                        ds.append(DsLlm(
                            arm_id=a,
                            predicted_mean=0.5,
                            predicted_std=0.5,
                            reasoning="(missing from LLM output)",
                            raw_text=text,
                        ))
                ds.sort(key=lambda x: x.arm_id)
                return ds
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # 4) Fallback
    return [
        DsLlm(
            arm_id=a,
            predicted_mean=1.0 / max(num_arms, 1),
            predicted_std=0.5,
            reasoning="(fallback: LLM output unparseable)",
            raw_text=text,
        )
        for a in range(num_arms)
    ]


# ====================================================================== #
#  3. GenerativeFrozenLLM — 生成式 Causal LM                              #
# ====================================================================== #

class GenerativeFrozenLLM(FrozenLLMEncoder):
    """生成式冻结 LLM（SmolLM2-360M-Instruct, Qwen2.5-0.5B-Instruct 等）。

    h_t   = 输入 prompt 最后一个 token 的隐藏状态（L2 归一化）
    ds_llm = 模型生成文本 → _parse_ds_llm_from_text → DsLlm 列表

    Args:
        model_name:        HuggingFace 模型标识符或本地路径
        device:            推理设备 (None=自动检测 cuda→mps→cpu)
        max_input_length:  输入最大 token 数
        max_new_tokens:    生成最大 token 数
        temperature:       生成温度 (0=贪心, >0=采样)
        use_chat_template: 是否用 tokenizer 的 chat_template 包装 prompt
        generate_ds_llm:   是否生成 ds_llm (False 则只提取 h_t，速度更快)
        system_prompt:     系统提示词 (仅 chat_template 模式)
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        device: Optional[str] = None,
        max_input_length: int = 1024,
        max_new_tokens: int = 384,
        temperature: float = 0.1,
        use_chat_template: bool = True,
        generate_ds_llm: bool = True,
        use_half: bool = True,
        system_prompt: str = (
            "You are an expert decision-making assistant. "
            "Always respond with valid JSON only, no extra text."
        ),
    ):
        # ============================================================ #
        #  参数指南 (Parameter Guide)
        # ============================================================ #
        #
        # 🔴 重点关注
        #
        #   temperature (float, default=0.1)
        #       生成温度。0=贪心(确定性), >0=采样(随机性)。
        #       冷启动预测: 建议 0.1~0.3 (低随机性, 稳定预测)。
        #       若想增加 ds_llm 多样性可升至 0.5+，但可能降低质量。
        #
        #   max_new_tokens (int, default=384)
        #       单次生成的最大 token 数。必须足够覆盖所有 arm 的 JSON 输出。
        #       经验公式: ~50 tokens/arm。7 arms ≈ 350 tokens。
        #       太小 → 输出截断，部分 arm 丢失；太大 → 浪费时间。
        #       典型范围: [256, 512]，arm 数多时需增大。
        #
        #   generate_ds_llm (bool, default=True)
        #       是否生成文本并解析 ds_llm。
        #       冷启动阶段必须 True；在线阶段设 False 可大幅提速
        #       (只提取 h_t，跳过文本生成，速度从 ~8s → ~0.3s/步)。
        #
        # 🟡 可按需调整
        #
        #   model_name (str)
        #       HuggingFace 模型标识符或本地路径。
        #       更换模型会改变 hidden_dim，需同步更新 Compressor 的 input_dim。
        #       注册新模型后通过 FrozenLLMRegistry 管理。
        #
        #   max_input_length (int, default=1024)
        #       输入 prompt 的最大 token 数。超出会被截断。
        #       Statlog (9维特征, 7 arms) 通常 ~300 tokens，1024 足够。
        #       如果特征维度很大或反馈条目多，可能需要增大。
        #
        #   system_prompt (str)
        #       系统提示词，仅在 chat_template 模式下使用。
        #       当前通用设计已适配 JSON 输出。
        #       针对特定领域可自定义，但需确保引导模型输出 JSON。
        #
        # 🟢 最好不调整
        #
        #   device (str, default=auto)
        #       自动检测: cuda → mps → cpu。手动指定仅在特殊情况。
        #
        #   use_chat_template (bool, default=True)
        #       保持 True。SmolLM2/Qwen 等 Instruct 模型需要 chat 格式。
        #       设为 False 可能导致生成质量严重下降。
        #
        #   use_half (bool, default=True)
        #       GPU/MPS 上用 float16 加速推理。CPU 上自动忽略。
        #       除非遇到精度问题，否则保持 True。
        # ============================================================ #
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "GenerativeFrozenLLM requires `transformers` and `torch`. "
                "Install: pip install transformers torch"
            )

        self._gen_ds = generate_ds_llm
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt
        self.model_name = model_name

        # Device auto-detect
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # dtype: float16 on GPU/MPS for speed, float32 on CPU
        model_dtype = torch.float32
        if use_half and device in ("cuda", "mps"):
            model_dtype = torch.float16
        self._model_dtype = model_dtype

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model (frozen)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=model_dtype,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)
        self._dim = self.model.config.hidden_size

    # ------------------------------------------------------------------ #
    def encode(
        self, prompt: str, num_arms: int = 0,
    ) -> Tuple[np.ndarray, Optional[List[DsLlm]]]:
        """编码 prompt → (h_t, ds_llm)。

        h_t:    最后输入 token 的最后一层隐藏状态, L2 归一化, (hidden_size,)
        ds_llm: 生成文本解析出的 DsLlm 列表 (仅 generate_ds_llm=True 且 num_arms>0)
        """
        input_text = self._prepare_input(prompt)

        inputs = self.tokenizer(
            input_text, return_tensors="pt",
            max_length=self.max_input_length, truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attn_mask = inputs.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        # ---- h_t: 最后输入 token 的隐藏状态 ---- #
        with torch.no_grad():
            out = self.model(
                input_ids, attention_mask=attn_mask,
                output_hidden_states=True,
            )
        last_hidden = out.hidden_states[-1]              # (1, seq, hidden)
        h_t = last_hidden[0, -1, :].float().cpu().numpy().astype(np.float64)
        nrm = np.linalg.norm(h_t)
        if nrm > 0:
            h_t = h_t / nrm

        # ---- ds_llm: 生成文本 → JSON 解析 ---- #
        ds_llm: Optional[List[DsLlm]] = None
        if self._gen_ds and num_arms > 0:
            ds_llm = self._generate_ds_llm(input_ids, attn_mask, num_arms)

        return h_t, ds_llm

    def get_hidden_dim(self) -> int:
        return self._dim

    @property
    def generate_ds_llm(self) -> bool:
        return self._gen_ds

    @generate_ds_llm.setter
    def generate_ds_llm(self, value: bool):
        self._gen_ds = value

    # ---- 内部方法 ---- #

    def _prepare_input(self, prompt: str) -> str:
        """用 chat template 包装 prompt（如已配置）。"""
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass
        return prompt

    def _generate_ds_llm(
        self, input_ids, attn_mask, num_arms: int,
    ) -> List[DsLlm]:
        """调用 model.generate 生成文本并解析为 DsLlm。"""
        gen_kwargs: Dict = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
        }
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            gen_out = self.model.generate(
                input_ids, attention_mask=attn_mask, **gen_kwargs,
            )

        gen_tokens = gen_out[0, input_ids.shape[1]:]
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return _parse_ds_llm_from_text(gen_text, num_arms)


# ====================================================================== #
#  4. MLXGenerativeFrozenLLM — Apple MLX 原生加速                          #
# ====================================================================== #

class MLXGenerativeFrozenLLM(FrozenLLMEncoder):
    """基于 Apple MLX 框架的冻结 LLM，专为 Apple Silicon 优化。

    相比 PyTorch MPS 版 GenerativeFrozenLLM：
      - 生成速度 ~2.5x 更快 (SmolLM2-360M, M4)
      - 编码速度 ~1.4x 更快
      - 统一内存，无 CPU↔GPU 拷贝开销

    接口与 GenerativeFrozenLLM 完全一致：
      h_t   = 最后输入 token 的隐藏状态 (L2 归一化)
      ds_llm = 生成文本 → JSON 解析 → DsLlm

    Args:
        model_name:       HuggingFace 模型标识符
        max_input_length: 输入最大 token 数
        max_new_tokens:   生成最大 token 数
        temperature:      生成温度
        generate_ds_llm:  是否生成 ds_llm
        system_prompt:    系统提示词
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        max_input_length: int = 1024,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        generate_ds_llm: bool = True,
        system_prompt: str = (
            "You are an expert decision-making assistant. "
            "Always respond with valid JSON only, no extra text."
        ),
    ):
        if not _HAS_MLX:
            raise ImportError(
                "MLXGenerativeFrozenLLM requires `mlx` and `mlx-lm`. "
                "Install: pip install mlx mlx-lm"
            )

        self._gen_ds = generate_ds_llm
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.model_name = model_name

        # 加载 MLX 模型
        self.model, self.tokenizer = mlx_load(model_name)
        self._dim = self.model.args.hidden_size

    # ------------------------------------------------------------------ #
    def encode(
        self, prompt: str, num_arms: int = 0,
    ) -> Tuple[np.ndarray, Optional[List[DsLlm]]]:
        """编码 prompt → (h_t, ds_llm)。"""
        input_text = self._prepare_input(prompt)
        input_ids = self.tokenizer.encode(input_text)
        if len(input_ids) > self.max_input_length:
            input_ids = input_ids[:self.max_input_length]
        input_arr = mx.array([input_ids])

        # ---- h_t: base model → 最后 token 隐藏状态 ---- #
        hidden = self.model.model(input_arr)       # (1, seq, hidden_size)
        mx.eval(hidden)
        last_vec = hidden[0, -1, :]                # (hidden_size,)
        h_t = np.array(last_vec.tolist(), dtype=np.float64)
        nrm = np.linalg.norm(h_t)
        if nrm > 0:
            h_t = h_t / nrm

        # ---- ds_llm: 生成文本 → JSON 解析 ---- #
        ds_llm: Optional[List[DsLlm]] = None
        if self._gen_ds and num_arms > 0:
            ds_llm = self._generate_ds_llm(prompt, num_arms)

        return h_t, ds_llm

    def get_hidden_dim(self) -> int:
        return self._dim

    @property
    def generate_ds_llm(self) -> bool:
        return self._gen_ds

    @generate_ds_llm.setter
    def generate_ds_llm(self, value: bool):
        self._gen_ds = value

    # ---- 内部方法 ---- #

    def _prepare_input(self, prompt: str) -> str:
        """用 chat template 包装 prompt。"""
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass
        return prompt

    def _generate_ds_llm(
        self, prompt: str, num_arms: int,
    ) -> List[DsLlm]:
        """调用 mlx_lm.generate 生成文本并解析为 DsLlm。"""
        input_text = self._prepare_input(prompt)
        gen_text = mlx_generate(
            self.model, self.tokenizer,
            prompt=input_text,
            max_tokens=self.max_new_tokens,
            verbose=False,
        )
        return _parse_ds_llm_from_text(gen_text, num_arms)


# ====================================================================== #
#  5. FrozenLLMRegistry — 多模型注册与选择                                  #
# ====================================================================== #

class FrozenLLMRegistry:
    """管理多个冻结 LLM 实例，运行时按名选择。

    用法::

        registry = FrozenLLMRegistry()
        registry.register("stub", StubFrozenLLM())
        registry.register("smollm2", GenerativeFrozenLLM(...), default=True)
        llm = registry.get("smollm2")          # 按名选
        llm = registry.get_default()            # 选默认
    """

    def __init__(self):
        self._models: Dict[str, FrozenLLMEncoder] = {}
        self._default: Optional[str] = None

    def register(self, name: str, llm: FrozenLLMEncoder, default: bool = False):
        """注册一个冻结 LLM 实例。"""
        self._models[name] = llm
        if default or self._default is None:
            self._default = name

    def get(self, name: str) -> FrozenLLMEncoder:
        """按名获取 LLM 实例。"""
        if name not in self._models:
            avail = ", ".join(self._models.keys()) or "(empty)"
            raise KeyError(f"LLM '{name}' not registered. Available: {avail}")
        return self._models[name]

    def get_default(self) -> FrozenLLMEncoder:
        """获取默认 LLM。"""
        if self._default is None:
            raise RuntimeError("No LLM registered.")
        return self._models[self._default]

    @property
    def default_name(self) -> Optional[str]:
        return self._default

    def list_models(self) -> List[str]:
        """列出所有已注册模型名称。"""
        return list(self._models.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def __len__(self) -> int:
        return len(self._models)
