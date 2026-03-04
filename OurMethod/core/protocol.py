"""Step 0 — 主干协议: 三大协议对象 + 四大 ABC 接口。

协议对象（一旦固定，后续只增不改）：
  Context   — 上下文（含 user/machine id、环境字段、统计特征）
  Arm       — 候选动作
  DsLlm     — LLM 对单个 arm 的奖励分布预测
  DecisionRecord — 一次完整决策记录（prompt / h / z / ds_llm / reward / 3种半径 / …)

四大接口 (ABC)：
  PromptBuilder     .build(context, arms, last_feedback) -> str
  FrozenLLMEncoder  .encode(prompt, num_arms) -> (h_t, ds_llm | None)
  Compressor        .forward(h_t, extra) -> z_t
  Policy            .select(context, arms, z_t, ds_llm) -> arm_id
                    .update(record)
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ====================================================================== #
#                        协议对象                                          #
# ====================================================================== #

@dataclass
class Context:
    """上下文，包含用户/设备标识和环境特征。"""
    features: np.ndarray                                    # (d,)
    user_id: Optional[str] = None
    machine_id: Optional[str] = None
    env_fields: Dict[str, Any] = field(default_factory=dict)
    stat_features: Optional[np.ndarray] = None              # 可选交互统计

    def dim(self) -> int:
        return int(self.features.shape[0])

    # ---------- 序列化 ---------- #
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "user_id": self.user_id,
            "machine_id": self.machine_id,
            "dim": self.dim(),
            "features": self.features.tolist(),
            "env_fields": self.env_fields,
        }
        if self.stat_features is not None:
            d["stat_features"] = self.stat_features.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Context":
        return cls(
            features=np.array(d["features"], dtype=np.float64),
            user_id=d.get("user_id"),
            machine_id=d.get("machine_id"),
            env_fields=d.get("env_fields", {}),
            stat_features=(np.array(d["stat_features"], dtype=np.float64)
                           if d.get("stat_features") else None),
        )


@dataclass
class Arm:
    """一个候选动作。"""
    arm_id: int
    name: Optional[str] = None
    description: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "name": self.name or f"arm_{self.arm_id}",
            "description": self.description,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Arm":
        return cls(
            arm_id=d["arm_id"],
            name=d.get("name"),
            description=d.get("description"),
            attributes=d.get("attributes", {}),
        )


@dataclass
class DsLlm:
    """LLM 对单个 arm 的奖励分布预测。"""
    arm_id: int
    predicted_mean: float = 0.0
    predicted_std: float = 1.0
    reasoning: str = ""
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "predicted_mean": self.predicted_mean,
            "predicted_std": self.predicted_std,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DsLlm":
        return cls(
            arm_id=d["arm_id"],
            predicted_mean=d.get("predicted_mean", 0.0),
            predicted_std=d.get("predicted_std", 1.0),
            reasoning=d.get("reasoning", ""),
        )


@dataclass
class DecisionRecord:
    """一次完整决策记录 — 包含落盘日志所需的全部字段。"""
    step: int
    context: Context
    arms: List[Arm]
    prompt: str                                         # 完整 prompt 文本
    h_t: Optional[np.ndarray] = None                    # LLM 隐层向量
    z_t: Optional[np.ndarray] = None                    # 压缩后特征
    ds_llm: Optional[List[DsLlm]] = None                # LLM 分布预测
    chosen_arm: int = -1
    reward: float = 0.0
    optimal_reward: Optional[float] = None
    regret: Optional[float] = None
    ucb_values: Optional[List[float]] = None             # 各 arm 最终 score
    radius_ucb1: Optional[List[float]] = None            # UCB1 半径
    radius_linucb: Optional[List[float]] = None          # LinUCB 半径
    radius_llm: Optional[List[float]] = None             # LLM 仿真半径
    algorithm: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.optimal_reward is not None and self.regret is None:
            self.regret = self.optimal_reward - self.reward

    # ---------- 序列化 ---------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":           self.step,
            "context":        self.context.to_dict(),
            "arms":           [a.to_dict() for a in self.arms],
            "prompt":         self.prompt,
            "h_t":            self.h_t.tolist() if self.h_t is not None else None,
            "z_t":            self.z_t.tolist() if self.z_t is not None else None,
            "ds_llm":         [x.to_dict() for x in self.ds_llm] if self.ds_llm else None,
            "chosen_arm":     self.chosen_arm,
            "reward":         self.reward,
            "optimal_reward": self.optimal_reward,
            "regret":         self.regret,
            "ucb_values":     self.ucb_values,
            "radius_ucb1":    self.radius_ucb1,
            "radius_linucb":  self.radius_linucb,
            "radius_llm":     self.radius_llm,
            "algorithm":      self.algorithm,
            "timestamp":      self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionRecord":
        return cls(
            step=d["step"],
            context=Context.from_dict(d["context"]),
            arms=[Arm.from_dict(a) for a in d["arms"]],
            prompt=d.get("prompt", ""),
            h_t=np.array(d["h_t"], dtype=np.float64) if d.get("h_t") else None,
            z_t=np.array(d["z_t"], dtype=np.float64) if d.get("z_t") else None,
            ds_llm=([DsLlm.from_dict(x) for x in d["ds_llm"]]
                    if d.get("ds_llm") else None),
            chosen_arm=d.get("chosen_arm", -1),
            reward=d.get("reward", 0.0),
            optimal_reward=d.get("optimal_reward"),
            regret=d.get("regret"),
            ucb_values=d.get("ucb_values"),
            radius_ucb1=d.get("radius_ucb1"),
            radius_linucb=d.get("radius_linucb"),
            radius_llm=d.get("radius_llm"),
            algorithm=d.get("algorithm", ""),
            timestamp=d.get("timestamp", 0.0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "DecisionRecord":
        return cls.from_dict(json.loads(s))


# ====================================================================== #
#                        四大接口 (ABC)                                    #
# ====================================================================== #

class PromptBuilder(ABC):
    """Prompt 生成器接口。"""

    @abstractmethod
    def build(
        self,
        context: Context,
        arms: List[Arm],
        last_feedback: Optional[List[DecisionRecord]] = None,
    ) -> str:
        """根据上下文、候选 arm、历史反馈生成 prompt 文本。"""
        ...


class FrozenLLMEncoder(ABC):
    """冻结 LLM 编码器接口。"""

    @abstractmethod
    def encode(
        self, prompt: str, num_arms: int = 0,
    ) -> Tuple[np.ndarray, Optional[List[DsLlm]]]:
        """输入 prompt，返回 (h_t, ds_llm | None)。"""
        ...

    @abstractmethod
    def get_hidden_dim(self) -> int:
        ...


class Compressor(ABC):
    """特征压缩器接口 fφ : h_t → z_t。"""

    @abstractmethod
    def forward(
        self, h_t: np.ndarray, extra: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """h_t (d_h,) -> z_t (d_z,)。extra 为可选拼接统计向量。"""
        ...

    @abstractmethod
    def get_output_dim(self) -> int:
        ...


class Policy(ABC):
    """在线决策策略接口。"""

    @abstractmethod
    def select(
        self,
        context: Context,
        arms: List[Arm],
        z_t: np.ndarray,
        ds_llm: Optional[List[DsLlm]] = None,
    ) -> int:
        """返回选中的 arm_id。"""
        ...

    @abstractmethod
    def update(self, record: DecisionRecord):
        """用一条完整记录更新内部状态。"""
        ...
