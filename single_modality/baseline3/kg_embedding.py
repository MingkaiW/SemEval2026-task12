"""
SemEval 2026 Task 12: Abductive Event Reasoning
Baseline 3: Knowledge Graph Embedding Module

实现多种KG Embedding方法:
1. TransE - 翻译距离模型
2. ComplEx - 复数空间嵌入
3. RotatE - 旋转模型

参考论文:
- TransE: Translating Embeddings for Modeling Multi-relational Data (NeurIPS 2013)
- ComplEx: Complex Embeddings for Simple Link Prediction (ICML 2016)
- RotatE: Knowledge Graph Embedding by Relational Rotation (ICLR 2019)

参考实现:
- https://github.com/thunlp/OpenKE
- https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
- https://github.com/torchkge-team/torchkge
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np


@dataclass
class KGEConfig:
    """KG Embedding 配置"""
    embedding_dim: int = 256
    margin: float = 1.0  # TransE margin
    learning_rate: float = 0.001
    batch_size: int = 256
    num_epochs: int = 100
    negative_samples: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TransE(nn.Module):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data

    核心思想: h + r ≈ t
    对于三元组 (head, relation, tail)，头实体加上关系约等于尾实体

    损失函数: max(0, margin + d(h+r, t) - d(h'+r, t'))
    其中 d 是 L1 或 L2 距离
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        margin: float = 1.0,
        norm: int = 1  # L1 or L2
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.norm = norm

        # 实体和关系嵌入
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # 归一化关系嵌入
        with torch.no_grad():
            self.relation_embeddings.weight.data = F.normalize(
                self.relation_embeddings.weight.data, p=2, dim=1
            )

    def _normalize_entities(self):
        """归一化实体嵌入（每个batch后调用）"""
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        计算三元组的得分（距离越小越好）

        Args:
            heads: (batch_size,) 头实体索引
            relations: (batch_size,) 关系索引
            tails: (batch_size,) 尾实体索引

        Returns:
            scores: (batch_size,) 距离得分
        """
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)

        # h + r - t 的范数
        score = torch.norm(h + r - t, p=self.norm, dim=1)
        return score

    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_tails: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Margin Ranking Loss

        L = max(0, margin + d_pos - d_neg)
        """
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, pos_relations, neg_tails)

        loss = torch.relu(self.margin + pos_score - neg_score)
        return loss.mean()

    def get_entity_embedding(self, entity_id: int) -> torch.Tensor:
        """获取单个实体的嵌入"""
        idx = torch.tensor([entity_id], device=self.entity_embeddings.weight.device)
        return self.entity_embeddings(idx).squeeze(0)

    def get_relation_embedding(self, relation_id: int) -> torch.Tensor:
        """获取单个关系的嵌入"""
        idx = torch.tensor([relation_id], device=self.relation_embeddings.weight.device)
        return self.relation_embeddings(idx).squeeze(0)


class ComplEx(nn.Module):
    """
    ComplEx: Complex Embeddings for Simple Link Prediction

    核心思想: 使用复数空间建模实体和关系
    得分函数: Re(<h, r, conj(t)>) = Re(Σ h_i * r_i * conj(t_i))

    优势: 可以建模非对称关系
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        reg_weight: float = 0.01
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.reg_weight = reg_weight

        # 实部和虚部分开存储
        self.entity_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_im = nn.Embedding(num_entities, embedding_dim)
        self.relation_re = nn.Embedding(num_relations, embedding_dim)
        self.relation_im = nn.Embedding(num_relations, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化"""
        for emb in [self.entity_re, self.entity_im,
                    self.relation_re, self.relation_im]:
            nn.init.xavier_uniform_(emb.weight)

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        计算三元组得分

        score = Re(<h, r, conj(t)>)
              = h_re * r_re * t_re
              + h_re * r_im * t_im
              + h_im * r_re * t_im
              - h_im * r_im * t_re
        """
        h_re = self.entity_re(heads)
        h_im = self.entity_im(heads)
        r_re = self.relation_re(relations)
        r_im = self.relation_im(relations)
        t_re = self.entity_re(tails)
        t_im = self.entity_im(tails)

        score = (
            (h_re * r_re * t_re).sum(dim=1) +
            (h_re * r_im * t_im).sum(dim=1) +
            (h_im * r_re * t_im).sum(dim=1) -
            (h_im * r_im * t_re).sum(dim=1)
        )

        return score

    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Binary Cross Entropy Loss + L2 正则化
        """
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, pos_relations, neg_tails)

        # Sigmoid + BCE
        pos_loss = F.softplus(-pos_score).mean()
        neg_loss = F.softplus(neg_score).mean()

        # L2 正则化
        reg = self._regularization(pos_heads, pos_relations, pos_tails)

        return pos_loss + neg_loss + self.reg_weight * reg

    def _regularization(self, heads, relations, tails):
        """L2 正则化"""
        reg = (
            self.entity_re(heads).norm(p=2, dim=1).mean() +
            self.entity_im(heads).norm(p=2, dim=1).mean() +
            self.relation_re(relations).norm(p=2, dim=1).mean() +
            self.relation_im(relations).norm(p=2, dim=1).mean() +
            self.entity_re(tails).norm(p=2, dim=1).mean() +
            self.entity_im(tails).norm(p=2, dim=1).mean()
        )
        return reg / 6

    def get_entity_embedding(self, entity_id: int) -> torch.Tensor:
        """获取实体嵌入（拼接实部和虚部）"""
        idx = torch.tensor([entity_id], device=self.entity_re.weight.device)
        re = self.entity_re(idx).squeeze(0)
        im = self.entity_im(idx).squeeze(0)
        return torch.cat([re, im], dim=0)


class RotatE(nn.Module):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space

    核心思想: t = h ◦ r (复数空间中的元素乘法/旋转)
    关系被建模为复平面上的旋转

    优势:
    - 可以建模对称、反对称、逆、组合关系
    - 比TransE和ComplEx表达能力更强
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        margin: float = 9.0,
        epsilon: float = 2.0
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.epsilon = epsilon

        # 实体嵌入（复数）
        self.entity_re = nn.Embedding(num_entities, embedding_dim)
        self.entity_im = nn.Embedding(num_entities, embedding_dim)

        # 关系嵌入（相位/角度）
        self.relation_phase = nn.Embedding(num_relations, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化"""
        embedding_range = (self.margin + self.epsilon) / self.embedding_dim

        nn.init.uniform_(
            self.entity_re.weight, -embedding_range, embedding_range
        )
        nn.init.uniform_(
            self.entity_im.weight, -embedding_range, embedding_range
        )
        nn.init.uniform_(
            self.relation_phase.weight, -math.pi, math.pi
        )

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        计算得分: ||h ◦ r - t||

        h ◦ r = (h_re + i*h_im) * (cos(θ) + i*sin(θ))
              = (h_re*cos(θ) - h_im*sin(θ)) + i*(h_re*sin(θ) + h_im*cos(θ))
        """
        h_re = self.entity_re(heads)
        h_im = self.entity_im(heads)
        t_re = self.entity_re(tails)
        t_im = self.entity_im(tails)

        phase = self.relation_phase(relations)
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)

        # 复数乘法
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # 距离
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im

        score = torch.sqrt(diff_re ** 2 + diff_im ** 2).sum(dim=1)
        return score

    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_tails: torch.Tensor
    ) -> torch.Tensor:
        """Self-adversarial negative sampling loss"""
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, pos_relations, neg_tails)

        pos_loss = F.logsigmoid(self.margin - pos_score).mean()
        neg_loss = F.logsigmoid(neg_score - self.margin).mean()

        return -pos_loss - neg_loss

    def get_entity_embedding(self, entity_id: int) -> torch.Tensor:
        """获取实体嵌入（拼接实部和虚部）"""
        idx = torch.tensor([entity_id], device=self.entity_re.weight.device)
        re = self.entity_re(idx).squeeze(0)
        im = self.entity_im(idx).squeeze(0)
        return torch.cat([re, im], dim=0)


# =============================================================================
# Lorentz/Hyperbolic Space KG Embedding Models
# =============================================================================
#
# 参考论文:
# - RotH/RefH/AttH: Low-Dimensional Hyperbolic KG Embeddings (ACL 2020)
#   https://github.com/HazyResearch/KGEmb
# - LorentzKG: Enhancing Hyperbolic KG Embeddings via Lorentz Transformations (ACL 2024)
#
# 为什么使用Lorentz模型而非Poincaré?
# - Poincaré球模型: 边界数值不稳定，梯度计算复杂
# - Lorentz/Hyperboloid模型: 数值稳定，梯度简洁，适合实际训练
#
# 依赖库: geoopt (pip install geoopt)
# =============================================================================

try:
    import geoopt
    from geoopt.manifolds import Lorentz as LorentzManifold
    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False
    print("Warning: geoopt not installed. Lorentz models unavailable.")
    print("Install with: pip install geoopt")


class LorentzKGBase(nn.Module):
    """
    Lorentz空间KG嵌入基类

    Lorentz/Hyperboloid模型定义在 n+1 维空间中的双曲面:
    H^n = {x ∈ R^{n+1} : <x,x>_L = -1/K, x_0 > 0}

    其中 K > 0 是曲率，Lorentz内积:
    <u,v>_L = -u_0*v_0 + sum(u_i*v_i)

    注意: geoopt的曲率定义是 geoopt.K = -1/K
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 32,
        curvature: float = 1.0,
        margin: float = 2.0
    ):
        super().__init__()
        if not GEOOPT_AVAILABLE:
            raise ImportError("geoopt is required for Lorentz models. Install with: pip install geoopt")

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim  # 空间维度 (实际嵌入是 dim+1)
        self.curvature = curvature
        self.margin = margin

        # geoopt使用 k = -1/K 的约定
        self.manifold = LorentzManifold(k=-1.0/curvature)

        # 实体嵌入 (在Lorentz流形上)
        self.entity_embeddings = self._init_lorentz_embeddings(num_entities)

    def _init_lorentz_embeddings(self, num_points: int) -> nn.Parameter:
        """初始化Lorentz流形上的点"""
        # 初始化在切空间，然后投影到流形
        # x_0 = sqrt(1/K + ||x_{1:}||^2)
        tangent = torch.randn(num_points, self.dim) * 0.01
        x0 = torch.sqrt(1.0/self.curvature + (tangent ** 2).sum(dim=-1, keepdim=True))
        init_points = torch.cat([x0, tangent], dim=-1)

        return geoopt.ManifoldParameter(
            init_points,
            manifold=self.manifold
        )

    def lorentz_inner(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Lorentz内积: <u,v>_L = -u_0*v_0 + sum(u_i*v_i)

        Args:
            u, v: (..., dim+1) tensors on Lorentz manifold

        Returns:
            (...,) inner product values
        """
        return -u[..., 0] * v[..., 0] + (u[..., 1:] * v[..., 1:]).sum(dim=-1)

    def lorentz_distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Lorentz距离: d_L(u,v) = sqrt(1/K) * arcosh(-K * <u,v>_L)

        对于 K=1: d_L(u,v) = arcosh(-<u,v>_L)

        Args:
            u, v: (..., dim+1) tensors on Lorentz manifold

        Returns:
            (...,) distance values
        """
        inner = self.lorentz_inner(u, v)
        # clamp to avoid numerical issues with arcosh
        inner_clamped = torch.clamp(-inner * self.curvature, min=1.0 + 1e-6)
        return (1.0 / math.sqrt(self.curvature)) * torch.acosh(inner_clamped)

    def project_to_lorentz(self, x: torch.Tensor) -> torch.Tensor:
        """
        将点投影到Lorentz流形

        给定空间分量 x_{1:}, 计算 x_0 使得点在流形上
        """
        spatial = x[..., 1:]
        x0 = torch.sqrt(1.0/self.curvature + (spatial ** 2).sum(dim=-1, keepdim=True))
        return torch.cat([x0, spatial], dim=-1)

    def get_entity_embedding(self, entity_id: int) -> torch.Tensor:
        """获取实体嵌入"""
        idx = torch.tensor([entity_id], device=self.entity_embeddings.device)
        return self.entity_embeddings[idx].squeeze(0)


class RotH(LorentzKGBase):
    """
    RotH: Rotation in Hyperbolic Space (ACL 2020)

    核心思想: 使用Givens旋转在双曲空间中变换实体
    score(h, r, t) = -d_L(Rot_r(h), t)^2 + b_h + b_t

    Givens旋转保持Lorentz内积不变，是双曲空间的等距变换
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 32,
        curvature: float = 1.0,
        margin: float = 2.0
    ):
        super().__init__(num_entities, num_relations, dim, curvature, margin)

        # 关系参数: Givens旋转角度
        # 每对维度一个角度，共 dim*(dim-1)/2 个角度（但我们简化为 dim 个）
        self.rel_diag = nn.Embedding(num_relations, dim)

        # 偏置项
        self.entity_bias = nn.Embedding(num_entities, 1)
        self.rel_bias = nn.Embedding(num_relations, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.rel_diag.weight)
        nn.init.zeros_(self.entity_bias.weight)
        nn.init.zeros_(self.rel_bias.weight)

    def givens_rotation(self, v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        应用Givens旋转（简化版：对空间分量应用对角旋转）

        实际的Givens旋转应该是成对维度的旋转矩阵，
        这里使用对角近似：每个维度独立旋转

        Args:
            v: (..., dim+1) Lorentz向量
            theta: (..., dim) 旋转角度

        Returns:
            旋转后的向量
        """
        # 分离时间分量和空间分量
        x0 = v[..., 0:1]  # 时间分量保持不变
        spatial = v[..., 1:]  # 空间分量

        # 对空间分量应用旋转 (简化为缩放+旋转近似)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # 成对旋转: 将维度两两配对
        rotated = spatial.clone()
        half_dim = self.dim // 2

        for i in range(half_dim):
            j = i + half_dim
            if j < self.dim:
                c, s = cos_theta[..., i:i+1], sin_theta[..., i:i+1]
                x_i, x_j = spatial[..., i:i+1], spatial[..., j:j+1]
                rotated[..., i:i+1] = c * x_i - s * x_j
                rotated[..., j:j+1] = s * x_i + c * x_j

        return torch.cat([x0, rotated], dim=-1)

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        计算得分: -d_L(Rot_r(h), t)^2 + b_h + b_t
        """
        h = self.entity_embeddings[heads]
        t = self.entity_embeddings[tails]
        rot = self.rel_diag(relations)

        # 应用旋转
        h_rot = self.givens_rotation(h, rot)

        # 计算距离
        dist = self.lorentz_distance(h_rot, t)

        # 添加偏置
        b_h = self.entity_bias(heads).squeeze(-1)
        b_t = self.entity_bias(tails).squeeze(-1)

        return -(dist ** 2) + b_h + b_t

    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_tails: torch.Tensor
    ) -> torch.Tensor:
        """Binary cross-entropy loss"""
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, pos_relations, neg_tails)

        pos_loss = F.softplus(-pos_score).mean()
        neg_loss = F.softplus(neg_score).mean()

        return pos_loss + neg_loss


class RefH(LorentzKGBase):
    """
    RefH: Reflection in Hyperbolic Space (ACL 2020)

    核心思想: 使用反射变换在双曲空间中变换实体
    score(h, r, t) = -d_L(Ref_r(h), t)^2 + b_h + b_t

    反射变换: Ref_r(x) = x - 2<x,r>_L * r (对于单位向量 r)
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 32,
        curvature: float = 1.0,
        margin: float = 2.0
    ):
        super().__init__(num_entities, num_relations, dim, curvature, margin)

        # 关系参数: 反射超平面的法向量（在切空间中）
        self.rel_reflect = nn.Embedding(num_relations, dim + 1)

        # 偏置项
        self.entity_bias = nn.Embedding(num_entities, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.rel_reflect.weight)
        nn.init.zeros_(self.entity_bias.weight)

    def reflection(self, v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Lorentz反射: Ref_r(x) = x - 2 * (<x,r>_L / <r,r>_L) * r

        Args:
            v: (..., dim+1) 要反射的向量
            r: (..., dim+1) 反射法向量

        Returns:
            反射后的向量
        """
        # 归一化反射向量
        r_norm_sq = self.lorentz_inner(r, r)
        # 避免除零
        r_norm_sq = torch.clamp(r_norm_sq.abs(), min=1e-6)

        # 计算反射
        inner_xr = self.lorentz_inner(v, r)
        coeff = 2 * inner_xr / r_norm_sq

        reflected = v - coeff.unsqueeze(-1) * r

        # 投影回流形
        return self.project_to_lorentz(reflected)

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """计算得分"""
        h = self.entity_embeddings[heads]
        t = self.entity_embeddings[tails]
        r = self.rel_reflect(relations)

        # 应用反射
        h_ref = self.reflection(h, r)

        # 计算距离
        dist = self.lorentz_distance(h_ref, t)

        # 偏置
        b_h = self.entity_bias(heads).squeeze(-1)
        b_t = self.entity_bias(tails).squeeze(-1)

        return -(dist ** 2) + b_h + b_t

    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_tails: torch.Tensor
    ) -> torch.Tensor:
        """Binary cross-entropy loss"""
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, pos_relations, neg_tails)

        pos_loss = F.softplus(-pos_score).mean()
        neg_loss = F.softplus(neg_score).mean()

        return pos_loss + neg_loss


class AttH(LorentzKGBase):
    """
    AttH: Attention-based Hyperbolic KG Embedding (ACL 2020)

    核心思想: 结合旋转和反射，使用注意力机制加权
    score(h, r, t) = -d_L(Att_r(h), t)^2 + b_h + b_t

    Att_r(h) = α * Rot_r(h) + (1-α) * Ref_r(h)
    其中 α 是学习的注意力权重
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 32,
        curvature: float = 1.0,
        margin: float = 2.0
    ):
        super().__init__(num_entities, num_relations, dim, curvature, margin)

        # 旋转参数
        self.rel_diag = nn.Embedding(num_relations, dim)

        # 反射参数
        self.rel_reflect = nn.Embedding(num_relations, dim + 1)

        # 注意力权重
        self.attention = nn.Embedding(num_relations, 1)

        # 偏置
        self.entity_bias = nn.Embedding(num_entities, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.rel_diag.weight)
        nn.init.xavier_uniform_(self.rel_reflect.weight)
        nn.init.zeros_(self.attention.weight)  # sigmoid(0) = 0.5
        nn.init.zeros_(self.entity_bias.weight)

    def givens_rotation(self, v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """同RotH"""
        x0 = v[..., 0:1]
        spatial = v[..., 1:]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        rotated = spatial.clone()
        half_dim = self.dim // 2

        for i in range(half_dim):
            j = i + half_dim
            if j < self.dim:
                c, s = cos_theta[..., i:i+1], sin_theta[..., i:i+1]
                x_i, x_j = spatial[..., i:i+1], spatial[..., j:j+1]
                rotated[..., i:i+1] = c * x_i - s * x_j
                rotated[..., j:j+1] = s * x_i + c * x_j

        return torch.cat([x0, rotated], dim=-1)

    def reflection(self, v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """同RefH"""
        r_norm_sq = self.lorentz_inner(r, r)
        r_norm_sq = torch.clamp(r_norm_sq.abs(), min=1e-6)

        inner_xr = self.lorentz_inner(v, r)
        coeff = 2 * inner_xr / r_norm_sq

        reflected = v - coeff.unsqueeze(-1) * r
        return self.project_to_lorentz(reflected)

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """计算得分"""
        h = self.entity_embeddings[heads]
        t = self.entity_embeddings[tails]

        rot_angles = self.rel_diag(relations)
        ref_vectors = self.rel_reflect(relations)
        alpha = torch.sigmoid(self.attention(relations))  # (batch, 1)

        # 旋转和反射
        h_rot = self.givens_rotation(h, rot_angles)
        h_ref = self.reflection(h, ref_vectors)

        # 注意力加权组合 (在切空间中近似)
        # 实际应该使用geodesic插值，这里简化
        h_att = alpha * h_rot + (1 - alpha) * h_ref
        h_att = self.project_to_lorentz(h_att)

        # 距离
        dist = self.lorentz_distance(h_att, t)

        # 偏置
        b_h = self.entity_bias(heads).squeeze(-1)
        b_t = self.entity_bias(tails).squeeze(-1)

        return -(dist ** 2) + b_h + b_t

    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_tails: torch.Tensor
    ) -> torch.Tensor:
        """Binary cross-entropy loss"""
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, pos_relations, neg_tails)

        pos_loss = F.softplus(-pos_score).mean()
        neg_loss = F.softplus(neg_score).mean()

        return pos_loss + neg_loss


class LorentzKG(LorentzKGBase):
    """
    LorentzKG: Lorentz Transformation-based KG Embedding (ACL 2024 Findings)

    核心思想: 使用完整的Lorentz变换（boost + rotation）
    - Lorentz boost: 沿某方向的双曲"平移"
    - Lorentz rotation: 空间部分的旋转

    score(h, r, t) = -d_L(L_r(h), t)^2 + b_h + b_t
    其中 L_r = Boost_r ∘ Rot_r
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int = 32,
        curvature: float = 1.0,
        margin: float = 2.0
    ):
        super().__init__(num_entities, num_relations, dim, curvature, margin)

        # Lorentz boost参数 (rapidity)
        self.rel_boost = nn.Embedding(num_relations, dim)

        # 旋转参数
        self.rel_rotation = nn.Embedding(num_relations, dim)

        # 平移向量 (在切空间中)
        self.rel_translation = nn.Embedding(num_relations, dim)

        # 偏置
        self.entity_bias = nn.Embedding(num_entities, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.rel_boost.weight, std=0.01)
        nn.init.zeros_(self.rel_rotation.weight)
        nn.init.normal_(self.rel_translation.weight, std=0.01)
        nn.init.zeros_(self.entity_bias.weight)

    def lorentz_boost(self, v: torch.Tensor, rapidity: torch.Tensor) -> torch.Tensor:
        """
        Lorentz boost (沿第一个空间方向简化)

        对于rapidity φ:
        x'_0 = x_0 * cosh(φ) + x_1 * sinh(φ)
        x'_1 = x_0 * sinh(φ) + x_1 * cosh(φ)
        x'_i = x_i (i > 1)

        这里我们对每个维度独立应用小boost（近似）
        """
        x0 = v[..., 0:1]  # (batch, 1)
        spatial = v[..., 1:]  # (batch, dim)

        # 限制rapidity大小以保持数值稳定
        rapidity = torch.clamp(rapidity, -2.0, 2.0)

        cosh_phi = torch.cosh(rapidity)  # (batch, dim)
        sinh_phi = torch.sinh(rapidity)

        # 应用boost (简化: 只boost第一个分量)
        new_x0 = x0 * cosh_phi[..., 0:1] + spatial[..., 0:1] * sinh_phi[..., 0:1]
        new_x1 = x0 * sinh_phi[..., 0:1] + spatial[..., 0:1] * cosh_phi[..., 0:1]

        new_spatial = torch.cat([new_x1, spatial[..., 1:]], dim=-1)

        boosted = torch.cat([new_x0, new_spatial], dim=-1)
        return self.project_to_lorentz(boosted)

    def spatial_rotation(self, v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """空间部分的旋转（同RotH）"""
        x0 = v[..., 0:1]
        spatial = v[..., 1:]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        rotated = spatial.clone()
        half_dim = self.dim // 2

        for i in range(half_dim):
            j = i + half_dim
            if j < self.dim:
                c, s = cos_theta[..., i:i+1], sin_theta[..., i:i+1]
                x_i, x_j = spatial[..., i:i+1], spatial[..., j:j+1]
                rotated[..., i:i+1] = c * x_i - s * x_j
                rotated[..., j:j+1] = s * x_i + c * x_j

        return torch.cat([x0, rotated], dim=-1)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Lorentz流形上的指数映射 (切向量 -> 流形上的点)

        exp_x(v) = cosh(||v||_L) * x + sinh(||v||_L) * v / ||v||_L
        """
        v_norm = torch.sqrt(torch.clamp(self.lorentz_inner(v, v), min=1e-6))
        v_norm = v_norm.unsqueeze(-1)

        result = torch.cosh(v_norm) * x + torch.sinh(v_norm) * v / v_norm
        return self.project_to_lorentz(result)

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """计算得分"""
        h = self.entity_embeddings[heads]
        t = self.entity_embeddings[tails]

        boost = self.rel_boost(relations)
        rotation = self.rel_rotation(relations)
        translation = self.rel_translation(relations)

        # 1. 旋转
        h_rot = self.spatial_rotation(h, rotation)

        # 2. Boost
        h_boost = self.lorentz_boost(h_rot, boost)

        # 3. 平移 (通过指数映射)
        # 构造切向量
        tangent = torch.cat([torch.zeros_like(translation[..., 0:1]), translation], dim=-1)
        h_trans = self.exp_map(h_boost, tangent * 0.1)  # 缩放以保持稳定

        # 距离
        dist = self.lorentz_distance(h_trans, t)

        # 偏置
        b_h = self.entity_bias(heads).squeeze(-1)
        b_t = self.entity_bias(tails).squeeze(-1)

        return -(dist ** 2) + b_h + b_t

    def loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_tails: torch.Tensor
    ) -> torch.Tensor:
        """Self-adversarial negative sampling loss"""
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, pos_relations, neg_tails)

        pos_loss = F.logsigmoid(pos_score).mean()
        neg_loss = F.logsigmoid(-neg_score).mean()

        return -pos_loss - neg_loss


class CausalKnowledgeGraph:
    """
    因果知识图谱

    用于构建和管理事件因果关系的知识图谱
    """

    def __init__(self):
        self.entities: Dict[str, int] = {}  # entity_text -> entity_id
        self.relations: Dict[str, int] = {}  # relation_type -> relation_id
        self.triples: List[Tuple[int, int, int]] = []  # (head, relation, tail)

        # 反向映射
        self.id2entity: Dict[int, str] = {}
        self.id2relation: Dict[int, str] = {}

        # 预定义因果关系类型
        self.causal_relations = [
            "causes",           # 直接因果
            "enables",          # 使能
            "prevents",         # 阻止
            "leads_to",         # 导致
            "results_in",       # 结果
            "is_caused_by",     # 被...引起
            "happens_before",   # 时序在前
            "happens_after",    # 时序在后
        ]

        for rel in self.causal_relations:
            self._add_relation(rel)

    def _add_entity(self, entity_text: str) -> int:
        """添加实体"""
        if entity_text not in self.entities:
            entity_id = len(self.entities)
            self.entities[entity_text] = entity_id
            self.id2entity[entity_id] = entity_text
        return self.entities[entity_text]

    def _add_relation(self, relation_type: str) -> int:
        """添加关系"""
        if relation_type not in self.relations:
            relation_id = len(self.relations)
            self.relations[relation_type] = relation_id
            self.id2relation[relation_id] = relation_type
        return self.relations[relation_type]

    def add_triple(
        self,
        head: str,
        relation: str,
        tail: str
    ):
        """添加三元组"""
        h_id = self._add_entity(head)
        r_id = self._add_relation(relation)
        t_id = self._add_entity(tail)
        self.triples.append((h_id, r_id, t_id))

    def add_causal_relation(self, cause: str, effect: str):
        """添加因果关系"""
        self.add_triple(cause, "causes", effect)
        self.add_triple(effect, "is_caused_by", cause)

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_relations(self) -> int:
        return len(self.relations)

    def get_triples_tensor(self) -> torch.Tensor:
        """返回所有三元组的张量"""
        return torch.tensor(self.triples, dtype=torch.long)

    def save(self, path: str):
        """保存知识图谱"""
        data = {
            "entities": self.entities,
            "relations": self.relations,
            "triples": self.triples
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """加载知识图谱"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.entities = data["entities"]
        self.relations = data["relations"]
        self.triples = [tuple(t) for t in data["triples"]]

        self.id2entity = {v: k for k, v in self.entities.items()}
        self.id2relation = {v: k for k, v in self.relations.items()}


class KGDataset(Dataset):
    """KG训练数据集"""

    def __init__(
        self,
        triples: torch.Tensor,
        num_entities: int,
        negative_samples: int = 10
    ):
        self.triples = triples
        self.num_entities = num_entities
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.triples) * self.negative_samples

    def __getitem__(self, idx):
        triple_idx = idx // self.negative_samples
        h, r, t = self.triples[triple_idx]

        # 随机替换头或尾生成负样本
        if torch.rand(1).item() < 0.5:
            neg_h = torch.randint(0, self.num_entities, (1,)).item()
            neg_t = t.item()
        else:
            neg_h = h.item()
            neg_t = torch.randint(0, self.num_entities, (1,)).item()

        return {
            "pos_head": h,
            "pos_relation": r,
            "pos_tail": t,
            "neg_head": torch.tensor(neg_h),
            "neg_tail": torch.tensor(neg_t)
        }


class KGEmbeddingTrainer:
    """KG Embedding 训练器"""

    def __init__(
        self,
        kg: CausalKnowledgeGraph,
        model_type: str = "TransE",
        config: Optional[KGEConfig] = None
    ):
        self.kg = kg
        self.config = config or KGEConfig()
        self.model_type = model_type

        # 创建模型
        self.model = self._create_model(model_type)
        self.model.to(self.config.device)

    def _create_model(self, model_type: str) -> nn.Module:
        """创建模型"""
        # Euclidean models
        euclidean_models = {
            "TransE": TransE,
            "ComplEx": ComplEx,
            "RotatE": RotatE
        }

        # Hyperbolic/Lorentz models (require geoopt)
        lorentz_models = {}
        if GEOOPT_AVAILABLE:
            lorentz_models = {
                "RotH": RotH,
                "RefH": RefH,
                "AttH": AttH,
                "LorentzKG": LorentzKG
            }

        all_models = {**euclidean_models, **lorentz_models}

        if model_type not in all_models:
            available = list(all_models.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

        model_cls = all_models[model_type]

        # Lorentz models use 'dim' instead of 'embedding_dim'
        if model_type in lorentz_models:
            return model_cls(
                num_entities=self.kg.num_entities,
                num_relations=self.kg.num_relations,
                dim=self.config.embedding_dim
            )
        else:
            return model_cls(
                num_entities=self.kg.num_entities,
                num_relations=self.kg.num_relations,
                embedding_dim=self.config.embedding_dim
            )

    def train(self) -> Dict[str, float]:
        """训练模型"""
        triples = self.kg.get_triples_tensor()
        dataset = KGDataset(
            triples,
            self.kg.num_entities,
            self.config.negative_samples
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # 使用RiemannianAdam优化Lorentz模型的流形参数
        lorentz_models = ["RotH", "RefH", "AttH", "LorentzKG"]
        if self.model_type in lorentz_models and GEOOPT_AVAILABLE:
            from geoopt.optim import RiemannianAdam
            optimizer = RiemannianAdam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                stabilize=10  # 每10步稳定化流形参数
            )
            print(f"Using RiemannianAdam optimizer for {self.model_type}")
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )

        print(f"Training {self.model_type} on {len(triples)} triples...")

        losses = []
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            batch_count = 0
            nan_detected = False
            for batch in dataloader:
                optimizer.zero_grad()

                loss = self.model.loss(
                    batch["pos_head"].to(self.config.device),
                    batch["pos_relation"].to(self.config.device),
                    batch["pos_tail"].to(self.config.device),
                    batch["neg_head"].to(self.config.device),
                    batch["neg_tail"].to(self.config.device)
                )

                # 数值稳定性保护：一旦出现 NaN / Inf，提前停止训练，保留最后一个正常 loss
                if not torch.isfinite(loss):
                    print(f"[Warning] Non-finite loss detected at epoch {epoch+1}. Stopping training early.")
                    nan_detected = True
                    break

                loss.backward()
                optimizer.step()

                if hasattr(self.model, '_normalize_entities'):
                    self.model._normalize_entities()

                epoch_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                losses.append(avg_loss)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}")

            if nan_detected:
                break

        final_loss = losses[-1] if losses else float('nan')
        return {"final_loss": final_loss, "losses": losses}

    def get_entity_embeddings(self) -> Dict[str, np.ndarray]:
        """获取所有实体嵌入"""
        embeddings = {}
        self.model.eval()

        with torch.no_grad():
            for entity_text, entity_id in self.kg.entities.items():
                emb = self.model.get_entity_embedding(entity_id)
                embeddings[entity_text] = emb.cpu().numpy()

        return embeddings

    def save_embeddings(self, path: str):
        """保存嵌入"""
        embeddings = self.get_entity_embeddings()
        np.savez(path, **embeddings)
        print(f"Embeddings saved to {path}")

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "config": self.config,
            "kg_entities": self.kg.entities,
            "kg_relations": self.kg.relations
        }, path)
        print(f"Model saved to {path}")


def get_kg_embedding_model(model_type: str = "TransE") -> type:
    """获取KG embedding模型类"""
    euclidean_models = {
        "TransE": TransE,
        "ComplEx": ComplEx,
        "RotatE": RotatE
    }

    lorentz_models = {}
    if GEOOPT_AVAILABLE:
        lorentz_models = {
            "RotH": RotH,
            "RefH": RefH,
            "AttH": AttH,
            "LorentzKG": LorentzKG
        }

    all_models = {**euclidean_models, **lorentz_models}
    return all_models.get(model_type, TransE)


def get_available_models() -> Dict[str, List[str]]:
    """获取可用的模型列表"""
    euclidean = ["TransE", "ComplEx", "RotatE"]
    lorentz = []
    if GEOOPT_AVAILABLE:
        lorentz = ["RotH", "RefH", "AttH", "LorentzKG"]
    return {
        "euclidean": euclidean,
        "lorentz": lorentz,
        "all": euclidean + lorentz
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test KG Embedding Module")
    parser.add_argument("--model", type=str, default="TransE",
                        help="Model type (TransE, ComplEx, RotatE, RotH, RefH, AttH, LorentzKG)")
    parser.add_argument("--dim", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing KG Embedding Module")
    print("=" * 60)

    # 显示可用模型
    available = get_available_models()
    print(f"\nAvailable Euclidean models: {available['euclidean']}")
    print(f"Available Lorentz models: {available['lorentz']}")
    if not available['lorentz']:
        print("  (Install geoopt for Lorentz models: pip install geoopt)")

    # 创建测试知识图谱
    print("\n--- Creating test knowledge graph ---")
    kg = CausalKnowledgeGraph()
    kg.add_causal_relation("Economic recession", "Unemployment rises")
    kg.add_causal_relation("Interest rate cut", "Stock market rises")
    kg.add_causal_relation("Trade war", "Economic uncertainty")
    kg.add_causal_relation("Government stimulus", "Economic recovery")
    kg.add_causal_relation("US drone strike", "Iran missile attack")
    kg.add_causal_relation("Military tensions", "Diplomatic crisis")

    print(f"KG: {kg.num_entities} entities, {kg.num_relations} relations")
    print(f"Triples: {len(kg.triples)}")

    # 训练模型
    print(f"\n--- Training {args.model} ---")
    try:
        trainer = KGEmbeddingTrainer(
            kg,
            model_type=args.model,
            config=KGEConfig(embedding_dim=args.dim, num_epochs=args.epochs)
        )
        results = trainer.train()
        print(f"Training complete. Final loss: {results['final_loss']:.4f}")

        # 获取嵌入
        embeddings = trainer.get_entity_embeddings()
        print(f"\nGenerated embeddings for {len(embeddings)} entities")
        for entity, emb in list(embeddings.items())[:3]:
            print(f"  {entity}: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")

    except ImportError as e:
        print(f"Error: {e}")
        print("For Lorentz models, install geoopt: pip install geoopt")
    except Exception as e:
        print(f"Error: {e}")

    # 如果geoopt可用，测试Lorentz模型
    if GEOOPT_AVAILABLE and args.model == "TransE":
        print("\n--- Testing RotH (Lorentz) ---")
        trainer_lorentz = KGEmbeddingTrainer(
            kg,
            model_type="RotH",
            config=KGEConfig(embedding_dim=32, num_epochs=30)
        )
        results_lorentz = trainer_lorentz.train()
        print(f"RotH training complete. Final loss: {results_lorentz['final_loss']:.4f}")
