import torch
import torch.nn as nn
import triton
import triton.language as tl
import weakref
import RcMoE_prefetcher
from transformers import PreTrainedModel, LlamaConfig, LlamaForCausalLM

# ===================================================================
# 1. 极致优化的 Triton 融合核
# ===================================================================


# ===================================================================
# 1. 极致优化的 Triton 融合核 (寄存器级重构)
# ===================================================================


@triton.jit
def fused_rcmoe_tail_kernel(
    out_p, in_p, w_p, nf4_p, m_p, s_p, cb_p, mlp_p,
    in_sn, in_sh, w_sh, w_se, nf4_sn, nf4_se, nf4_sh,
    m_sn, m_se, m_sb, s_sn, s_se, s_sb, mlp_sn, mlp_sh, out_sn, out_sh,
    N, E, H, BLK_H: tl.constexpr, BLK_E: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    # --- [Router 计算] ---
    logits = tl.zeros([BLK_E], dtype=tl.float32)
    e_off = tl.arange(0, BLK_E)
    e_mask = e_off < E

    for h_idx in range(0, H, BLK_H):
        # 拆分为独立行，严禁使用反斜杠和复杂元组解包
        h_off = h_idx + tl.arange(0, BLK_H)
        h_mask = h_off < H

        # 使用括号显式包裹长代码以支持换行（Triton 支持括号换行，不支持反斜杠）
        x_ptrs = in_p + pid * in_sn + h_off * in_sh
        x = tl.load(x_ptrs, mask=h_mask, other=0.0).to(tl.float32)

        w_ptrs = w_p + h_off[:, None] * w_sh + e_off[None, :] * w_se
        w_mask = h_mask[:, None] & e_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        logits += tl.sum(x[:, None] * w, axis=0)

    logits = tl.where(e_mask, logits, -float('inf'))
    probs = tl.exp(logits - tl.max(logits, axis=0))
    probs = (probs / tl.sum(probs, axis=0)).to(tl.float16)

    # --- [反量化与融合计算] ---
    for h_idx in range(0, H, BLK_H):
        h_off = h_idx + tl.arange(0, BLK_H)
        h_mask = h_off < H
        acc = tl.zeros([BLK_H], dtype=tl.float32)

        # 预计算分块索引 (512 元素为一个解压块)
        b_idx = h_off // 512

        for e in range(E):
            prob_e = tl.sum(tl.where(e_off == e, probs, 0.0), axis=0)

            # 128-bit 向量化加载压缩数据
            pack_ptr = nf4_p + pid * nf4_sn + \
                e * nf4_se + (h_off // 2) * nf4_sh
            pack_b = tl.load(pack_ptr, mask=h_mask, other=0).to(tl.int32)

            # 位运算原地解压
            is_high = (h_off % 2) == 1
            idx = tl.where(is_high, (pack_b >> 4) & 0x0F, pack_b & 0x0F)

            # 查表
            val_dq = tl.load(cb_p + idx)

            # 128-bit 加载 Mean 和 Std (分拆多行以保证安全)
            m_ptr = m_p + pid * m_sn + e * m_se + b_idx * m_sb
            m_val = tl.load(m_ptr, mask=h_mask, other=0.0).to(tl.float32)

            s_ptr = s_p + pid * s_sn + e * s_se + b_idx * s_sb
            s_val = tl.load(s_ptr, mask=h_mask, other=1.0).to(tl.float32)

            # 所有的乘加运算均在寄存器中进行
            acc += (val_dq * s_val + m_val) * prob_e

        # 加载共享专家输出并写回
        mlp_ptr = mlp_p + pid * mlp_sn + h_off * mlp_sh
        mlp_val = tl.load(mlp_ptr, mask=h_mask, other=0.0).to(tl.float32)

        out_ptr = out_p + pid * out_sn + h_off * out_sh
        tl.store(out_ptr, (acc + mlp_val).to(out_p.dtype.element_ty), mask=h_mask)

def run_fused_rcmoe_tail_kernel(rout_in, router_w, exp_nf4, mean, std, codebook, mlp):
    B, L, H = rout_in.shape
    N, E = B * L, router_w.shape[1]
    r_in, mlp_f = rout_in.reshape(N, H), mlp.reshape(N, H)
    out = torch.empty_like(mlp_f)

    # 为了触发 128-bit 向量化加载，BLK_H 建议至少为 32
    BLK_H = 128 if H >= 128 else triton.next_power_of_2(H)
    BLK_E = triton.next_power_of_2(E)

    fused_rcmoe_tail_kernel[(N,)](
        out, r_in, router_w, exp_nf4, mean, std, codebook, mlp_f,
        *r_in.stride(), *router_w.stride(), *exp_nf4.stride(), *mean.stride(),
        *std.stride(), *mlp_f.stride(), *out.stride(),
        N, E, H, BLK_H, BLK_E
    )
    return out.view(B, L, H)


class RcMoEConfig(LlamaConfig):
    model_type = "RcMoE"


class RcMoE_Rep_Layer(nn.Module):
    def __init__(self, base_mlp, num_experts, hidden_size, layer_idx, parent_model):
        super().__init__()
        self.layer_idx = layer_idx
        self.parent_ref = weakref.ref(parent_model)
        self.shared_expert = base_mlp
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.FloatTensor):
        p, cur_b = self.parent_ref(), self.layer_idx % 2
        shared_out = self.shared_expert(x)
        main_stream = torch.cuda.current_stream()

        # 1. 后台预取下一层 (C++ 处理，释放 GIL)
        if (nxt := self.layer_idx + 1) < p.num_layers:
            nxt_b = nxt % 2
            # GPU 级同步：预取流必须等主计算流用完 next buffer 后再覆盖
            p.prefetch_stream.wait_event(p.evt_compute_done[nxt_b])
            with torch.cuda.stream(p.prefetch_stream):
                b = p.bufs[nxt_b]
                RcMoE_prefetcher.launch_prefetch(p.model.current_token_ids, nxt, p.model.lut_idx,
                                                p.model.lut_mean, p.model.lut_std, b['nf4'], b['mean'], b['std'], nxt_b)
                p.evt_prefetch_done[nxt_b].record()

        # 2. 主流等待当前层数据就绪 (完全是 GPU 级同步，不阻塞 Python)
        main_stream.wait_event(p.evt_prefetch_done[cur_b])

        # 3. 融合计算
        b = p.bufs[cur_b]
        out = run_fused_rcmoe_tail_kernel(x, self.router.weight.t().contiguous(
        ), b['nf4'], b['mean'], b['std'], p.model.nf4_cb, shared_out)

        # 4. 通知缓冲区已计算完毕，可供覆写
        p.evt_compute_done[cur_b].record(main_stream)

        return out


class RcMoEForCausalLM(PreTrainedModel):
    config_class, _no_split_modules = RcMoEConfig, ["LlamaDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaForCausalLM(config)
        self.num_layers = len(self.model.model.layers)

        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'mlp'):
                layer.mlp = RcMoE_Rep_Layer(
                    layer.mlp, config.num_experts, config.hidden_size, i, self)

        def init_pipeline_hook(module, args, output):
            self.model.current_token_ids = args[0].cpu().view(-1)
            B, L = args[0].shape

            # Lazy 懒加载初始化 Buffer 和 Events
            if getattr(self, 'bufs', None) is None or self.bufs[0]['nf4'].shape[0] != B * L:
                def mk_b(): return {'nf4': torch.empty((B*L, config.num_experts, config.hidden_size//2), dtype=torch.uint8, device=output.device),
                                    'mean': torch.empty((B*L, config.num_experts, config.hidden_size//512), dtype=torch.float16, device=output.device),
                                    'std': torch.empty((B*L, config.num_experts, config.hidden_size//512), dtype=torch.float16, device=output.device)}
                self.bufs = [mk_b(), mk_b()]
                self.prefetch_stream = torch.cuda.Stream()

                # evt_compute_done 初始状态为 "已完成"，允许第一轮安全写入
                self.evt_compute_done = [torch.cuda.Event(
                    enable_timing=False) for _ in range(2)]
                self.evt_prefetch_done = [torch.cuda.Event(
                    enable_timing=False) for _ in range(2)]
                for evt in self.evt_compute_done:
                    evt.record()

            # 启动 Layer 0 的首次预取
            self.prefetch_stream.wait_event(self.evt_compute_done[0])
            with torch.cuda.stream(self.prefetch_stream):
                b = self.bufs[0]
                RcMoE_prefetcher.launch_prefetch(self.model.current_token_ids, 0, self.model.lut_idx,
                                                self.model.lut_mean, self.model.lut_std, b['nf4'], b['mean'], b['std'], 0)
                self.evt_prefetch_done[0].record()

            return output

        self.model.get_input_embeddings().register_forward_hook(init_pipeline_hook)
        self.post_init()

    def load_lut(self, lut_path):
        d = torch.load(lut_path, mmap=True,
                       map_location='cpu', weights_only=False)
        v, l, e, _ = d['original_shape']
        self.model.lut_idx, self.model.lut_mean, self.model.lut_std = (
            d[k].view(v, l, e, -1) for k in ('idx', 'mean', 'std'))
        self.model.nf4_cb = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367,
                                          -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
                                          -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                                          0.24611230194568634, 0.33791524171829224, 0.44070982933044434,
                                          0.5626170039176941, 0.7229568362236023, 1.0], dtype=torch.float32, device="cuda")

    def forward(self, *args, **kwargs): return self.model(*args, **kwargs)

    def generate(self, *args, **
                 kwargs): return self.model.generate(*args, **kwargs)
