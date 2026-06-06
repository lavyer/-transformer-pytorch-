"""
绘制 Transformer 神经网络组织结构图 —— 用于课程设计报告。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def draw_transformer_architecture(output_path='figures/transformer_architecture.png'):
    """绘制 Transformer 模型结构图（类似论文中的经典架构图）"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 22))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 22)
    ax.axis('off')
    ax.set_facecolor('white')

    # ========== 颜色方案 ==========
    C_EMBED = '#E8D5B7'       # 嵌入层 - 浅棕
    C_ATTN = '#A8D8EA'        # 注意力 - 浅蓝
    C_FFN = '#C3E6CB'         # 前馈网络 - 浅绿
    C_NORM = '#F7DC6F'        # 归一化 - 浅黄
    C_PROJ = '#F5B7B1'        # 投影层 - 浅红
    C_INPUT = '#D5D8DC'       # 输入输出 - 灰色
    C_ENC = '#EBF5FB'         # Encoder 外框
    C_DEC = '#FDEDEC'         # Decoder 外框

    def draw_box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
        """绘制圆角矩形框"""
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.1", facecolor=color,
                             edgecolor='#555555', linewidth=1.2)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight=weight, color='#333333')

    def draw_arrow(ax, x1, y1, x2, y2, color='#666666', lw=1.2):
        """绘制箭头"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))

    def draw_plus_circle(ax, x, y):
        """绘制残差连接的 ⊕ 符号"""
        circle = plt.Circle((x, y), 0.18, facecolor='white',
                           edgecolor='#555555', linewidth=1.2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, '+', ha='center', va='center', fontsize=11,
                fontweight='bold', color='#333333', zorder=6)

    # ========================
    # Encoder 区域（左侧）
    # ========================
    ENC_X = 5.5
    ENC_TOP = 20.5
    ENC_BOTTOM = 0.8

    # Encoder 外框
    enc_frame = FancyBboxPatch((0.5, ENC_BOTTOM), 10, ENC_TOP - ENC_BOTTOM,
                               boxstyle="round,pad=0.3", facecolor=C_ENC,
                               edgecolor='#3498DB', linewidth=2.5, alpha=0.3)
    ax.add_patch(enc_frame)
    ax.text(ENC_X, ENC_TOP - 0.3, 'Encoder (编码器)', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#2980B9')

    # --- 输入嵌入 ---
    EMBED_Y = ENC_TOP - 1.5
    draw_box(ax, ENC_X, EMBED_Y, 6, 1.0, 'Input Embedding\n输入嵌入层', C_EMBED, fontsize=10, bold=True)

    # --- 位置编码 ---
    POS_Y = EMBED_Y - 1.3
    draw_box(ax, ENC_X, POS_Y, 6, 0.9, 'Positional Encoding\n正弦位置编码', C_EMBED, fontsize=10)

    # 输入箭头
    draw_box(ax, ENC_X, ENC_TOP, 6, 0.9, '输入: 中文 Token IDs\n(seq_len=50)', C_INPUT, fontsize=10)
    draw_arrow(ax, ENC_X, ENC_TOP - 0.5, ENC_X, EMBED_Y + 0.55)
    draw_arrow(ax, ENC_X, EMBED_Y - 0.55, ENC_X, POS_Y + 0.5)

    # --- Encoder Block × N ---
    BLOCK_START_Y = POS_Y - 1.2

    for i in range(3):  # 画3层 EncoderBlock
        base_y = BLOCK_START_Y - i * 4.8
        label = f'EncoderBlock × {i+1}'

        # 残差1开始位置
        res1_y = base_y - 0.6

        # Multi-Head Self-Attention
        attn_y = res1_y - 0.9
        draw_box(ax, ENC_X, attn_y, 5.8, 1.2,
                 f'Multi-Head\nSelf-Attention (h=8)\n多头自注意力', C_ATTN, fontsize=9, bold=True)

        # 残差连接 ⊕
        draw_plus_circle(ax, ENC_X, res1_y)
        draw_arrow(ax, ENC_X, base_y, ENC_X, res1_y + 0.2)
        draw_arrow(ax, ENC_X, res1_y - 0.2, ENC_X, attn_y + 0.65)
        # 跳跃连接
        ax.plot([ENC_X + 3.3, ENC_X + 3.3], [base_y, res1_y], color='#E74C3C', lw=1.5,
                linestyle='--', zorder=3)

        # LayerNorm
        norm1_y = attn_y - 1.0
        draw_box(ax, ENC_X, norm1_y, 2.5, 0.6, 'Add & LayerNorm', C_NORM, fontsize=8)
        draw_arrow(ax, ENC_X, attn_y - 0.65, ENC_X, norm1_y + 0.35)
        ax.plot([ENC_X + 3.3, ENC_X + 3.3], [res1_y, norm1_y], color='#E74C3C', lw=1.5,
                linestyle='--', zorder=3)

        # Feed-Forward
        ffn_y = norm1_y - 1.0
        draw_box(ax, ENC_X, ffn_y, 5.8, 1.2,
                 f'Feed-Forward Network\nLinear→ReLU→Dropout→Linear\nd_ff={1024}', C_FFN, fontsize=9)

        # 残差连接 ⊕
        draw_plus_circle(ax, ENC_X, norm1_y - 0.25)
        draw_arrow(ax, ENC_X, norm1_y - 0.15, ENC_X, ffn_y + 0.65)

        # LayerNorm
        norm2_y = ffn_y - 1.0
        draw_box(ax, ENC_X, norm2_y, 2.5, 0.6, 'Add & LayerNorm', C_NORM, fontsize=8)
        draw_arrow(ax, ENC_X, ffn_y - 0.65, ENC_X, norm2_y + 0.35)

        # 记录最低点用于层间连接
        if i == 2:
            last_norm2_y = norm2_y

        # 层间箭头
        if i < 2:
            draw_arrow(ax, ENC_X, norm2_y - 0.35, ENC_X, base_y - 4.8 + 0.5)

    # Encoder 最终 LayerNorm
    final_norm_y = last_norm2_y - 1.2
    draw_box(ax, ENC_X, final_norm_y, 3, 0.7, 'LayerNorm (最终)', C_NORM, fontsize=9, bold=True)
    draw_arrow(ax, ENC_X, last_norm2_y - 0.35, ENC_X, final_norm_y + 0.4)

    # Encoder Output
    enc_out_y = final_norm_y - 0.9
    draw_box(ax, ENC_X, enc_out_y, 6, 0.9, 'Encoder Output\n编码器输出 (B, seq, d_model)', '#BDC3C7', fontsize=10, bold=True)
    draw_arrow(ax, ENC_X, final_norm_y - 0.4, ENC_X, enc_out_y + 0.5)

    # ========================
    # 交叉连接箭头
    # ========================
    cross_x1 = ENC_X + 3.5
    cross_y = enc_out_y - 2.5
    ax.annotate('', xy=(12.5, cross_y), xytext=(cross_x1, cross_y),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5,
                               connectionstyle='arc3,rad=0'))
    ax.text(9, cross_y + 0.5, 'Cross-Attention\n(K, V from Encoder)',
            ha='center', fontsize=9, color='#E74C3C', fontweight='bold')

    # ========================
    # Decoder 区域（右侧）
    # ========================
    DEC_X = 12.5
    DEC_TOP = ENC_TOP
    DEC_BOTTOM = enc_out_y - 1.5

    # Decoder 外框
    dec_frame = FancyBboxPatch((9, DEC_BOTTOM), 8.5, DEC_TOP - DEC_BOTTOM,
                               boxstyle="round,pad=0.3", facecolor=C_DEC,
                               edgecolor='#E74C3C', linewidth=2.5, alpha=0.3)
    ax.add_patch(dec_frame)
    ax.text(DEC_X, DEC_TOP - 0.3, 'Decoder (解码器)', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#C0392B')

    # --- 输出嵌入 ---
    DEMBED_Y = DEC_TOP - 1.5
    draw_box(ax, DEC_X, DEMBED_Y, 6, 1.0, 'Output Embedding\n目标嵌入层', C_EMBED, fontsize=10, bold=True)

    # --- 位置编码 ---
    DPOS_Y = DEMBED_Y - 1.3
    draw_box(ax, DEC_X, DPOS_Y, 6, 0.9, 'Positional Encoding\n正弦位置编码', C_EMBED, fontsize=10)

    # 输入
    draw_box(ax, DEC_X, DEC_TOP, 6, 0.9, '输入: 英文 Token IDs\n(右移一位)', C_INPUT, fontsize=10)
    draw_arrow(ax, DEC_X, DEC_TOP - 0.5, DEC_X, DEMBED_Y + 0.55)
    draw_arrow(ax, DEC_X, DEMBED_Y - 0.55, DEC_X, DPOS_Y + 0.5)

    # --- Decoder Block × N ---
    DBLOCK_START_Y = DPOS_Y - 1.2

    for i in range(3):
        base_y = DBLOCK_START_Y - i * 5.6

        # Masked Multi-Head Self-Attention
        d_res1_y = base_y - 0.6
        d_attn_y = d_res1_y - 0.9
        draw_box(ax, DEC_X, d_attn_y, 5.8, 1.2,
                 'Masked Multi-Head\nSelf-Attention (h=8)\n掩码多头自注意力', C_ATTN, fontsize=9, bold=True)

        draw_plus_circle(ax, DEC_X, d_res1_y)
        draw_arrow(ax, DEC_X, base_y, DEC_X, d_res1_y + 0.2)
        draw_arrow(ax, DEC_X, d_res1_y - 0.2, DEC_X, d_attn_y + 0.65)
        ax.plot([DEC_X + 3.3, DEC_X + 3.3], [base_y, d_res1_y], color='#E74C3C', lw=1.5,
                linestyle='--', zorder=3)

        d_norm1_y = d_attn_y - 1.0
        draw_box(ax, DEC_X, d_norm1_y, 2.5, 0.6, 'Add & LayerNorm', C_NORM, fontsize=8)
        draw_arrow(ax, DEC_X, d_attn_y - 0.65, DEC_X, d_norm1_y + 0.35)

        # Cross-Attention
        d_res2_y = d_norm1_y - 0.3
        d_cross_y = d_res2_y - 0.95
        draw_box(ax, DEC_X, d_cross_y, 5.8, 1.2,
                 'Cross-Attention (h=8)\nQ from Decoder\nK, V from Encoder\n编码器-解码器交叉注意力', C_ATTN, fontsize=9)

        draw_plus_circle(ax, DEC_X, d_res2_y)
        draw_arrow(ax, DEC_X, d_norm1_y - 0.35, DEC_X, d_res2_y + 0.2)
        draw_arrow(ax, DEC_X, d_res2_y - 0.2, DEC_X, d_cross_y + 0.65)
        ax.plot([DEC_X + 3.3, DEC_X + 3.3], [d_norm1_y - 0.05, d_res2_y], color='#E74C3C', lw=1.5,
                linestyle='--', zorder=3)

        d_norm2_y = d_cross_y - 0.95
        draw_box(ax, DEC_X, d_norm2_y, 2.5, 0.6, 'Add & LayerNorm', C_NORM, fontsize=8)
        draw_arrow(ax, DEC_X, d_cross_y - 0.65, DEC_X, d_norm2_y + 0.35)

        # Feed-Forward
        d_res3_y = d_norm2_y - 0.25
        d_ffn_y = d_res3_y - 0.9
        draw_box(ax, DEC_X, d_ffn_y, 5.8, 1.2,
                 f'Feed-Forward Network\nLinear→ReLU→Dropout→Linear\nd_ff={1024}', C_FFN, fontsize=9)

        draw_plus_circle(ax, DEC_X, d_res3_y)
        draw_arrow(ax, DEC_X, d_norm2_y - 0.35, DEC_X, d_res3_y + 0.2)
        draw_arrow(ax, DEC_X, d_res3_y - 0.2, DEC_X, d_ffn_y + 0.65)

        d_norm3_y = d_ffn_y - 0.95
        draw_box(ax, DEC_X, d_norm3_y, 2.5, 0.6, 'Add & LayerNorm', C_NORM, fontsize=8)
        draw_arrow(ax, DEC_X, d_ffn_y - 0.65, DEC_X, d_norm3_y + 0.35)
        ax.plot([DEC_X + 3.3, DEC_X + 3.3], [d_norm2_y - 0.05, d_res3_y], color='#E74C3C', lw=1.5,
                linestyle='--', zorder=3)

        if i == 2:
            d_last_norm_y = d_norm3_y

        if i < 2:
            draw_arrow(ax, DEC_X, d_norm3_y - 0.35, DEC_X, base_y - 5.6 + 0.5)

    # Decoder 最终 LayerNorm
    d_final_norm_y = d_last_norm_y - 1.2
    draw_box(ax, DEC_X, d_final_norm_y, 3, 0.7, 'LayerNorm (最终)', C_NORM, fontsize=9, bold=True)
    draw_arrow(ax, DEC_X, d_last_norm_y - 0.35, DEC_X, d_final_norm_y + 0.4)

    # ========================
    # 输出投影层
    # ========================
    proj_y = d_final_norm_y - 1.5
    draw_box(ax, DEC_X, proj_y, 6, 1.2, 'Linear Projection\n(d_model → vocab_size=4853)\n线性投影层', C_PROJ, fontsize=10, bold=True)
    draw_arrow(ax, DEC_X, d_final_norm_y - 0.4, DEC_X, proj_y + 0.65)

    # Softmax
    softmax_y = proj_y - 1.2
    draw_box(ax, DEC_X, softmax_y, 5, 0.8, 'Softmax\n输出概率分布', C_PROJ, fontsize=10)
    draw_arrow(ax, DEC_X, proj_y - 0.65, DEC_X, softmax_y + 0.45)

    # 输出
    out_y = softmax_y - 1.0
    draw_box(ax, DEC_X, out_y, 6, 0.9, 'Output: 英文 Token\n逐词生成 (Greedy/Beam Search)', C_INPUT, fontsize=10, bold=True)
    draw_arrow(ax, DEC_X, softmax_y - 0.45, DEC_X, out_y + 0.5)

    # ========================
    # 图例
    # ========================
    legend_y = 1.2
    legends = [
        (1.5, C_EMBED, 'Embedding 嵌入层'),
        (1.5 + 3.2, C_ATTN, 'Attention 注意力'),
        (1.5 + 6.4, C_FFN, 'Feed-Forward 前馈网络'),
        (1.5 + 10.0, C_NORM, 'LayerNorm 归一化'),
        (1.5 + 13.4, C_PROJ, 'Projection 投影层'),
    ]
    for x, color, label in legends:
        box = FancyBboxPatch((x - 0.2, legend_y - 0.2), 2.8, 0.55,
                             boxstyle="round,pad=0.05", facecolor=color,
                             edgecolor='#888888', linewidth=0.8)
        ax.add_patch(box)
        ax.text(x + 1.2, legend_y + 0.08, label, ha='center', va='center', fontsize=7.5)

    ax.text(9, legend_y + 0.8, 'Transformer 神经网络结构图（基于《Attention Is All You Need》）',
            ha='center', fontsize=16, fontweight='bold', color='#2C3E50')

    # 残差连接说明
    ax.text(16.5, legend_y - 0.6, '── 残差连接 (Residual Connection)', ha='center',
            fontsize=7.5, color='#E74C3C', style='italic')
    ax.text(16.5, legend_y - 1.0, '⊕  逐元素相加 (Element-wise Add)', ha='center',
            fontsize=7.5, color='#555555', style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Transformer architecture diagram saved to {output_path}")


def draw_code_organization(output_path='figures/code_organization.png'):
    """绘制代码模块组织结构图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_facecolor('white')

    modules = [
        # (x, y, w, h, name, desc, color, is_main)
        # 顶层：主入口
        (8, 11.2, 4, 0.8, 'train.py', '主训练入口', '#E74C3C', True),
        (8, 10.0, 4, 0.8, 'translate.py', '翻译推理入口', '#E74C3C', True),

        # 第二层：核心模块
        (2.5, 8.5, 3.5, 0.8, 'config.py', '配置管理\n(模型/训练/数据参数)', '#3498DB', False),
        (8, 8.5, 4.5, 0.8, 'model.py\n(280行)', 'Transformer完整实现\nEncoder/Decoder/Attention/FFN', '#2ECC71', True),
        (14, 8.5, 3.5, 0.8, 'dataset.py\n(110行)', 'BilingualDataset\n中文逐字分词/因果掩码', '#3498DB', False),

        # 第三层：模型子模块
        (1, 6.3, 2.8, 1.8, '', 'InputEmbeddings\n输入嵌入\nPositionalEncoding\n位置编码', '#E8D5B7', False),
        (4.2, 6.3, 3.2, 1.8, '', 'MultiHeadAttention\n多头注意力 (h=8)\nScaled Dot-Product\nAttention', '#A8D8EA', False),
        (7.8, 6.3, 2.8, 1.8, '', 'FeedForwardBlock\n前馈网络\nLinear→ReLU\n→Dropout→Linear', '#C3E6CB', False),
        (11, 6.3, 2.8, 1.8, '', 'LayerNormalization\n层归一化\nResidualConnection\n残差连接', '#F7DC6F', False),
        (14.2, 6.3, 2.8, 1.8, '', 'ProjectionLayer\n输出投影\n(d_model→vocab)', '#F5B7B1', False),

        # 第四层：组合模块
        (2.5, 4.2, 3.5, 1.2, '', 'EncoderBlock ×N\n自注意力 + FFN\n+ 残差连接 + LayerNorm', '#D5E8D4', False),
        (8, 4.2, 3.5, 1.2, '', 'DecoderBlock ×N\n掩码自注意力 + 交叉注意力\n+ FFN + 残差连接 + LayerNorm', '#FADBD8', False),
        (13, 4.2, 3.5, 1.2, '', 'Transformer\nencode() / decode()\n/ project()', '#D6EAF8', True),

        # 第五层：数据
        (2, 2.2, 3, 0.9, '', 'cmn-eng 数据集\n29,909 句对\n(中文→英文)', '#D5D8DC', False),
        (7, 2.2, 3.5, 0.9, '', 'Tokenizer\n中文: 字符级 (3,080)\n英文: 词级 (4,853)', '#D5D8DC', False),
        (12, 2.2, 3.5, 0.9, '', '解码策略\nGreedy Decode\nBeam Search (K=4)', '#D5D8DC', False),
    ]

    for x, y, w, h, name, desc, color, is_main in modules:
        edge_color = '#444' if is_main else '#888'
        lw = 2.5 if is_main else 1.5
        fontsize = 9 if is_main else 8
        fw = 'bold' if is_main else 'normal'

        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.08", facecolor=color,
                             edgecolor=edge_color, linewidth=lw)
        ax.add_patch(box)

        if name:
            ax.text(x, y + h/2 - 0.18, name, ha='center', va='top',
                    fontsize=fontsize, fontweight='bold', color='#2C3E50')
        ax.text(x, y - (0.05 if name else 0), desc, ha='center', va='center',
                fontsize=fontsize-1 if not name else 7, fontweight=fw,
                color='#333333')

    # 连接箭头（垂直）
    arrow_data = [
        # 入口 → 核心
        (8, 10.0, 8, 9.25),
        (8, 9.25, 8, 8.95),
        # config → model
        (4.2, 8.1, 5.5, 7.0),
        # model 子模块之间
        (2.5, 7.0, 2.5, 7.2),
        (2.5, 5.4, 2.5, 5.5),
        # EncoderBlock → EncoderBlock
        (2.5, 3.6, 2.5, 3.6),
        (8, 3.6, 8, 3.6),
        (13, 3.6, 13, 3.6),
    ]

    for x1, y1, x2, y2 in arrow_data:
        if x1 == x2 and y1 > y2:
            ax.annotate('', xy=(x2, y2 + 0.05), xytext=(x1, y1 - 0.05),
                        arrowprops=dict(arrowstyle='->', color='#666', lw=1.2))

    # 层间标签
    ax.text(2.5, 7.5, '│', ha='center', fontsize=14, color='#666')
    ax.text(2.5, 5.5, '│', ha='center', fontsize=14, color='#666')
    ax.text(8, 7.5, '│', ha='center', fontsize=14, color='#666')
    ax.text(8, 5.5, '│', ha='center', fontsize=14, color='#666')
    ax.text(13, 7.5, '│', ha='center', fontsize=14, color='#666')
    ax.text(13, 5.5, '│', ha='center', fontsize=14, color='#666')

    # 数据流说明
    ax.text(8, 2.85, '↓ 训练数据流', ha='center', fontsize=7, color='#888', style='italic')

    ax.text(8, 0.5, 'Transformer 中英机器翻译 — 代码模块组织结构图',
            ha='center', fontsize=16, fontweight='bold', color='#2C3E50')
    ax.text(8, 0.1, '箭头表示代码模块间的调用/依赖关系',
            ha='center', fontsize=9, color='#888', style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Code organization diagram saved to {output_path}")


if __name__ == '__main__':
    from pathlib import Path
    Path('figures').mkdir(exist_ok=True)
    draw_transformer_architecture('figures/transformer_architecture.png')
    draw_code_organization('figures/code_organization.png')
    print("All diagrams generated!")
