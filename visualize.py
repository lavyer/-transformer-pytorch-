"""
可视化工具：用于生成课程设计报告所需的图表。
包括: 训练损失曲线、BLEU分数曲线、注意力热力图等。
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path
import os
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curves(log_dir='runs/tmodel_zh_en', output_dir='figures'):
    """
    从 TensorBoard 日志中提取数据并绘制训练曲线。
    如果没有 TensorBoard 日志，可以从训练日志文件读取。
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 尝试从 TensorBoard 事件文件读取
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()

        tags = ea.Tags()
        print(f"Available tags: {tags}")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 训练损失曲线
        ax1 = axes[0, 0]
        if 'train/loss_step' in tags['scalars']:
            events = ea.Scalars('train/loss_step')
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax1.plot(steps, values, alpha=0.3, color='blue', linewidth=0.5, label='Step Loss')
            # 平滑曲线
            if len(values) > 100:
                window = len(values) // 100
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                ax1.plot(steps[window-1:], smoothed, 'b-', linewidth=1.5, label='Smoothed')
            ax1.set_xlabel('Global Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No training data found', ha='center', va='center', transform=ax1.transAxes)

        # 2. 学习率曲线
        ax2 = axes[0, 1]
        if 'train/lr' in tags['scalars']:
            events = ea.Scalars('train/lr')
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax2.plot(steps, values, 'g-', linewidth=1)
            ax2.set_xlabel('Global Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule (Warmup)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No LR data found', ha='center', va='center', transform=ax2.transAxes)

        # 3. BLEU 分数曲线
        ax3 = axes[1, 0]
        bleu_keys = [k for k in tags['scalars'] if 'bleu' in k.lower()]
        colors = {'greedy': 'blue', 'beam': 'red'}
        for key in bleu_keys:
            events = ea.Scalars(key)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            label = key.split('/')[-1]
            color = 'blue' if 'greedy' in label else 'red'
            ax3.plot(steps, values, 'o-', markersize=4, linewidth=1.5, label=label, color=color)
        ax3.set_xlabel('Global Step')
        ax3.set_ylabel('BLEU Score')
        ax3.set_title('Validation BLEU Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. CER/WER 曲线
        ax4 = axes[1, 1]
        error_keys = [k for k in tags['scalars'] if 'cer' in k.lower() or 'wer' in k.lower()]
        for key in error_keys:
            events = ea.Scalars(key)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            label = key.split('/')[-1]
            ax4.plot(steps, values, 'o-', markersize=4, linewidth=1.5, label=label)
        ax4.set_xlabel('Global Step')
        ax4.set_ylabel('Error Rate')
        ax4.set_title('Validation Error Rate (CER/WER)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle('Transformer Chinese-English Machine Translation - Training Curves', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {output_dir}/training_curves.png")

    except Exception as e:
        print(f"Could not read TensorBoard logs: {e}")
        print("Generating placeholder figure...")

        # 生成占位图表框架
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'Train the model first\nto generate curves',
                    ha='center', va='center', fontsize=14, transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[0, 0].set_title('Training Loss')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[1, 0].set_title('BLEU Score')
        axes[1, 1].set_title('Error Rate (CER/WER)')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'training_curves_placeholder.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_attention_map(model, tokenizer_src, tokenizer_tgt, sentence, device,
                       output_path='figures/attention_map.png', layer_idx=-1, head_idx=0):
    """
    绘制注意力热力图。

    Args:
        model: Transformer 模型
        tokenizer_src: 源语言分词器
        tokenizer_tgt: 目标语言分词器
        sentence: 待翻译的源语言句子
        device: 设备
        output_path: 输出图片路径
        layer_idx: 可视化哪一层（-1 表示最后一层）
        head_idx: 可视化哪一个注意力头
    """
    from dataset import add_spaces_to_chinese, causal_mask
    import torch

    config = __import__('config').get_config()

    model.eval()
    with torch.no_grad():
        # 编码源语言
        src_text = add_spaces_to_chinese(sentence)
        src_tokens = tokenizer_src.encode(src_text).ids
        seq_len = config['seq_len']
        src_len = len(src_tokens)

        # 填充
        num_padding = seq_len - src_len - 2
        src_input = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(src_tokens, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * num_padding, dtype=torch.int64)
        ]).unsqueeze(0).to(device)

        src_mask = (src_input != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

        # 前向传播获取注意力权重
        # 我们需要 hook 到编码器的自注意力层
        attention_weights = []

        def hook_fn(module, input, output):
            # MultiHeadAttentionBlock stores attention_scores
            if hasattr(module, 'attention_scores') and module.attention_scores is not None:
                attention_weights.append(module.attention_scores.detach().cpu())

        hooks = []
        for layer in model.encoder.layers:
            hooks.append(layer.self_attention_block.register_forward_hook(hook_fn))

        # 前向传播
        encoder_output = model.encode(src_input, src_mask)

        # 移除 hooks
        for h in hooks:
            h.remove()

        # 绘制注意力图
        if attention_weights:
            # 取指定层的注意力
            attn = attention_weights[layer_idx][0, head_idx]  # (seq_len, seq_len)
            actual_len = min(src_len + 2, attn.shape[0])
            attn = attn[:actual_len, :actual_len]

            # Token 标签
            token_labels = ['[SOS]'] + list(sentence) + ['[EOS]']

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(attn.numpy(), cmap='YlOrRd', aspect='auto')

            ax.set_xticks(range(actual_len))
            ax.set_xticklabels(token_labels[:actual_len], rotation=90, fontsize=10)
            ax.set_yticks(range(actual_len))
            ax.set_yticklabels(token_labels[:actual_len], fontsize=10)

            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            ax.set_title(f'Encoder Self-Attention (Layer {layer_idx}, Head {head_idx})\nSource: {sentence}',
                         fontsize=12, fontweight='bold')

            plt.colorbar(im, ax=ax)
            plt.tight_layout()

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Attention map saved to {output_path}")
        else:
            print("No attention weights captured. Make sure the model has been run forward first.")


def plot_translation_examples(examples, output_path='figures/translation_examples.png'):
    """
    绘制翻译示例对比表。

    Args:
        examples: list of (source, target, predicted_greedy, predicted_beam)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, len(examples) * 1.5))
    ax.axis('off')

    # 创建表格
    col_labels = ['Source (Chinese)', 'Reference (English)', 'Greedy Decode', 'Beam Search']
    table_data = []
    for src, ref, greedy, beam in examples:
        table_data.append([src, ref, greedy, beam])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='left',
        loc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表头样式
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 交替行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[i, j].set_facecolor('#D9E2F3')
            else:
                table[i, j].set_facecolor('#FFFFFF')

    ax.set_title('Transformer Chinese-English Translation Examples',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Translation examples table saved to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default='runs/tmodel_zh_en')
    parser.add_argument('--output-dir', default='figures')
    args = parser.parse_args()

    plot_training_curves(args.log_dir, args.output_dir)
    print("Visualization complete!")
