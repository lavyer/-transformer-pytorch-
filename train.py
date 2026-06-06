from model import build_transformer
from dataset import BilingualDataset, causal_mask, add_spaces_to_chinese
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Tokenizers for building char-level Chinese + word-level English tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


# ========================
# Greedy Decoding
# ========================
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


# ========================
# Beam Search Decoding
# ========================
def beam_search_decode(model, source, source_mask, tokenizer_tgt, max_len, device,
                        beam_size=4, length_penalty_alpha=0.6):
    """
    Beam Search 解码，带长度惩罚。

    Args:
        model: Transformer 模型
        source: 源语言编码器输入 (1, seq_len)
        source_mask: 编码器 mask (1, 1, 1, seq_len)
        tokenizer_src: 源语言分词器
        tokenizer_tgt: 目标语言分词器
        max_len: 最大生成长度
        device: 设备
        beam_size: beam 宽度
        length_penalty_alpha: 长度惩罚系数（0=无惩罚, 1=强惩罚）

    Returns:
        最佳解码序列 (seq_len,)
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # 预计算编码器输出
    encoder_output = model.encode(source, source_mask)

    # 每个 beam: (sequence, log_prob_sum, finished)
    # sequence: list of token indices
    beams = [([sos_idx], 0.0, False)]

    for _ in range(max_len - 1):
        all_candidates = []

        for seq, log_prob_sum, finished in beams:
            if finished:
                # 已完成的序列保持不动
                all_candidates.append((seq, log_prob_sum, finished, 0.0))
                continue

            # 构建解码器输入
            decoder_input = torch.tensor([seq], dtype=torch.long).to(device)  # (1, cur_len)
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

            # 计算输出
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            prob = model.project(out[:, -1])  # (1, vocab_size)
            log_probs = torch.log_softmax(prob, dim=-1).squeeze(0)  # (vocab_size,)

            # 取 top-k
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

            for k in range(beam_size):
                token = topk_indices[k].item()
                log_p = topk_log_probs[k].item()
                new_seq = seq + [token]
                new_sum = log_prob_sum + log_p
                new_finished = (token == eos_idx)
                all_candidates.append((new_seq, new_sum, new_finished, log_p))

        # 排序：已完成序列排在前面且不受长度惩罚影响
        # 未完成序列用长度惩罚
        def score_fn(candidate):
            seq, log_sum, finished, _ = candidate
            length = len(seq)
            if finished:
                return log_sum / (length ** length_penalty_alpha)
            else:
                return log_sum / (length ** length_penalty_alpha)

        # 按分数排序并保留 top beam_size
        ordered = sorted(all_candidates, key=score_fn, reverse=True)
        beams = []
        seen_seqs = set()
        for seq, log_sum, finished, _ in ordered:
            # 避免重复序列（去重）
            seq_key = tuple(seq)
            if seq_key in seen_seqs:
                continue
            seen_seqs.add(seq_key)
            beams.append((seq, log_sum, finished))
            if len(beams) >= beam_size:
                break

        # 如果所有 beam 都已完成，提前结束
        if all(finished for _, _, finished in beams):
            break

    # 返回最佳序列（去除 SOS token）
    best_seq = beams[0][0]
    return torch.tensor(best_seq, dtype=torch.long, device=device)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer,
                   num_examples=3, use_beam_search=True, beam_size=4):
    """验证函数：分别用 Greedy Decode 和 Beam Search 解码并计算 BLEU/CER/WER。"""
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted_greedy = []
    predicted_beam = []

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Greedy decode
            model_out_greedy = greedy_decode(model, encoder_input, encoder_mask,
                                             tokenizer_src, tokenizer_tgt, max_len, device)

            # Beam search decode
            if use_beam_search:
                model_out_beam = beam_search_decode(model, encoder_input, encoder_mask,
                                                    tokenizer_tgt, max_len, device,
                                                    beam_size=beam_size)
            else:
                model_out_beam = model_out_greedy

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            model_out_text_greedy = tokenizer_tgt.decode(model_out_greedy.detach().cpu().numpy())
            model_out_text_beam = tokenizer_tgt.decode(model_out_beam.detach().cpu().numpy()) if use_beam_search else model_out_text_greedy

            source_texts.append(source_text)
            expected.append(target_text)
            predicted_greedy.append(model_out_text_greedy)
            predicted_beam.append(model_out_text_beam)

            # 打印结果
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'GREEDY: ':>12}{model_out_text_greedy}")
            if use_beam_search:
                print_msg(f"{f'BEAM({beam_size}): ':>12}{model_out_text_beam}")

            if count >= num_examples:
                print_msg('-' * console_width)
                break

    if writer:
        # ---- Greedy 评估 ----
        if predicted_greedy:
            metric_cer = torchmetrics.CharErrorRate()
            cer_g = metric_cer(predicted_greedy, expected)
            writer.add_scalar('validation/greedy_cer', cer_g, global_step)

            metric_wer = torchmetrics.WordErrorRate()
            wer_g = metric_wer(predicted_greedy, expected)
            writer.add_scalar('validation/greedy_wer', wer_g, global_step)

            metric_bleu = torchmetrics.BLEUScore()
            bleu_g = metric_bleu(predicted_greedy, [[e] for e in expected])
            writer.add_scalar('validation/greedy_bleu', bleu_g, global_step)

        # ---- Beam Search 评估 ----
        if use_beam_search and predicted_beam:
            metric_cer2 = torchmetrics.CharErrorRate()
            cer_b = metric_cer2(predicted_beam, expected)
            writer.add_scalar('validation/beam_cer', cer_b, global_step)

            metric_wer2 = torchmetrics.WordErrorRate()
            wer_b = metric_wer2(predicted_beam, expected)
            writer.add_scalar('validation/beam_wer', wer_b, global_step)

            metric_bleu2 = torchmetrics.BLEUScore()
            bleu_b = metric_bleu2(predicted_beam, [[e] for e in expected])
            writer.add_scalar('validation/beam_bleu', bleu_b, global_step)

        writer.flush()

    # 返回最后一个样本用于打印
    return source_texts, expected, predicted_greedy, predicted_beam


# ========================
# 加载 cmn-eng 数据集
# ========================
def load_cmn_eng_dataset(data_dir="cmn-eng (1)"):
    """
    加载本地的 cmn-eng 数据集（制表符分隔的英中句对）。
    格式: English sentence \t Chinese sentence \t metadata
    """
    data_path = Path(data_dir) / "cmn.txt"
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                en_text = parts[0].strip()
                zh_text = parts[1].strip()
                # 过滤过长的句子
                if len(en_text) > 200 or len(zh_text) > 200:
                    continue
                pairs.append({
                    'translation': {
                        'en': en_text,
                        'zh': zh_text
                    }
                })
    print(f"Loaded {len(pairs)} sentence pairs from {data_path}")
    return pairs


def get_all_sentences(ds, lang):
    """生成器：逐个返回指定语言的句子。"""
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang, char_level=False):
    """
    构建或加载分词器。
    - char_level=True: 用于中文，在字符之间添加空格后再分词
    - char_level=False: 用于英文，按空格分词（WordLevel）
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        # 英文限制词表大小，中文不限制
        trainer_kwargs = dict(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )
        if not char_level:
            trainer_kwargs['vocab_size'] = config.get('english_vocab_size', 30000)
        trainer = WordLevelTrainer(**trainer_kwargs)
        if char_level:
            # 中文逐字分词：生成时自动加空格
            def char_level_iterator(ds, lang):
                for item in ds:
                    yield add_spaces_to_chinese(item['translation'][lang])
            tokenizer.train_from_iterator(char_level_iterator(ds, lang), trainer=trainer)
        else:
            tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    """加载数据并构建 dataloader。"""
    # 加载本地 cmn-eng 数据集
    ds_raw_list = load_cmn_eng_dataset(data_dir=config.get('data_dir', 'cmn-eng (1)'))

    # 包装为简单的列表 Dataset
    class ListDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    # 可选：限制训练样本数以加速CPU训练
    max_samples = config.get('max_train_samples', len(ds_raw_list))
    if max_samples and max_samples < len(ds_raw_list):
        import random
        random.seed(42)
        ds_raw_list = random.sample(ds_raw_list, max_samples)
        print(f"Sampled {max_samples} sentence pairs for training")

    full_ds = ListDataset(ds_raw_list)

    # 构建分词器（中文用字符级，英文用词级）
    print("Building tokenizers...")
    tokenizer_src = get_or_build_tokenizer(config, full_ds, config['lang_src'], char_level=True)  # 中文逐字
    tokenizer_tgt = get_or_build_tokenizer(config, full_ds, config['lang_tgt'], char_level=False)  # 英文按词

    print(f"Source vocabulary size: {tokenizer_src.get_vocab_size()}")
    print(f"Target vocabulary size: {tokenizer_tgt.get_vocab_size()}")

    # 划分训练集和验证集（90/10）
    train_ds_size = int(0.9 * len(full_ds))
    val_ds_size = len(full_ds) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(full_ds, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'],
                                config['seq_len'], char_level_src=True)
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                              config['lang_src'], config['lang_tgt'],
                              config['seq_len'], char_level_src=True)

    # 查找最大句子长度
    max_len_src = 0
    max_len_tgt = 0
    for item in full_ds:
        src_text = item['translation'][config['lang_src']]
        tgt_text = item['translation'][config['lang_tgt']]
        src_ids = tokenizer_src.encode(add_spaces_to_chinese(src_text)).ids
        tgt_ids = tokenizer_tgt.encode(tgt_text).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # 过滤掉超出 seq_len 的样本
    def collate_filter(batch_list):
        """过滤过长的样本后再 collate。"""
        valid_batch = []
        for item in batch_list:
            try:
                valid_batch.append(item)
            except ValueError:
                continue  # 跳过过长样本
        if len(valid_batch) == 0:
            return None
        # 手动 collate
        keys = valid_batch[0].keys()
        collated = {}
        for key in keys:
            tensors = [item[key] for item in valid_batch]
            collated[key] = torch.stack(tensors) if isinstance(tensors[0], torch.Tensor) else tensors
        return collated

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                                  collate_fn=collate_filter)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    """构建 Transformer 模型。"""
    model = build_transformer(
        vocab_src_len, vocab_tgt_len,
        config["seq_len"], config['seq_len'],
        d_model=config['d_model'],
        N=config.get('n_layers', 3),
        h=config.get('n_heads', 8),
        dropout=config.get('dropout', 0.1),
        d_ff=config.get('d_ff', 1024)
    )
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    return model


def train_model(config):
    """主训练函数。"""
    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if device == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    device = torch.device(device)

    # 创建模型保存目录
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # 加载数据
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # TensorBoard
    writer = SummaryWriter(config['experiment_name'])

    # 优化器 Adam + Warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9, betas=(0.9, 0.98))

    # Warmup 学习率调度
    warmup_steps = 4000

    def lr_lambda(step):
        step += 1  # 避免 step=0 时除零
        return min(step ** (-0.5), step * warmup_steps ** (-1.5)) * (config['d_model'] ** (-0.5))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # 预加载检查点
    initial_epoch = 0
    global_step = 0
    preload = config.get('preload', None)
    model_filename = None
    if preload == 'latest':
        model_filename = latest_weights_file_path(config)
    elif preload:
        model_filename = get_weights_file_path(config, preload)
    if model_filename and os.path.exists(model_filename):
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # 损失函数
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'),
        label_smoothing=0.1
    ).to(device)

    # ========================
    # 训练循环
    # ========================
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        total_loss = 0.0
        num_batches = 0

        for batch in batch_iterator:
            if batch is None:
                continue  # 跳过空 batch

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # 前向传播
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            # 计算损失
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_loss += loss.item()
            num_batches += 1

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # TensorBoard 记录
            writer.add_scalar('train/loss_step', loss.item(), global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
            writer.flush()

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1

        # 每个 epoch 结束时训练损失
        avg_loss = total_loss / max(num_batches, 1)
        writer.add_scalar('train/avg_loss_epoch', avg_loss, epoch)

        # 验证
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                       config['seq_len'], device,
                       lambda msg: batch_iterator.write(msg),
                       global_step, writer,
                       num_examples=3, use_beam_search=True, beam_size=4)

        # 保存模型
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        print(f"Model saved to {model_filename}")

    print("Training completed!")
    writer.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
