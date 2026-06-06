"""
基于 Transformer 的中英机器翻译 - 推理脚本
支持 Greedy Decode 和 Beam Search 两种解码策略。
用法:
    python translate.py              # 交互模式
    python translate.py "你好世界"    # 单句翻译
"""
from pathlib import Path
from config import get_config, latest_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
from dataset import BilingualDataset, causal_mask, add_spaces_to_chinese
import torch
import sys


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    """贪心解码"""
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def beam_search_decode(model, source, source_mask, tokenizer_tgt, max_len, device,
                        beam_size=4, length_penalty_alpha=0.6):
    """
    Beam Search 解码，带长度惩罚。

    Args:
        beam_size: beam 宽度（越大效果越好但越慢）
        length_penalty_alpha: 长度惩罚系数
            - 0.0: 无惩罚（偏好短序列）
            - 0.6~1.0: 适度惩罚短序列
            - 1.0: 强惩罚（偏好长序列）
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)

    # 每个 beam: (sequence_list, log_prob_sum, is_finished)
    beams = [([sos_idx], 0.0, False)]

    for _ in range(max_len - 1):
        all_candidates = []

        for seq, log_prob_sum, finished in beams:
            if finished:
                all_candidates.append((seq, log_prob_sum, finished))
                continue

            decoder_input = torch.tensor([seq], dtype=torch.long).to(device)
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            prob = model.project(out[:, -1])
            log_probs = torch.log_softmax(prob, dim=-1).squeeze(0)

            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

            for k in range(beam_size):
                token = topk_indices[k].item()
                log_p = topk_log_probs[k].item()
                new_seq = seq + [token]
                new_sum = log_prob_sum + log_p
                new_finished = (token == eos_idx)
                all_candidates.append((new_seq, new_sum, new_finished))

        # 按长度惩罚后的分数排序
        def score_fn(candidate):
            seq, log_sum, _ = candidate
            return log_sum / (len(seq) ** length_penalty_alpha)

        ordered = sorted(all_candidates, key=score_fn, reverse=True)

        # 去重并保留 top beam_size
        beams = []
        seen = set()
        for seq, log_sum, finished in ordered:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            beams.append((seq, log_sum, finished))
            if len(beams) >= beam_size:
                break

        if all(finished for _, _, finished in beams):
            break

    best_seq = beams[0][0]
    return torch.tensor(best_seq, dtype=torch.long, device=device)


def translate(sentence: str, use_beam=True, beam_size=4):
    """
    翻译一句中文到英文。

    Args:
        sentence: 中文句子
        use_beam: 是否使用 Beam Search
        beam_size: Beam 大小

    Returns:
        (greedy_result, beam_result)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()

    # 加载分词器
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))

    print(f"Source vocab size: {tokenizer_src.get_vocab_size()}")
    print(f"Target vocab size: {tokenizer_tgt.get_vocab_size()}")

    # 构建模型
    model = build_transformer(
        tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
        config["seq_len"], config['seq_len'],
        d_model=config['d_model'],
        N=config.get('n_layers', 3),
        h=config.get('n_heads', 8),
        dropout=config.get('dropout', 0.1),
        d_ff=config.get('d_ff', 1024)
    ).to(device)

    # 加载权重
    model_filename = latest_weights_file_path(config)
    if model_filename is None:
        print("ERROR: No trained model found! Please train the model first.")
        print(f"Expected weights in: {config['datasource']}_{config['model_folder']}/")
        return None, None
    print(f"Loading model: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    print(f"Loaded model from epoch {state['epoch'] + 1}")

    seq_len = config['seq_len']

    model.eval()
    with torch.no_grad():
        # 编码源语言句子（中文需要逐字分词）
        src_text_processed = add_spaces_to_chinese(sentence)
        source = tokenizer_src.encode(src_text_processed)

        if len(source.ids) > seq_len - 2:
            print(f"Warning: Input too long ({len(source.ids)} tokens), truncating to {seq_len - 2}")
            source_ids = source.ids[:seq_len - 2]
        else:
            source_ids = source.ids

        num_padding = seq_len - len(source_ids) - 2
        source_tensor = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(source_ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * num_padding, dtype=torch.int64)
        ], dim=0).to(device)

        source_mask = (source_tensor != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

        # Greedy Decode
        result_greedy_ids = greedy_decode(model, source_tensor, source_mask, tokenizer_tgt, seq_len, device)
        result_greedy = tokenizer_tgt.decode(result_greedy_ids.detach().cpu().numpy())
        result_greedy = result_greedy.replace('[SOS]', '').replace('[EOS]', '').replace('[PAD]', '').strip()

        # Beam Search
        result_beam = None
        if use_beam:
            result_beam_ids = beam_search_decode(model, source_tensor, source_mask, tokenizer_tgt, seq_len, device,
                                                 beam_size=beam_size)
            result_beam = tokenizer_tgt.decode(result_beam_ids.detach().cpu().numpy())
            result_beam = result_beam.replace('[SOS]', '').replace('[EOS]', '').replace('[PAD]', '').strip()

        return result_greedy, result_beam


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Chinese->English Transformer Translation')
    parser.add_argument('sentence', nargs='?', default=None, help='Chinese sentence to translate')
    parser.add_argument('--no-beam', action='store_true', help='Disable Beam Search')
    parser.add_argument('--beam-size', type=int, default=4, help='Beam size (default: 4)')
    args = parser.parse_args()

    if args.sentence:
        sentence = args.sentence
    else:
        sentence = input("请输入中文句子（Chinese→English）: ").strip()
        if not sentence:
            sentence = "今天天气真好。"

    print(f"\n{'='*60}")
    print(f"Source (Chinese): {sentence}")
    print(f"{'='*60}")

    greedy_res, beam_res = translate(sentence, use_beam=not args.no_beam, beam_size=args.beam_size)

    if greedy_res:
        print(f"\n  Greedy Decode: {greedy_res}")

    if beam_res:
        print(f"  Beam Search ({args.beam_size}): {beam_res}")

    # 交互模式
    if not args.sentence:
        print(f"\n{'='*60}")
        print("交互模式 - 输入中文句子进行翻译，输入 'quit' 退出")
        print(f"{'='*60}")
        while True:
            try:
                text = input("\n中文 > ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                if not text:
                    continue
                greedy_res, beam_res = translate(text, use_beam=not args.no_beam, beam_size=args.beam_size)
                if greedy_res:
                    print(f"  Greedy: {greedy_res}")
                if beam_res:
                    print(f"  Beam:   {beam_res}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
