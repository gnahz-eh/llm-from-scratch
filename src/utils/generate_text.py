import torch
import torch.nn as nn

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx是当前上下文中的索引数组，形状为(B, T)
    for _ in range(max_new_tokens):

        # 如果当前上下文超过了支持的长度，就对当前上下文进行截断
        # 例如，如果LLM只支持5个token，而上下文长度为10，
        # 那么只有最后5个token会被用作上下文

        idx_cond = idx[:, -context_size:]
        
        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 只关注最后一个时间步
        # (batch, n_token, vocab_size)变为(batch, vocab_size)
        logits = logits[:, -1, :]  

        # 通过softmax函数获得对应的概率
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # 获取概率值最高的单词索引
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # 将采样到的索引添加到当前运行的上下文索引序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def generate(model, idx, max_new_tokens, context_size, temperature, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # 使用top_k采样对logits值进行过滤
        if top_k is not None:
            # 仅保留top_k的值
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 使用温度缩放
        if temperature > 0.0:
            logits = logits / temperature

            # 使用softmax函数得到概率
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # 从概率分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 否则和之前的generate_simple函数中的处理相同，使用argmax函数取得概率最大的token
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 和之前相同的序列拼接处理
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx