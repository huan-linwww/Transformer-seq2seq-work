import torch
from torch import nn

from model.transformer import Transformer
from conf import *
from data import *
from collections import Counter
import math
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO,  # 设置日志级别为 INFO
                    format='%(asctime)s - %(levelname)s - %(message)s')  # 日志格式

# 创建日志记录器
logger = logging.getLogger(__name__)

# 打印日志信息
# logger.debug('This is a debug message')  # 调试信息（开发阶段使用）
# logger.info('This is an info message')   # 一般信息（正常运行时）
# logger.warning('This is a warning message')  # 警告信息
# logger.error('This is an error message')  # 错误信息
# logger.critical('This is a critical message')  # 严重错误信息

model = Transformer(vocab_size=vocab_size , dec_vocab_size=vocab_size, embed_size=embeding_size, max_len=max_len, hidden_size= embeding_size, feedforward_size=ffn_hidden, head_num=n_heads, dropout=drop_prob, device=device, layer_num=n_layers, src_pad_idx=pad_id, trg_pad_idx=pad_id, trg_sos_idx=sos_id)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

model.apply(initialize_weights)#初始化模型参数

optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, eps=adam_eps, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True, eps=1e-8)

criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch["input"]
        trg = batch["label"]
        
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])# tgt从sos到wn
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)# 适应crossentropy， [(batch_size * (trg_seq_len - 1)), vocab_size]
        trg = trg[:, 1:].reshape(-1)#拉为一维 [(batch_size * (trg_seq_len - 1))]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)#用来裁剪梯度范数
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def match_ngram(prediction, tgt):
    stats = []
    stats.append(len(prediction))
    stats.append(len(tgt))
    
    for n in range(5):
        s =Counter([tuple(prediction[i:i+n]) for i in range(len(prediction) - n)])
        
        r =Counter([tuple(tgt[i:i+n]) for i in range(len(tgt) - n)])
        
        stats.append(max(sum(s & r).values(), 0))
        stats.append(max(len(prediction) + 1 - n, 0))
    return stats
    
# BLEU的计算，用于评估
def BLEU(stats):
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    r , c = stats[:2]
    
    log_bleu = sum([math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])] )
    return math.exp(min(0, 1 - float(r) / c) + log_bleu / 4)

def get_bleu(predictions, reference):
    stats = np.zeros(10)
    for pred, tgt in zip(predictions, reference):
        stats += np.array(match_ngram(pred, tgt))
    return BLEU(stats) * 100
    
def idx_to_word(idx, vocab):
    return ' '.join([vocab.itos[i] for i in idx])

def eval(model, iterator, criterion):
    model.eval()
    loss_sum = 0
    batch_blue = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch["input"]
            trg = batch["label"]
            output = model(src, trg[:, :-1])# tgt从sos到wn
            output_dim = output.shape[-1]
            output_reshape = output[:, 1:].reshape(-1, output_dim)# 适应crossentropy， [(batch_size * (trg_seq_len - 1)), vocab_size]
            trg = trg[:, 1:].reshape(-1)#拉为一维 [(batch_size * (trg_seq_len - 1))]
            loss = criterion(output_reshape, trg)
            loss_sum += loss.item()
            total_bleu = 0
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch["label"][j], tokenizer.vocab)
                    output_words = output[j].max(dim=1)[1]# 返回的是 (values, indices),这里获取下标
                    output_words = idx_to_word(output_words, tokenizer.vocab)
                    logger.info('输出的句子为：{}'.format(output_words))
                    bleu = get_bleu(prediction=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass
            batch_blue.append(np.mean(total_bleu))
    return loss_sum / len(iterator), np.mean(batch_blue)


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        train_loss = train(model, train_dataloader, optimizer, criterion, clip)
        valid_loss, bleu = eval(model, val_data_loader, criterion)

        if step > warmup:
            scheduler.step(valid_loss)# 根据验证损失调整学习率

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)