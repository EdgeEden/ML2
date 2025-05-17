import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================
# 配置参数
# ======================
class Config:
    pretrained_vec_path = "light_Tencent_AILab_ChineseEmbedding.bin"  # 二进制词向量文件路径
    synonym_file = "syno.txt"  # 同义词文件路径
    antonym_file = "anto.txt"
    embedding_dim = 200  # 与预训练向量维度一致
    batch_size = 32
    epochs = 30
    lr = 0.001
    alpha = 0.8  # 同义词相似度下限
    beta = 0.3  # 反义词相似度上限
    epsilon = 0.05  # 扰动强度
    lambda_adv = 0.4  # 对抗损失权重
    neg_sample_ratio = 3  # 负样本比例（每个正样本生成3个负样本）


# ======================
# 数据处理
# ======================
class SynonymDataset:
    def __init__(self, config):
        # 加载预训练词向量
        self.wv = KeyedVectors.load_word2vec_format(
            config.pretrained_vec_path,
            binary=True
        )

        # 加载同义词对
        self.syn_pairs = []
        with open(config.synonym_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    w1, w2 = parts
                    if w1 in self.wv and w2 in self.wv:
                        self.syn_pairs.append((w1, w2))

        # 加载反义词对
        self.neg_pairs = []
        with open(config.antonym_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    w1, w2 = parts
                    if w1 in self.wv and w2 in self.wv:
                        self.neg_pairs.append((w1, w2))

        # 构建词汇表
        self.vocab = list(self.wv.key_to_index.keys())
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {v: k for k, v in self.word2idx.items()}



    def get_batches(self):
        # 合并正负样本并打乱
        all_pairs = (
                [(pair, 1) for pair in self.syn_pairs] +
                [(pair, 0) for pair in self.neg_pairs]
        )
        np.random.shuffle(all_pairs)

        # 生成批次
        for i in range(0, len(all_pairs), Config.batch_size):
            batch = all_pairs[i:i + Config.batch_size]
            pos_batch = [pair for pair, label in batch if label == 1]
            neg_batch = [pair for pair, label in batch if label == 0]
            yield pos_batch, neg_batch


# ======================
# 模型定义
# ======================
class EnhancedWord2Vec(nn.Module):
    def __init__(self, word2idx, pretrained_vectors):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=len(word2idx),
            embedding_dim=pretrained_vectors.vector_size
        ).to(device)  # 初始化时直接放到GPU

        self._init_embeddings(word2idx, pretrained_vectors)

    def _init_embeddings(self, word2idx, wv):
        weight_matrix = np.zeros((len(word2idx), wv.vector_size))
        for word, idx in word2idx.items():
            weight_matrix[idx] = wv[word]
        self.embedding.weight.data.copy_(
            torch.from_numpy(weight_matrix).to(device)  # 数据拷贝到GPU
        )

    def forward(self, word_ids):
        return self.embedding(word_ids)

# ======================
# 对抗训练器
# ======================
class AdversarialTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        # 添加安全系数
        self.eps = 1e-10  # 防止除以零

    def contrastive_loss(self, pos_sims, neg_sims):
        pos_loss = torch.mean(torch.relu(self.config.alpha - pos_sims))
        neg_loss = torch.mean(torch.relu(neg_sims - self.config.beta))
        return pos_loss + neg_loss

    def generate_perturbation(self, anchor, negative):
        """生成对抗扰动 (FGSM)"""
        # 清零梯度
        self.model.zero_grad()

        neg_emb = self.model(negative)
        neg_emb.retain_grad()  # 保留梯度

        sim = torch.cosine_similarity(self.model(anchor), neg_emb)
        loss = torch.relu(sim - self.config.beta).mean()
        loss.backward()

        # 添加梯度安全处理
        grad = neg_emb.grad
        grad_norm = torch.norm(grad, dim=1, keepdim=True)
        grad_norm = torch.where(grad_norm < self.eps,
                                torch.ones_like(grad_norm) * self.eps,
                                grad_norm)

        delta = self.config.epsilon * (grad / grad_norm)
        return delta.detach()

    def train_step(self, word2idx, pos_batch, neg_batch, optimizer):
        # 转换输入时添加异常处理
        try:
            anchor_pos = torch.tensor([word2idx[p[0]] for p in pos_batch], device=device)
            positive = torch.tensor([word2idx[p[1]] for p in pos_batch], device=device)
            anchor_neg = torch.tensor([word2idx[n[0]] for n in neg_batch], device=device)
            negative = torch.tensor([word2idx[n[1]] for n in neg_batch], device=device)
        except KeyError as e:
            print(f"发现无效词汇: {e}")
            return 0.0  # 跳过包含无效词汇的批次

        # 生成对抗扰动
        delta = self.generate_perturbation(anchor_neg, negative)

        # 添加扰动安全限制
        perturbed_neg = self.model(negative) + delta
        perturbed_neg = torch.clamp(perturbed_neg, -10, 10)  # 限制数值范围

        # 计算损失时添加数值保护
        with torch.cuda.amp.autocast():  # 如果使用GPU则启用混合精度
            pos_sims = torch.cosine_similarity(
                self.model(anchor_pos),
                self.model(positive))
            # 添加相似度安全处理
            pos_sims = torch.clamp(pos_sims, -1 + self.eps, 1 - self.eps)

            neg_sims = torch.cosine_similarity(
                self.model(anchor_neg),
                self.model(negative))
            neg_sims = torch.clamp(neg_sims, -1 + self.eps, 1 - self.eps)

            orig_loss = self.contrastive_loss(pos_sims, neg_sims)

            adv_sims = torch.cosine_similarity(
                self.model(anchor_neg),
                perturbed_neg)
            adv_sims = torch.clamp(adv_sims, -1 + self.eps, 1 - self.eps)

            adv_loss = torch.relu(adv_sims - self.config.beta).mean()

            # 总损失
            total_loss = orig_loss + self.config.lambda_adv * adv_loss

            # 梯度裁剪
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()

        return total_loss.item()


# ======================
# 评估函数
# ======================
def evaluate(model, dataset):
    model.eval()
    pos_sims, neg_sims = [], []

    with torch.no_grad():
        for pos, neg in dataset.get_batches():
            # 正样本计算
            anchor_pos = torch.tensor([dataset.word2idx[p[0]] for p in pos], device=device)
            positive = torch.tensor([dataset.word2idx[p[1]] for p in pos], device=device)
            pos_emb = model(anchor_pos)
            pos_emb_p = model(positive)
            pos_sims.extend(torch.cosine_similarity(pos_emb, pos_emb_p).cpu().tolist())

            # 负样本计算
            anchor_neg = torch.tensor([dataset.word2idx[n[0]] for n in neg], device=device)
            negative = torch.tensor([dataset.word2idx[n[1]] for n in neg], device=device)
            neg_emb = model(anchor_neg)
            neg_emb_n = model(negative)
            neg_sims.extend(torch.cosine_similarity(neg_emb, neg_emb_n).cpu().tolist())

    print(f"[评估] 同义词平均相似度: {np.mean(pos_sims):.4f}")
    print(f"[评估] 反义词平均相似度: {np.mean(neg_sims):.4f}")
    print("=" * 50)


def save_updated_vectors(model, dataset, filename):
    """将更新后的词向量保存为Word2Vec格式的二进制文件"""
    from gensim.models import KeyedVectors

    # 获取词汇表和向量
    vocab = dataset.vocab
    vectors = model.embedding.weight.data.cpu().numpy()

    # 创建KeyedVectors实例
    kv = KeyedVectors(vector_size=vectors.shape[1])
    kv.add_vectors(
        keys=vocab,  # 必须按原始顺序排列
        weights=vectors
    )

    # 保存为二进制格式
    kv.save_word2vec_format(filename, binary=True)
    print(f"成功保存更新后的词向量到 {filename}")

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    # 初始化
    config = Config()
    dataset = SynonymDataset(config)
    model = EnhancedWord2Vec(dataset.word2idx, dataset.wv).to(device)
    trainer = AdversarialTrainer(model, config)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    torch.cuda.empty_cache()
    # 初始评估
    print("初始评估:")
    evaluate(model, dataset)

    # 训练循环
    for epoch in range(config.epochs):
        total_loss = 0
        model.train()

        for batch_idx, (pos_batch, neg_batch) in enumerate(dataset.get_batches()):
            if not pos_batch or not neg_batch:
                continue

            loss = trainer.train_step(dataset.word2idx, pos_batch, neg_batch, optimizer)
            total_loss += loss
            # print(total_loss)

        print(f"Epoch {epoch + 1}/{config.epochs} | Loss: {total_loss:.4f}")
        # 每5轮评估
        if (epoch + 1) % 5 == 0:
            evaluate(model, dataset)

    # 保存改进后的词向量
    save_updated_vectors(
        model=model,
        dataset=dataset,
        filename="enhanced_vectors.bin"  # 新的二进制文件名
    )
    saved_vectors = model.embedding.weight.data.cpu().numpy()
    np.save("enhanced_vectors.npy", saved_vectors)
