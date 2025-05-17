import torch
import torch.nn as nn

# 定义 SparseModule, get_w, get_m_tilde 和 compute_formula

class SparseModule(nn.Module):
    def __init__(self, channel=32, layers=6):
        super(SparseModule, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        self.alpha = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)

    def forward(self, m_hat_k_1, m_tilde_k_1):
        w_m_hat_k_1 = get_w(m_hat_k_1)
        w_m_tilde_k_1 = get_w(m_tilde_k_1)
        x = w_m_hat_k_1 * m_hat_k_1 - w_m_tilde_k_1 * m_tilde_k_1
        # x = m_hat_k_1 - m_tilde_k_1
        s = self.convs(x)
        w_s = get_w(s)
        s = self.alpha * s * w_s
        # s = self.alpha * s
        return s

def get_w(prob_map):
    mask = (prob_map > 0.4) & (prob_map < 0.6)
    modified_prob_map = torch.where(mask, torch.tensor(0.0), torch.tensor(1.0))
    return modified_prob_map

def get_m_tilde(prob_map):
    prob_map = torch.where((prob_map > 0.1) & (prob_map < 0.4), torch.tensor(0.1), prob_map)
    prob_map = torch.where((prob_map > 0.6) & (prob_map < 0.9), torch.tensor(0.9), prob_map)
    return prob_map

def compute_formula(I, B_k_1, m_tilde_k, m_hat_k_1, m_tilde_k_1, m_k_1, s, mu, alpha):
    I_squared = I ** 2
    w = get_w((m_tilde_k + m_hat_k_1 - m_tilde_k_1))
    w_squared = w ** 2
    ones_matrix = torch.ones_like(I)
    L_s = ones_matrix
    denominator = I_squared + L_s + mu * ones_matrix + 1e-6
    # denominator = I_squared + mu * ones_matrix + 1e-6
    # result = (I_squared - I * B_k_1 + alpha * L_s * (m_tilde_k + m_hat_k_1 - m_tilde_k_1) - s + mu * m_k_1) / denominator
    result = (I_squared - I * B_k_1 + alpha * L_s * w_squared * (m_tilde_k + m_hat_k_1 - m_tilde_k_1) - s + mu * m_k_1) / denominator
    return result


class SparsePipeline(nn.Module):
    def __init__(self, channels=32, layers=6, alpha=0.01, L_s=0.5):
        super(SparsePipeline, self).__init__()
        self.channels = channels
        self.layers = layers
        self.alpha = alpha
        self.L_s = L_s
        self.mu = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)
        self.sparse_module = SparseModule(channel=channels, layers=layers)

    def forward(self, I, B_k_1, m_k_1, m_hat_k_1, m_k_2):
        m_tilde_k_1 = get_m_tilde(m_k_2)
        m_tilde_k = get_m_tilde(m_k_1)

        s = self.sparse_module(m_hat_k_1, m_tilde_k_1)
        result = compute_formula(
            I=I, B_k_1=B_k_1, m_tilde_k=m_tilde_k, m_hat_k_1=m_hat_k_1,
            m_tilde_k_1=m_tilde_k_1, m_k_1=m_k_1, s=s, mu=self.mu, alpha=self.alpha
        )
        result = torch.mean(result, dim=1, keepdim=True)
        return result

# 使用类进行测试
if __name__ == "__main__":
    # 定义输入张量
    I = torch.rand(2, 3, 384, 384)  # 原始图像
    B_k_1 = torch.rand(2, 3, 384, 384)  # 背景图
    m_k_1 = torch.rand(2, 1, 384, 384)  # 前一时刻的m
    m_hat_k_1 = torch.rand(2, 1, 384, 384)  # 前一时刻的m_hat

    # 初始化测试类
    pipeline = SparsePipeline(channels=32, layers=6, alpha=0.01, L_s=0.5, mu=0.1)

    # 运行 pipeline
    result = pipeline.run(I, B_k_1, m_k_1, m_hat_k_1)

    # 打印结果
    print("Result shape:", result.shape)
    print("Result sample:", result[0, 0, :5, :5])  # 打印部分结果检查
