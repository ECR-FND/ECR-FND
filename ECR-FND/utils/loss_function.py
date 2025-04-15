import torch
import torch.nn.functional as F

class Loss_function(nn.Module):
    def __init__(self, margin=0.2, tau=0.07):
        super(Loss_function, self).__init__()
        self.margin = margin
        self.tau = tau

    def margin_ranking_loss(self, consistency_scores):

        batch_size, _ = consistency_scores.shape
        loss = 0.0
        for i in range(batch_size):
            scores = consistency_scores[i]  # (L_video,)
            s_high = torch.max(scores)
            s_low = torch.min(scores)
            loss += F.relu(self.margin + s_low - s_high)
        return loss / batch_size

    def global_contrastive_loss(self, text_global, video_global):

        batch_size = text_global.size(0)

        sim_matrix = torch.matmul(text_global, video_global.t()) / self.tau
        labels = torch.arange(batch_size).to(text_global.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss
