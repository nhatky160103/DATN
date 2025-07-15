import torch
import math
import torch.nn.functional as F

class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s        

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits


class CombinedDynamicMarginLoss(torch.nn.Module):
    def __init__(self,
                 s: float = 64.0,
                 m1: float = 1.0,
                 m2: float = 0.5,
                 m3: float = 0,
                 alpha: float = 0.1,
                 interclass_filtering_threshold: float = 0.0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.alpha = alpha
        self.interclass_filtering_threshold = interclass_filtering_threshold

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        original_logits = logits.clone()
        adjusted_logits = original_logits.clone()
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive].unsqueeze(1), 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = logits * tensor_mul

        if index_positive.numel() == 0:
            return logits * self.s

        pos_labels = labels[index_positive]
        cos_y = logits[index_positive, pos_labels]

        logits_clone = logits[index_positive].clone()
        logits_clone[torch.arange(index_positive.size(0)), pos_labels] = -1e9
        max_other, _ = logits_clone.max(dim=1)

        h = 1.0 - (cos_y - max_other)
        m_i = self.m2 + self.alpha * h
        theta_y = torch.acos(cos_y.clamp(-1.0, 1.0))
        phi_y = torch.cos(self.m1 * theta_y + m_i) - self.m3

        # đảm bảo đồng biến: nếu phi_y < cos_y thì update
        # (tùy bạn muốn chắc chắn giữ thứ tự)
        mask_update = phi_y < cos_y
        final_phi = torch.where(mask_update, phi_y, cos_y)

        adjusted_logits[index_positive, pos_labels] = final_phi
        return adjusted_logits * self.s


class CombinedDynamicMarginLoss_arc(torch.nn.Module):
    def __init__(self,
                 s: float = 64.0,
                 m1: float = 1.0,
                 m2: float = 0.5,
                 m3: float = 0.0,
                 alpha: float = 0.1,
                 interclass_filtering_threshold: float = 0.0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.alpha = alpha
        self.interclass_filtering_threshold = interclass_filtering_threshold

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        original_logits = logits.clone()
        adjusted_logits = original_logits.clone()
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive].unsqueeze(1), 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = logits * tensor_mul

        if index_positive.numel() == 0:
            return logits * self.s

        pos_labels = labels[index_positive]
        cos_y = logits[index_positive, pos_labels]

        # Tìm cos(max) của các class khác
        logits_clone = logits[index_positive].clone()
        logits_clone[torch.arange(index_positive.size(0)), pos_labels] = -1e9
        max_other, _ = logits_clone.max(dim=1)

        # Chuyển về góc
        theta_y = torch.acos(cos_y.clamp(-1.0, 1.0))
        theta_max = torch.acos(max_other.clamp(-1.0, 1.0))

        # Tính khoảng cách góc rồi chuẩn hóa theo π/2
        delta_theta = theta_max - theta_y
        h = math.pi/2 - delta_theta
        h = torch.clamp(h, 0.0, math.pi/3)

        # Tính margin động
        m_i = self.m2 + self.alpha * h
        phi_y = torch.cos(self.m1 * theta_y + m_i) - self.m3

        # Giữ thứ tự nếu phi_y < cos_y
        mask_update = phi_y < cos_y
        final_phi = torch.where(mask_update, phi_y, cos_y)

        adjusted_logits[index_positive, pos_labels] = final_phi
        return adjusted_logits * self.s



class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits

if __name__ == "__main__":
    # Example usage (4 identity(class) and vector dim = 3)
    logits = torch.tensor([
    [0.1, 0.2, 0.7],
    [0.3, 0.4, 0.3],
    [0.2, 0.5, 0.3],
    [0.9, 0.05, 0.05],
], dtype=torch.float32)

    labels = torch.tensor([2, 1, 1, 0])
    loss = CombinedDynamicMarginLoss(64, 1, 0.5, 0.0)
    output = loss(logits, labels)
    print(output)

   

