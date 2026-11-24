import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model):
        self.model = model.eval()

    def __call__(self, images, target=None):
        images = images.clone().requires_grad_(True)
        with torch.enable_grad():
            features = self.model.forward_features(images)
            features.retain_grad()
            logits = self.model.forward_head(features)
            if target is None:
                target = logits.argmax(1)
            score = logits.gather(1, target.view(-1, 1)).sum()
            self.model.zero_grad(set_to_none=True)
            score.backward()
        weights = features.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * features).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.flatten(1)
        low = flat.min(dim=1, keepdim=True).values
        high = flat.max(dim=1, keepdim=True).values
        cam = ((flat - low) / (high - low + 1e-8)).view_as(cam)
        probs = torch.softmax(logits.detach().float(), dim=1)
        return cam.detach().cpu().numpy(), probs.cpu().numpy()
