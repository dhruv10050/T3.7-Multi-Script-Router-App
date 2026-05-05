import torch
import torch.nn as nn

from config import IMG_SIZE


class ScriptRouter(nn.Module):
    """Stage-1 script identifier that classifies an input glyph into one of
    the four Indic scripts before it is routed to the appropriate character
    classifier."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ScriptCNN(nn.Module):
    """Stage-2 per-script character classifier with configurable depth.
    Depth is controlled by num_layers, enabling ablation studies to swap
    network capacity without modifying any other part of the architecture."""

    # Channel counts indexed by depth; shallower variants used in ablation
    # studies isolate the effect of network depth on classification accuracy.
    _CHANNELS = {2: [32, 64], 3: [32, 64, 128], 4: [32, 64, 128, 256]}

    def __init__(
        self, num_classes: int, num_layers: int = 4, dropout: float = 0.5
    ):
        super().__init__()
        channels = self._CHANNELS[num_layers]
        blocks, in_ch = [], 1
        for out_ch in channels:
            blocks += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)

        feat_hw = IMG_SIZE // (2 ** num_layers)
        fc_in   = channels[-1] * feat_hw * feat_hw
        # Hidden layer size scales proportionally with the feature map size
        # so shallower models do not receive a disproportionately large head.
        fc_hid  = max(256, fc_in // 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, fc_hid), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hid, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class TwoStagePipeline(nn.Module):
    """Inference-only wrapper that first identifies the script via the router
    and then dispatches the image to the corresponding per-script ScriptCNN."""

    SCRIPTS = ["devanagari", "tamil", "bengali", "telugu"]

    def __init__(self, router: ScriptRouter, classifiers: dict):
        super().__init__()
        self.router      = router
        self.classifiers = nn.ModuleDict(classifiers)

    def forward(self, x: torch.Tensor, topk: int = 5) -> dict:
        with torch.no_grad():
            s_logits    = self.router(x)
            script_id   = int(s_logits.argmax(dim=1)[0])
            script_name = self.SCRIPTS[script_id]
            c_logits    = self.classifiers[script_name](x)
            probs       = torch.softmax(c_logits, dim=1)
            k           = min(topk, probs.shape[1])
            top_p, top_i = probs.topk(k, dim=1)
            return {
                "script_id"  : script_id,
                "script_name": script_name,
                "script_conf": float(torch.softmax(s_logits, dim=1)[0, script_id]),
                "top_ids"    : top_i[0].cpu().tolist(),
                "top_probs"  : top_p[0].cpu().tolist(),
            }
