import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv1D Block (Conv1D + BN + ReLU)

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation
        )
        self.bn   = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): # x: (B, C, T)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

# Selective Kernel Convolution

class SKConv(nn.Module):
    def __init__(self, in_ch, out_ch, M=3, G=32, r=16):
        super().__init__()
        # reduction for squeeze
        d = max(in_ch // r, 4)
        # split: three 3×1 convs with dilations [1,2,3]
        dilations = [1, 2, 3]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch,
                          kernel_size=3,
                          padding=dil,
                          dilation=dil,
                          groups=G),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            ) for dil in dilations
        ])
        # fuse: squeeze
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Sequential(
            nn.Conv1d(out_ch, d, kernel_size=1, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True)
        )
        # select: excite
        self.attn_convs = nn.ModuleList([
            nn.Conv1d(d, out_ch, kernel_size=1) for _ in range(M)
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # split
        feats = torch.stack([b(x) for b in self.branches], dim=1)
        # fuse
        U = feats.sum(dim=1) # (B,C,H,W)
        s = self.gap(U) # (B,C,1,1)
        z = self.fc(s) # (B,d,1,1)
        # select
        # compute attention weights
        weights = torch.stack([conv(z) for conv in self.attn_convs], dim=1)
        attn    = self.softmax(weights) # (B, M, out_ch, 1)
        # aggregate
        V = (feats * attn).sum(dim=1) # (B,C,H,W)
        return V


# Recognition Head
class RecognitionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(RecognitionHead, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            num_classes,
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (B, C_multi, T)
        logits = self.conv(x) # (B, num_classes, T)
        return logits


# Segmentation Head
class SegmentationHead(nn.Module):
    def __init__(self, in_channels):
        super(SegmentationHead, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            2,
            kernel_size=1, 
            stride=1, 
            padding=0
        )

    def forward(self, x):
        # x: (B, C_multi, T)
        return self.conv(x)  # (B, 2, T)
    


# Multi Scale Window Generator
class MultiScaleWindowGenerator(nn.Module):
    """
    Generate, for each scale s in self.scales, two window lengths
      L1 = floor(N * sqrt(s)) and L2 = floor(N / sqrt(s)),
    then extract, at every feature position, the raw L-length slice
    (with zero-padding if L > N) from the feature sequence.
    Returns a list of 2*m tensors of shape (B, N, C, L_k).
    """
    def __init__(self, scales):
        super(MultiScaleWindowGenerator, self).__init__()
        self.scales = scales

    def forward(self, feature_seq):
        """
        Args:
            feature_seq: Tensor of shape (B, C, N)
                B = batch size
                C = channels
                N = feature-sequence length
        Returns:
            windows_per_length: list of 2*m tensors, each of shape (B, N, C, L_k)
        """
        B, C, N = feature_seq.size()
        device, dtype = feature_seq.device, feature_seq.dtype

        windows_per_length = []

        for s in self.scales:
            # compute both nominal lengths
            L1 = int(math.floor(N * math.sqrt(s)))
            L2 = int(math.floor(N / math.sqrt(s)))

            for L in (L1, L2):
                half = L // 2

                # allocate storage: (B, N, C, L)
                windows_L = torch.zeros(B, N, C, L, device=device, dtype=dtype)

                for center in range(N):
                    # compute raw window bounds
                    start = center - half
                    end = start + L

                    # determine how much padding is needed
                    pad_left  = max(0, -start)
                    pad_right = max(0, end - N)

                    # clamp slice indices to [0, N]
                    slice_start = max(start, 0)
                    slice_end   = min(end, N)

                    # extract the valid portion
                    raw_slice = feature_seq[:, :, slice_start:slice_end]  # (B, C, slice_len)

                    # pad to length L: pad takes (pad_left, pad_right) for last dim
                    padded = F.pad(raw_slice, (pad_left, pad_right), "constant", 0.0)  # (B, C, L)

                    # store into the buffer
                    windows_L[:, center, :, :] = padded

                windows_per_length.append(windows_L)

        return windows_per_length




# MTHARS Model
class MTHARS(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MTHARS, self).__init__()

        self.layer1 = ConvBlock1D(in_channels, 64, kernel_size=5, stride=3, padding=1, dilation=1)
        self.skconv1 = SKConv(64, 128)
        self.skconv2 = SKConv(128, 256)
        scales=[0.2, 0.5, 0.8]

        num_windows = len(scales) * 2
        expanded_channels = 256 * num_windows  # 256 from skconv output


        self.multi_window_gen = MultiScaleWindowGenerator(scales)

        self.recognition_head  = RecognitionHead(expanded_channels, num_classes)
        self.segmentation_head = SegmentationHead(expanded_channels)

    def forward(self, x):
        x = self.layer1(x)
        x = self.skconv1(x)
        x = self.skconv2(x)

        # print("shape before MSWG")
        # print (x.shape)
        windows = self.multi_window_gen(x)  # (B, T, C_multi)

        pooled = [w.mean(dim=-1) for w in windows]

        multi_feat = torch.cat(pooled, dim=2)

        # permute to (B, C_multi, N) for 1D conv heads
        multi_feat = multi_feat.permute(0, 2, 1)

        cls_logits   = self.recognition_head(multi_feat)  # (B, num_classes, N)
        offset_preds = self.segmentation_head(multi_feat) # (B, 2, N)

        return cls_logits, offset_preds
    



def main():

    B = 8     
    C = 3     
    T = 450   
    NUM_CLASSES = 11

    dummy_input = torch.randn(B, C, T)

    model = MTHARS(in_channels=C, num_classes=NUM_CLASSES)

    cls_logits, offset_preds = model(dummy_input)

    print("Classification logits shape:", cls_logits.shape)
    print("Offset prediction shape:", offset_preds.shape)

if __name__ == "__main__":
    main()
