import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy

# ==========================================
# Global Classifier
# ==========================================
class Classifier(nn.Linear):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__(in_features=input_dim, out_features=num_classes)

    def forward(self, x):
        return super(Classifier, self).forward(x)

# ==========================================
# Global Conditional Generator
# ==========================================
class ConditionalGenerator(nn.Module):
    def __init__(self, num_global_classes, noise_dim, output_dim, embedding_dim=32):
        super().__init__()
        self.noise_dim = noise_dim
        
        self.label_emb = nn.Embedding(num_global_classes, embedding_dim)
        
        input_dim = noise_dim + embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim)
        )

    def forward(self, z, labels):
        c = self.label_emb(labels) 
        x = torch.cat([z, c], dim=1) 
        out = self.net(x)
        return out

# ==========================================
# 1. Base Interface
# base class: 統一了各種異質模型的「介面」
# 異質 Backbone → 同質 Global Feature 空間, 這樣 server 才能操作
# ==========================================
class BaseHeteroModel(nn.Module):
    """
    所有異質模型的父類別。
    1. 必須定義 self.feature_extractor
    2. 必須定義 self.output_dim (Feature Extractor 的輸出維度)
    3. 必須定義 self.classifier (輸入維度為 self.output_dim)
    """
    def __init__(self):
        super().__init__()
        self.output_dim = 0 # 子類別需設定此值
    
    def forward(self, x):
        # 1. Extract Native Features (e.g., 1280)

        # self.feature_extractor: 
        # 子類別必須定義好 feature extractor（CNN / MLP / MobileNet…）
        # 輸出通常是 (B, C, H, W) 或已經 flatten 過的 (B, D)
        native_feat = self.feature_extractor(x)

        # 保險：不管 feature_extractor 輸出是 (B, C, 1, 1) 還是 (B, D)，統一拍平成 (B, native_dim)
        native_feat = torch.flatten(native_feat, 1)
        
        # 2. Adapt to Global Dimension (e.g., 1280 -> 256)
        # 這一步是關鍵，讓 classifier 的輸入維度統一
        # 把各種 backbones 的 native feature（維度可能是 64/256/512/1024…）投影到一個共同的 global_dim 空間（例如 256）
        global_feat = self.adapter(native_feat)
        
        # 3. Classify (256 -> 10)
        # self.classifier: 類別 logits，輸入維度是 global_dim，輸出是 num_classes
        logits = self.classifier(global_feat)
        
        # 回傳 global_feat 以便做 Contrastive Distillation
        # forward 回傳 (global_feat, logits)
        # local training 用 logits 做 CE loss
        # FL / KD / Prototype / Generator 用 global_feat
        return global_feat, logits

# ==========================================
# 2. Heterogeneous Architectures
# ==========================================

# --- ID 0: MLP (Native Dim depends on hidden layers) ---
# 直接把整張圖全部展平，經過 3 層 MLP 得到 256 維 feature
# 對 MNIST / 小圖 data 很 OK，用來模擬 edge device 上「超小模型」很合理
class MLP(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, img_size, global_dim=256):
        super().__init__()
        # MLP 的輸入維度高度依賴圖片大小
        flat_dim = in_channels * img_size * img_size
        
        # 定義一個原生 MLP 架構，假設最後一層隱藏層是 256
        self.output_dim = 256 
        
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim), # Native Feature
            nn.ReLU()
        )

        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() # 可以加個非線性
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 1: Simple CNN (Native Dim: 64) ---
# 經典的「三層 CNN + global pool」，輸出 channel=64 → flatten 成 64 維 feature
# AdaptiveAvgPool2d((1,1)) 讓你不管輸入 28×28 或 32×32 都可以用同一套 FC
class CNN(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        self.output_dim = 64 # 最後一層 Conv 的 Channel 數
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(64, self.output_dim, 3, padding=1), 
            nn.ReLU(), 
            nn.AdaptiveAvgPool2d((1, 1)) # 強制變成 (B, 64, 1, 1)
        )

        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() # 可以加個非線性
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 2, 3: ResNet (Native Dim: 64 ~ 512) ---
# num_blocks=[1,1,1,0] → ResNet8-like（最後沒有 layer4，output_dim=256）
# num_blocks=[2,2,2,2] → ResNet18（有 layer4，output_dim=512）
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(BaseHeteroModel):
    def __init__(self, block, num_blocks, in_channels, num_classes, global_dim=256):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # ResNet8 (num_blocks=[1,1,1,0]) 最後一層只有 256
        # ResNet18 (num_blocks=[2,2,2,2]) 最後一層有 512
        if num_blocks[3] > 0:
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.output_dim = 512 * block.expansion
        else:
            self.layer4 = nn.Identity()
            self.output_dim = 256 * block.expansion
            
        self.feature_extractor = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(),
            self.layer1, self.layer2, self.layer3, self.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() # 可以加個非線性
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

# --- ID 4: MobileNetV2 (Native Dim: 1280) ---
# 用 torchvision 的 MobileNetV2 backbone，去掉原本 head，最後 global pool → 1280 維
# 灰階資料（in_channels=1）時，替換第一層 conv
class MobileNetV2(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        model = models.mobilenet_v2(weights=None)
        if in_channels != 3:
            model.features[0][0] = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)
        
        self.output_dim = 1280 # MobileNetV2 standard output
        
        self.feature_extractor = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() # 可以加個非線性
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 5: MobileNetV3 (Native Dim: 576 for Small) ---
# 手刻的簡化版 MobileNetV3 Small-ish：
# depthwise conv + pointwise conv，加上 Hardswish。
# 最後透過 AdaptiveAvgPool2d(1) → (B, 576, 1, 1) → flatten 576 維。
class MobileNetV3(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        # 簡化版結構
        self.output_dim = 576
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.Hardswish(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(32), nn.Hardswish(inplace=True),
            nn.Conv2d(32, self.output_dim, 1, bias=False),
            nn.BatchNorm2d(self.output_dim), nn.Hardswish(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
       
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() # 可以加個非線性
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 6: LeNet (Native Dim: 84) ---
# 經典 LeNet 結構，輸入 MNIST 這種 28×28 / 32×32 小圖很適合
class LeNet(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        self.output_dim = 84
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((5, 5)), # LeNet 標準是展平成 16*5*5
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, self.output_dim), nn.ReLU()
        )
        
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() # 可以加個非線性
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 7: AlexNet (Native Dim: 4096) ---
# 一個「適合 CIFAR / MNIST」的小 AlexNet（因為原版 stride=4 太兇會把小圖變太小）。
# 透過 3 次 pool + 最後的 AdaptiveAvgPool2d((2,2))，保證最後 feature 是 (B, 256*2*2) → 1024 維。
class AlexNet(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        self.output_dim = 256 * 2 * 2 # 根據 AdaptiveAvgPool((2,2)) 輸出決定 -> 1024
        
        # 我們不能用 torchvision.models.alexnet，因為它第一層 stride=4 會把小圖殺死
        # 這裡是針對 CIFAR/MNIST 優化的小型 AlexNet
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Layer 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3, 4, 5
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 強制固定輸出大小，避免不同輸入尺寸導致 FC 層報錯
            nn.AdaptiveAvgPool2d((2, 2)), 
            
            nn.Flatten()
        )
        
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 8: ShuffleNetV2 (Native Dim: 1024) ---
# 用 torchvision ShuffleNetV2，小寬度版（x0.5）。
# 把最後原本的 classifier 拿掉，自己加 global pool + flatten。
class ShuffleNetV2(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        model = models.shufflenet_v2_x0_5(weights=None)
        if in_channels != 3:
            model.conv1[0] = nn.Conv2d(in_channels, 24, 3, 2, 1, bias=False)
        
        self.output_dim = 1024
        # 移除原版最後的 FC
        backbone = list(model.children())[:-1] 
        self.feature_extractor = nn.Sequential(
            *backbone, # 輸出已經是 (B, 1024, 1, 1) 因為 ShuffleNet 內部有 global pool
            # 因為 list(model.children()) 裡面沒有包含 Pooling 操作
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten() # 確保拉平
        )
        
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() # 可以加個非線性
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)

# --- ID 9: SqueezeNet (Native Dim: 512) ---
# SqueezeNet 1.1 backbone，最後一層 conv channel 是 512 → flatten 成 512 維
class SqueezeNet(BaseHeteroModel):
    def __init__(self, in_channels, num_classes, global_dim=256):
        super().__init__()
        self.output_dim = 512
        model = models.squeezenet1_1(weights=None)
        if in_channels != 3:
            model.features[0] = nn.Conv2d(in_channels, 64, 3, 2)
            
        self.feature_extractor = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, global_dim),
            nn.ReLU() # 可以加個非線性
        )
        
        self.classifier = nn.Linear(global_dim, num_classes)


# ==========================================
# Twin Branch Nets (for FedTED baseline)
# ==========================================

class TwinBranchNets(nn.Module):
    def __init__(self, base_model: BaseHeteroModel):
        super().__init__()
        self.feature_extractor = base_model.feature_extractor
        self.adapter = base_model.adapter
        self.classifier = base_model.classifier  # generic branch
        self.twin_classifier = copy.deepcopy(base_model.classifier)  # personalized branch
        self.output_dim = base_model.output_dim
        self.use_twin = False
    
    def forward(self, x):
        native_feat = self.feature_extractor(x)
        native_feat = torch.flatten(native_feat, 1)
        global_feat = self.adapter(native_feat)
        
        if self.use_twin:
            logits = self.classifier(global_feat) + self.twin_classifier(global_feat)
        else:
            logits = self.classifier(global_feat)
        
        return global_feat, logits


def get_twin_branch_model(client_id, in_channels, num_classes, img_size, global_dim=256):
    base_model = get_heterogeneous_model(client_id, in_channels, num_classes, img_size, global_dim)
    return TwinBranchNets(base_model)   


# ==========================================
# 4. UDON Components (for UDON baseline)
# ==========================================


class CosineClassifier(nn.Module):
    """Cosine similarity classifier (normalized weights, no bias)"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, input_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x):
        # L2 normalize weights and input
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        logits = F.linear(x_norm, weight_norm)
        return logits


class UDONModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes, 
                 student_dim=[64], teacher_dim=[256]):
        super(UDONModel, self).__init__()
        
        # 1. Backbone (Shared)
        self.backbone = backbone
        self.feature_dim = feature_dim
        
        # 2. Universal Student Projection (Shared in FL)
        # Corresponds to 'universal_student_projection_domain_0'
        # Output dim is usually the last element of the list
        self.universal_projection = MLP(feature_dim, student_dim[:-1], student_dim[-1])
        
        # 3. Teacher Projection (Private/Local in FL)
        # Corresponds to 'teacher_projection_domain_{domain}'
        self.teacher_projection = MLP(feature_dim, teacher_dim[:-1], teacher_dim[-1])
        
        # 4. Classifiers (Private/Local in FL)
        # UDON uses separate classifiers per domain
        # Universal Student Head
        self.student_classifier = CosineClassifier(student_dim[-1], num_classes)
        # Teacher Head
        self.teacher_classifier = CosineClassifier(teacher_dim[-1], num_classes)

    def forward(self, x, train=True):
        outputs = {}
        
        # Backbone features
        backbone_feats = self.backbone(x)
        # Flatten if needed (assuming backbone output is [B, C, H, W] or similar)
        if len(backbone_feats.shape) > 2:
            backbone_feats = backbone_feats.view(backbone_feats.size(0), -1)
            
        # L2 Normalize backbone features (as per Flax code)
        backbone_feats = F.normalize(backbone_feats, p=2, dim=1)
        
        outputs['backbone_out'] = backbone_feats
        
        # --- Teacher Branch ---
        teacher_embedd = self.teacher_projection(backbone_feats)
        teacher_embedd = F.normalize(teacher_embedd, p=2, dim=1)
        outputs['teacher_embedd'] = teacher_embedd
        
        teacher_logits = self.teacher_classifier(teacher_embedd, normalize_input=False) # already normalized
        outputs['teacher_logits'] = teacher_logits
        
        # --- Universal Student Branch ---
        student_embedd = self.universal_projection(backbone_feats)
        student_embedd = F.normalize(student_embedd, p=2, dim=1)
        outputs['universal_student_embedd'] = student_embedd
        
        student_logits = self.student_classifier(student_embedd, normalize_input=False)
        outputs['universal_student_logits'] = student_logits
        
        return outputs


class CCVAE(nn.Module):
    def __init__(self, num_classes=10, latent_size=16, img_size=32, channels=3, **kwargs):
        super(CCVAE, self).__init__()
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.img_size = img_size
        self.channels = channels
        
        # Encoder
        self.conv1 = nn.Conv2d(channels + 1, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Calculate Flatten Size
        self.flat_size = 256 * (img_size // 8) * (img_size // 8) 
        
        self.mu = nn.Linear(self.flat_size, latent_size)
        self.logvar = nn.Linear(self.flat_size, latent_size)

        # Decoder
        self.linear = nn.Linear(latent_size + num_classes, self.flat_size)
        self.convT1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bnT1 = nn.BatchNorm2d(128)
        self.convT2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bnT2 = nn.BatchNorm2d(64)
        self.convT3 = nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1)

    def encode(self, x, y):
        # Conditioning on label
        y_cond = y.argmax(dim=1).view(-1, 1, 1, 1).to(x.device)
        y_cond = torch.ones(x.size(0), 1, x.size(2), x.size(3)).to(x.device) * y_cond
        
        x = torch.cat([x, y_cond], dim=1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        
        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.linear(z))
        x = x.view(x.size(0), 256, self.img_size // 8, self.img_size // 8)
        
        x = F.relu(self.bnT1(self.convT1(x)))
        x = F.relu(self.bnT2(self.convT2(x)))
        x = torch.tanh(self.convT3(x)) # Output range [-1, 1]
        return x

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        
        # Concat label info for decoder
        z = torch.cat([z, y.float()], dim=1)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def sample(self, num_samples, labels=None, device='cuda'):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_size).to(device)
            if labels is None:
                labels = torch.randint(0, self.num_classes, (num_samples,), device=device)
            
            y_onehot = F.one_hot(labels, self.num_classes).float().to(device)
            z = torch.cat([z, y_onehot], dim=1)
            return self.decode(z), labels

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ==========================================
# 3. Factory Function
# 用 client_id % 10 決定這個 client 用哪一種 backbone → Model Heterogeneity
# ==========================================
def get_heterogeneous_model(node_id, in_channels, num_classes, img_size, global_dim=256):
    model_idx = node_id % 10
    
    # MLP 需要 img_size 來決定第一層輸入
    if model_idx == 0: return MLP(in_channels, num_classes, img_size, global_dim)
    
    # 這些模型使用 AdaptiveAvgPool，對 img_size 不敏感，但需要 in_channels
    if model_idx == 1: return CNN(in_channels, num_classes, global_dim)
    
    # ResNet Variants
    if model_idx == 2: return ResNet(BasicBlock, [1, 1, 1, 0], in_channels, num_classes, global_dim) # ResNet8-like
    if model_idx == 3: return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_classes, global_dim) # ResNet18
    
    if model_idx == 4: return MobileNetV2(in_channels, num_classes, global_dim)
    if model_idx == 5: return MobileNetV3(in_channels, num_classes, global_dim)
    if model_idx == 6: return LeNet(in_channels, num_classes, global_dim)
    if model_idx == 7: return AlexNet(in_channels, num_classes, global_dim)
    if model_idx == 8: return ShuffleNetV2(in_channels, num_classes, global_dim)
    if model_idx == 9: return SqueezeNet(in_channels, num_classes, global_dim)
    
    return MLP(in_channels, num_classes, img_size, global_dim)