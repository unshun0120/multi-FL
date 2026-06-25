import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels)

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        if with_conv:
            self.op = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, temb_channels=512, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels or in_channels
        
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        
        self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        
        self.norm2 = Normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        
        if in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = F.silu(h) 
        h = self.conv1(h)
        
        temb = self.temb_proj(F.silu(temb))
        h = h + temb[:, :, None, None] 
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return self.nin_shortcut(x) + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W)
        
        w = torch.bmm(q.transpose(1, 2), k) * (int(C) ** (-0.5))
        w = F.softmax(w, dim=-1)
        
        h = torch.bmm(v, w.transpose(1, 2))
        h = h.reshape(B, C, H, W)
        h = self.proj_out(h)
        
        return x + h

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, has_attn, dropout):
        super().__init__()
        self.res = ResnetBlock(in_channels, out_channels, temb_channels, dropout)
        if has_attn:
            self.attn = AttnBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb):
        x = self.res(x, temb)
        x = self.attn(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, has_attn, dropout):
        super().__init__()
        self.res = ResnetBlock(in_channels, out_channels, temb_channels, dropout)
        if has_attn:
            self.attn = AttnBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb):
        x = self.res(x, temb)
        x = self.attn(x)
        return x
    
class ContextUnet(nn.Module):
    def __init__(self, in_channels=3, n_feat=64, n_classes=10, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        
        ch = n_feat
        ch_mult = (1, 2, 2, 2) 
        num_res_blocks = 2
        temb_ch = ch * 4
        
        self.time_embed = nn.Sequential(
            nn.Linear(ch, temb_ch),
            Swish(),
            nn.Linear(temb_ch, temb_ch)
        )
        self.class_embed = nn.Sequential(
            nn.Linear(n_classes, temb_ch),
            Swish(),
            nn.Linear(temb_ch, temb_ch)
        )

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        self.down = nn.ModuleList()
        
        in_ch_list = [ch]
        now_ch = ch
        for i_level, mult in enumerate(ch_mult):
            out_ch = ch * mult
            has_attn = (i_level == 1)
            for i_block in range(num_res_blocks):
                self.down.append(DownBlock(now_ch, out_ch, temb_ch, has_attn, dropout))
                now_ch = out_ch
                in_ch_list.append(now_ch)
            
            if i_level != len(ch_mult) - 1:
                self.down.append(Downsample(now_ch))
                in_ch_list.append(now_ch)

        self.mid_block1 = ResnetBlock(now_ch, now_ch, temb_channels=temb_ch, dropout=dropout)
        self.mid_attn = AttnBlock(now_ch)
        self.mid_block2 = ResnetBlock(now_ch, now_ch, temb_channels=temb_ch, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            has_attn = (i_level == 1)
            for i_block in range(num_res_blocks + 1):
                skip_ch = in_ch_list.pop()
                self.up.append(UpBlock(now_ch + skip_ch, out_ch, temb_ch, has_attn, dropout))
                now_ch = out_ch
                
            if i_level != 0:
                self.up.append(Upsample(now_ch))

        self.norm_out = Normalize(now_ch)
        self.conv_out = nn.Conv2d(now_ch, in_channels, kernel_size=3, stride=1, padding=1)

    def get_timestep_embedding(self, timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, x, c, t, context_mask):
        t = t.view(-1)
        t_emb = self.get_timestep_embedding(t * 1000, self.n_feat) 
        t_emb = self.time_embed(t_emb)
        
        c_onehot = F.one_hot(c, num_classes=self.n_classes).float()
        context_mask = context_mask[:, None].repeat(1, self.n_classes)
        context_mask = (-1 * (1 - context_mask))
        c_onehot = c_onehot * context_mask
        c_emb = self.class_embed(c_onehot)
        
        temb = t_emb + c_emb 
        
        h = self.conv_in(x)
        hs = [h]
        
        for module in self.down:
            if isinstance(module, DownBlock):
                h = module(h, temb)
            else:
                h = module(h)
            hs.append(h)
            
        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)
        
        for module in self.up:
            if isinstance(module, UpBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, temb)
            else:
                h = module(h)
                
        h = self.norm_out(h)
        h = F.silu(h)
        out = self.conv_out(h)
        
        return out


def ddpm_schedules(beta1, beta2, T):
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t, "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t, "alphabar_t": alphabar_t,
        "sqrtab": sqrtab, "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )
        context_mask = torch.bernoulli(torch.zeros_like(c, dtype=torch.float)+self.drop_prob).to(self.device)
        
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w = 0.0, label=None):
        x_i = torch.randn(n_sample, *size).to(device)
        if label is not None:
            c_i = torch.full((n_sample,), label, dtype=torch.long, device=device)
        else:
            c_i = torch.arange(0, 10).to(device) 
            c_i = c_i.repeat(max(1, int(n_sample/c_i.shape[0])) + 1)[:n_sample]


        context_mask = torch.zeros_like(c_i).to(device)

        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.

        x_i_store = []
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            x_i_double = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.nn_model(x_i_double, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        return x_i, x_i_store
