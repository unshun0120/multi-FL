import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torchvision.models as models
from utils.nets import ResNet, BasicBlock

class SlamDunkNet(nn.Module):
    def __init__(self, in_channels, num_nodes, num_candidates, feat_dim=256):
        super(SlamDunkNet, self).__init__()

        # Random initialized ResNet18
        #backbone = models.resnet18(pretrained=False)

        # ImageNet pre-trained ResNet18
        backbone = models.resnet18(pretrained=True)

        self.feature = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, feat_dim)

        self.gating_head = nn.Linear(feat_dim, num_nodes)
        self.classification_head = nn.Linear(feat_dim, num_candidates)

        nn.init.zeros_(self.gating_head.weight)
        nn.init.zeros_(self.gating_head.bias)
        nn.init.zeros_(self.classification_head.weight)
        nn.init.zeros_(self.classification_head.bias)

    def forward(self, x):
        feat = self.feature(x).flatten(1)
        feat = self.fc(feat)

        gate_logits = self.gating_head(feat)
        cls_logits = self.classification_head(feat)

        return gate_logits, cls_logits


def slam_dunk_label_mapping(get_images_func, dataset_ids, label_space_meta, logger, valid_labels_dict=None, clients_dict=None, **get_images_kwargs):
    """
    Multi-dataset SLAMDUNKS-style label mapping.
    It directly aligns multiple datasets together instead of doing pairwise alignment.
    """

    args = get_images_kwargs.get("args", None)
    gen_dict = get_images_kwargs.get("gen_dict", None)

    device = getattr(args, "device", None) if args is not None else None
    if device is None:
        if gen_dict is not None and len(gen_dict) > 0:
            first_gen = list(gen_dict.values())[0]
            device = next(first_gen.parameters()).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    slam_epochs = 20
    slam_lr = 1e-4
    slam_batch_size = 64
    slam_samples_per_label = 64
    slam_lambda = 0.5
    slam_feat_dim = 256
    slam_relation_threshold = 0.0
    slam_standalone_margin = 1.5
    slam_max_relation_order = 2

    local_id_to_global_id = {}
    for d_id in dataset_ids:
        local_id_to_global_id[d_id] = {}

    logger.log(f"SLAM-DUNK | epochs: {slam_epochs}, lr: {slam_lr}, lambda: {slam_lambda}")
    logger.log(f"SLAM-DUNK | relation_threshold: {slam_relation_threshold}, standalone_margin: {slam_standalone_margin}")

    node_to_info, info_to_node = {}, {}
    node_idx = 0
    label_options = []

    for d_id in dataset_ids:
        labels = []
        for l_idx in range(len(label_space_meta[d_id])):
            if valid_labels_dict is not None and d_id in valid_labels_dict:
                if l_idx not in valid_labels_dict[d_id]:
                    continue
            node_to_info[node_idx] = (d_id, l_idx)
            info_to_node[(d_id, l_idx)] = node_idx
            labels.append(l_idx)
            node_idx += 1
        label_options.append([-1] + labels)

    num_nodes = len(node_to_info)
    if num_nodes == 0:
        logger.log("SLAM-DUNK | No valid labels found.")
        return local_id_to_global_id

    candidates = []
    standalone_candidate = {}
    node_to_shared_candidates = {n: [] for n in range(num_nodes)}

    for cand in itertools.product(*label_options):
        if all(v == -1 for v in cand):
            continue

        relation_order = sum([1 for v in cand if v != -1])
        if slam_max_relation_order is not None and relation_order > slam_max_relation_order:
            continue

        c_idx = len(candidates)
        candidates.append(cand)

        included_nodes = []
        for d_pos, l_idx in enumerate(cand):
            if l_idx == -1:
                continue
            d_id = dataset_ids[d_pos]
            if (d_id, l_idx) in info_to_node:
                included_nodes.append(info_to_node[(d_id, l_idx)])

        if relation_order == 1 and len(included_nodes) == 1:
            standalone_candidate[included_nodes[0]] = c_idx
        elif relation_order >= 2:
            for n in included_nodes:
                node_to_shared_candidates[n].append(c_idx)

    num_candidates = len(candidates)
    logger.log(f"SLAM-DUNK | nodes: {num_nodes}, candidates: {num_candidates}")

    dataset_x, dataset_node, images_by_node = [], [], {}

    for d_id in dataset_ids:
        for l_idx in range(len(label_space_meta[d_id])):
            if (d_id, l_idx) not in info_to_node:
                continue

            if get_images_kwargs:
                imgs = get_images_func(d_id, l_idx, **get_images_kwargs)
            else:
                imgs = get_images_func(d_id, l_idx)

            if imgs is None:
                continue

            imgs = imgs.detach()
            if imgs.size(0) > slam_samples_per_label:
                imgs = imgs[:slam_samples_per_label]

            n_id = info_to_node[(d_id, l_idx)]
            images_by_node[n_id] = imgs.detach().to(device)

            dataset_x.append(imgs.detach().cpu())
            dataset_node.append(torch.full((imgs.size(0),), n_id, dtype=torch.long))

    if not dataset_x:
        logger.log("SLAM-DUNK | No images collected for alignment.")
        return local_id_to_global_id

    dataset_x = torch.cat(dataset_x, dim=0)
    dataset_node = torch.cat(dataset_node, dim=0)

    in_channels = dataset_x.size(1)
    train_dataset = TensorDataset(dataset_x, dataset_node)
    train_loader = DataLoader(train_dataset, batch_size=slam_batch_size, shuffle=True)

    model = SlamDunkNet(in_channels, num_nodes, num_candidates, feat_dim=slam_feat_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=slam_lr)

    for epoch in tqdm(range(slam_epochs), colour="green", ncols=100):
        model.train()
        epoch_loss = 0.0

        for imgs, node_batch in train_loader:
            imgs, node_batch = imgs.to(device), node_batch.to(device)

            gate_logits, cls_logits = model(imgs)
            loss = slam_dunk_loss(gate_logits, cls_logits, node_batch, node_to_shared_candidates, standalone_candidate, slam_lambda)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logger.log(f"SLAM-DUNK | Epoch {epoch} Loss: {epoch_loss / max(1, len(train_loader)):.4f}")

    local_id_to_global_id = extract_slam_dunk_mapping(
        model=model,
        images_by_node=images_by_node,
        node_to_info=node_to_info,
        candidates=candidates,
        dataset_ids=dataset_ids,
        local_id_to_global_id=local_id_to_global_id,
        node_to_shared_candidates=node_to_shared_candidates,
        standalone_candidate=standalone_candidate,
        relation_threshold=slam_relation_threshold,
        standalone_margin=slam_standalone_margin,
        batch_size=slam_batch_size,
        device=device,
        logger=logger,
    )

    for d_id in dataset_ids:
        for l_idx in range(len(label_space_meta[d_id])):
            if valid_labels_dict is not None and d_id in valid_labels_dict:
                if l_idx not in valid_labels_dict[d_id]:
                    continue

            if l_idx not in local_id_to_global_id[d_id]:
                current_max = -1
                for d in local_id_to_global_id:
                    if local_id_to_global_id[d]:
                        current_max = max(current_max, max(local_id_to_global_id[d].values()))
                local_id_to_global_id[d_id][l_idx] = current_max + 1

    return local_id_to_global_id


def slam_dunk_loss(gate_logits, cls_logits, node_batch, node_to_shared_candidates, standalone_candidate, slam_lambda):
    """
    gating_head:
        sigmoid(gate) means this local label tends to be standalone.

    classification_head:
        predicts one candidate universal class.
    """

    eps = 1e-8
    cls_probs = F.softmax(cls_logits, dim=1)
    gate_probs = torch.sigmoid(gate_logits)

    losses = []

    for i in range(node_batch.size(0)):
        n_id = node_batch[i].item()

        shared_indices = node_to_shared_candidates.get(n_id, [])
        standalone_idx = standalone_candidate.get(n_id, None)

        if standalone_idx is None:
            continue

        if len(shared_indices) > 0:
            shared_mass = cls_probs[i, shared_indices].sum()
        else:
            shared_mass = torch.tensor(0.0, device=cls_probs.device)

        standalone_mass = cls_probs[i, standalone_idx]
        standalone_gate = gate_probs[i, n_id]

        posterior = (1.0 - standalone_gate) * shared_mass + standalone_gate * standalone_mass
        loss = -torch.log(posterior + eps) - slam_lambda * torch.log(standalone_gate + eps)
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=cls_logits.device, requires_grad=True)

    return torch.stack(losses).mean()


def extract_slam_dunk_mapping(model, images_by_node, node_to_info, candidates, dataset_ids, local_id_to_global_id, node_to_shared_candidates, standalone_candidate, relation_threshold, standalone_margin, batch_size, device, logger):
    model.eval()

    num_nodes = len(node_to_info)
    num_candidates = len(candidates)

    node_candidate_scores = torch.zeros(num_nodes, num_candidates, device=device)
    standalone_scores = torch.zeros(num_nodes, device=device)

    with torch.no_grad():
        for n_id, imgs in images_by_node.items():
            if imgs is None or imgs.size(0) == 0:
                continue

            shared_indices = node_to_shared_candidates.get(n_id, [])
            standalone_idx = standalone_candidate.get(n_id, None)

            if standalone_idx is None:
                continue

            all_shared_scores = []
            all_standalone_scores = []

            for start in range(0, imgs.size(0), batch_size):
                batch = imgs[start:start + batch_size].to(device)
                gate_logits, cls_logits = model(batch)

                cls_probs = F.softmax(cls_logits, dim=1)
                gate_probs = torch.sigmoid(gate_logits)

                standalone_gate = gate_probs[:, n_id]
                all_standalone_scores.append(standalone_gate * cls_probs[:, standalone_idx])

                if len(shared_indices) > 0:
                    shared_score = (1.0 - standalone_gate).unsqueeze(1) * cls_probs[:, shared_indices]
                    all_shared_scores.append(shared_score)

            if all_standalone_scores:
                standalone_scores[n_id] = torch.cat(all_standalone_scores, dim=0).mean()

            if all_shared_scores and len(shared_indices) > 0:
                node_candidate_scores[n_id, shared_indices] = torch.cat(all_shared_scores, dim=0).mean(dim=0)

    best_candidate_by_node = {}

    for n_id in range(num_nodes):
        shared_indices = node_to_shared_candidates.get(n_id, [])
        if len(shared_indices) == 0:
            continue

        scores = node_candidate_scores[n_id, shared_indices]
        best_pos = torch.argmax(scores).item()
        best_candidate_by_node[n_id] = shared_indices[best_pos]

    candidate_results = []

    for c_idx, cand in enumerate(candidates):
        included_nodes = []

        for d_pos, l_idx in enumerate(cand):
            if l_idx == -1:
                continue
            d_id = dataset_ids[d_pos]

            for n_id, info in node_to_info.items():
                if info == (d_id, l_idx):
                    included_nodes.append(n_id)
                    break

        if len(included_nodes) < 2:
            continue

        scores = [node_candidate_scores[n_id, c_idx].item() for n_id in included_nodes]
        avg_score = sum(scores) / max(1, len(scores))

        if avg_score < relation_threshold:
            continue

        is_mutual = True
        for n_id in included_nodes:
            if best_candidate_by_node.get(n_id, None) != c_idx:
                is_mutual = False
                break
            if node_candidate_scores[n_id, c_idx] <= standalone_scores[n_id] * standalone_margin:
                is_mutual = False
                break

        if is_mutual:
            candidate_results.append((avg_score, c_idx, included_nodes))

    candidate_results = sorted(candidate_results, key=lambda x: x[0], reverse=True)

    for score, c_idx, included_nodes in candidate_results:
        labels = [node_to_info[n_id] for n_id in included_nodes]
        register_group(local_id_to_global_id, labels)

        label_text = " <==> ".join([f"{d}:{l}" for d, l in labels])
        logger.log(f"SLAM-DUNK Match | {label_text} | score: {score:.4f}")

    return local_id_to_global_id


def register_group(local_id_to_global_id, labels):
    if len(labels) < 2:
        return

    d1, l1 = labels[0]

    for i in range(1, len(labels)):
        d2, l2 = labels[i]
        register_mapping(local_id_to_global_id, d1, l1, d2, l2)


def register_mapping(local_id_to_global_id, d1, l1, d2, l2):
    gid_1 = local_id_to_global_id.get(d1, {}).get(l1)
    gid_2 = local_id_to_global_id.get(d2, {}).get(l2)

    if gid_1 is None and gid_2 is None:
        current_max = -1
        for d in local_id_to_global_id:
            if local_id_to_global_id[d]:
                current_max = max(current_max, max(local_id_to_global_id[d].values()))
        new_gid = current_max + 1
    else:
        new_gid = gid_1 if gid_1 is not None else gid_2

    if d1 in local_id_to_global_id:
        for existing_label, existing_gid in local_id_to_global_id[d1].items():
            if existing_gid == new_gid and existing_label != l1:
                return

    if d2 in local_id_to_global_id:
        for existing_label, existing_gid in local_id_to_global_id[d2].items():
            if existing_gid == new_gid and existing_label != l2:
                return

    if d1 not in local_id_to_global_id:
        local_id_to_global_id[d1] = {}
    if d2 not in local_id_to_global_id:
        local_id_to_global_id[d2] = {}

    if gid_1 is not None and gid_2 is not None and gid_1 != gid_2:
        clients_in_g1 = [d_name for d_name, lbls in local_id_to_global_id.items() if gid_1 in lbls.values()]
        clients_in_g2 = [d_name for d_name, lbls in local_id_to_global_id.items() if gid_2 in lbls.values()]

        if set(clients_in_g1).intersection(set(clients_in_g2)):
            return

        for d_name in local_id_to_global_id:
            for label_name, gid in list(local_id_to_global_id[d_name].items()):
                if gid == gid_2:
                    local_id_to_global_id[d_name][label_name] = gid_1

        new_gid = gid_1

    local_id_to_global_id[d1][l1] = new_gid
    local_id_to_global_id[d2][l2] = new_gid