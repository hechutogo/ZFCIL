import torch
import torch.nn as nn
import torch.nn.functional as F
import models.clip.clip as clip
import json
import numpy as np


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class ZFCIL(nn.Module):

    def __init__(self, cfg, template, device):
        super(ZFCIL, self).__init__()
        self.cfg = cfg
        self.device = device
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        print(f"Prompt template:{template}")
        self.template = template
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.ZFCIL.PREC == "fp32" or cfg.TRAINER.ZFCIL.PREC == "amp":
            clip_model.float()

        clip_model.eval()
        self.clip_model = clip_model.to(self.device)
        self.text_proto = None
        self.description_proto = None
        self.vision_proto = None

    @torch.no_grad()
    def inference_text_feature(self, class_names, template, class_index):
        print(f'class names: {class_names}')
        clip_weights = []
        all_targets = []

        k = 0
        for classname in class_names:
            targets = torch.full((len(template),), class_index[k])
            all_targets.append(targets)
            k += 1
            classname = classname.replace('_', ' ')
            classname = classname.replace('-', ' ')

            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = self.clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=0)
        clip_weights = F.normalize(clip_weights, dim=-1)
        all_targets = torch.cat(all_targets, dim=0)
        return clip_weights, all_targets

    @torch.no_grad()
    def inference_all_img_feature(self, loader, cls_begin_index):
        all_features = []
        all_labels = []
        for batch in loader:
            images, labels = self.parse_batch(batch)
            features = self.clip_model.encode_image(images)
            features = F.normalize(features, dim=-1)
            all_features.append(features)
            all_labels.append(labels)
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        unique_labels = torch.unique(all_labels)
        print(f'all targets:{unique_labels}')
        prototypes = []
        for c in unique_labels:
            idx = torch.where(c == all_labels)[0]
            class_features = all_features[idx]
            class_prototype = class_features.mean(dim=0)
            prototypes.append(class_prototype)
        prototypes = torch.stack(prototypes, dim=0)
        prototypes = F.normalize(prototypes, dim=-1)
        return all_features, all_labels, prototypes

    @torch.no_grad()
    def inference_all_description_feature(self, class_names, gpt_path, class_index):
        description_embeddings = []
        mean_embeddings = []
        all_targets = []

        file = open(gpt_path, "r")
        GPT_prompt_dict = json.load(file)
        GPT_prompt_dict = {k.lower(): v for k, v in GPT_prompt_dict.items()}
        k = 0
        for single_key in class_names:
            single_class_prompts = GPT_prompt_dict[single_key.lower()]
            targets = torch.full((len(single_class_prompts),), class_index[k])

            k += 1
            x_tokenized = torch.cat([clip.tokenize(p) for p in single_class_prompts])
            with torch.no_grad():
                text_features = self.clip_model.encode_text(x_tokenized.cuda())
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_embeddings.append(text_features.mean(0).unsqueeze(0))
            description_embeddings.append(text_features)
            all_targets.append(targets)
        description_embeddings = torch.cat(description_embeddings, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mean_embeddings = torch.cat(mean_embeddings, dim=0)
        mean_embeddings = F.normalize(mean_embeddings, dim=-1)
        return description_embeddings, all_targets, mean_embeddings

    def build_task_statistics(self, class_names, loader,
                              class_index, calibrate_novel_vision_proto=False):

        def shrink_cov(cov, alpha1=1.0, alpha2=0.0):
            diag_mean = torch.mean(torch.diagonal(cov))
            off_diag = cov.clone()
            off_diag.fill_diagonal_(0.0)
            mask = off_diag != 0.0
            off_diag_mean = (off_diag * mask).sum() / mask.sum()
            iden = torch.eye(cov.shape[0]).to(cov.device)
            cov_ = cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))
            return cov_

        cls_begin_index = class_index[0]

        text_features, text_targets = self.inference_text_feature(class_names, self.template, class_index)

        description_features, description_targets, description_proto = \
            self.inference_all_description_feature(class_names=class_names,
                                                   gpt_path=self.cfg.DATASET.GPT_PATH,
                                                   class_index=class_index)
        images_features, images_targets, images_proto = \
            self.inference_all_img_feature(loader, cls_begin_index)

        def per_filter(description_features, images_proto, images_features, class_index):
            np_description = description_features.cpu().numpy()
            np_images = images_proto.cpu().numpy()
            description_proto = []
            m = self.cfg.DATASET.M
            for cls in range(len(class_index)):
                cls_description = np_description[cls * m:cls * m + m]
                per_list = []
                k = m // 2
                cls_img = np_images[cls]
                for des in cls_description:
                    des = des.reshape(1, -1)
                    des_mean = des.mean(axis=1, keepdims=True)
                    img_mean = cls_img.mean()

                    des_centered = des - des_mean
                    img_centered = cls_img - img_mean
                    cov = np.mean(des_centered * img_centered[None, :], axis=1)
                    std_des = np.std(des, axis=1)
                    std_img = np.std(cls_img)
                    std_des = np.maximum(std_des, 1e-12)
                    std_img = max(std_img, 1e-12)
                    pearson = cov / (std_des * std_img)
                    per_list.extend(pearson)

                topk_indices = np.argsort(-np.array(per_list))[:k]

                tok_mean = []
                for indice in topk_indices:
                    tok_des = cls_description[indice]
                    tok_des = np.array(tok_des)
                    tok_des = torch.from_numpy(tok_des)
                    tok_mean.append(tok_des.unsqueeze(0))

                tok_mean = torch.cat(tok_mean, dim=0)
                tok_mean = tok_mean.mean(0)

                description_proto.append(tok_mean.unsqueeze(0))
            description_proto = torch.cat(description_proto, dim=0).to("cuda:0")
            description_proto = F.normalize(description_proto, dim=-1)
            print('description_proto shape:', description_proto.shape)
            return description_proto

        description_proto = per_filter(description_features, images_proto, images_features, class_index)

        yuan_proto = images_proto.clone()

        sim_description = F.softmax(images_proto @ description_proto.t(), dim=-1)
        weights = torch.diag(sim_description).unsqueeze(1)
        weights = torch.where(weights > 0, weights * self.cfg.DATASET.BETA, weights)

        images_proto = images_proto * (1 - weights) + (description_proto + text_features) / 2 * weights

        cov_images = torch.cov(images_features.T)
        cov_images = shrink_cov(cov_images, alpha1=1.0)

        return {
            'description_proto': description_proto,
            'description_features': description_features,
            'description_targets': description_targets,

            'text_features': text_features,
            'text_targets': text_targets,

            'yuan_proto': yuan_proto,
            'image_proto': images_proto,
            'images_features': images_features,
            'images_targets': images_targets,
            'cov_image': cov_images,

            'class_index': class_index,
            'sample_cnt': len(images_features)
        }

    def forward_ours(self, images, num_cls, num_base_cls,
                     image_proto, cov_image,
                     description_proto,
                     description_features, description_targets,
                     image_features,
                     text_features,
                     beta):

        def _mahalanobis(dist, cov_inv):
            left_term = torch.matmul(dist, cov_inv)
            mahal = torch.matmul(left_term, dist.T)
            return torch.diag(mahal)

        def _cov_forward(feat, proto, cov):
            maha_dist = []
            inv_covmat = torch.pinverse(cov.to(dtype=torch.float32))
            inv_covmat = inv_covmat.to(dtype=proto.dtype)

            for cl in range(num_cls):
                distance = feat - proto[cl]
                dist = _mahalanobis(distance, inv_covmat)
                maha_dist.append(dist)
            maha_dist = torch.stack(maha_dist)
            logits = -maha_dist.T
            return logits

        img_feat = self.extract_img_feature(images)
        img_feat = F.normalize(img_feat, dim=-1)

        logits_ma = _cov_forward(img_feat, image_proto, cov_image)

        prob_ma = F.softmax(logits_ma / 512, dim=-1)

        cnn_logits = self.clip_model.forward_img(images)["logits"]

        return cnn_logits, prob_ma

    @torch.no_grad()
    def extract_img_feature(self, images):
        images = images.to(self.device)
        image_features = self.clip_model.encode_image(images)
        return image_features

    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets
