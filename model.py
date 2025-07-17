import torch
from torchvision.models import resnet50

from CLIP import clip


DEVICE= "cuda" if torch.cuda.is_available() else "cpu"

class CLIP_EXIF_Net(torch.nn.Module):
    def __init__(self, model_name, device, resume=None, resume_img=None, freezeText=False,
                 fixed_scale=False, fixed_temp=10):
        super(CLIP_EXIF_Net, self).__init__()
        clipNet, _, __, ___ = clip.load(model_name, device=device, jit=False)
        clipNet.float()
        self.model = clipNet
        del self.model.visual

        image_encoder = resnet50(pretrained=True)
        image_encoder.fc = torch.nn.Linear(2048, 768)
        self.model.visual = image_encoder
        if resume_img is not None:
            if device == 'cpu':
                checkpoint = torch.load(resume_img, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(resume_img)

            # 定义要加载的具有 "visual" 前缀的键
            keys_to_load = [key for key in checkpoint['model'].keys() if 'visual' in key]
            # 创建一个新的 state_dict，只包含具有 "visual" 前缀的键
            state_dict_filtered = {key: value for key, value in checkpoint['model'].items() if
                                   any(k in key for k in keys_to_load)}
            prefix = 'model.visual'
            new_state_dict = {key[len(prefix) + 1:]: value for key, value in state_dict_filtered.items() if
                              key.startswith(prefix)}
            msg = self.model.visual.load_state_dict(new_state_dict, strict=False)
            print(f"load pretraining model img {resume_img}")
            print(msg)
        self.model.text_fc = torch.nn.Linear(1024, 768)

        if fixed_scale:
            self.fixed_scale = fixed_scale
            self.logit_scale = fixed_temp # 1/0.1
        else:
            self.model.logit_scale = clipNet.logit_scale

        if freezeText:
            for p in self.model.token_embedding.parameters():
                p.requires_grad = False
            for p in self.model.transformer.parameters():
                p.requires_grad = False
            self.model.positional_embedding.requires_grad = False
            self.model.text_projection.requires_grad = False
            for p in self.model.ln_final.parameters():
                p.requires_grad = False

        if resume is not None:
            if device == 'cpu':
                checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(resume)
            msg = self.load_state_dict(checkpoint['model'], strict=False)
            print(f"load pretraining model from {resume}...")
            print(msg)

    @property
    def dtype(self):
        return self.model.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.model.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.model.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection

        return x

    def forward(self, image, text, return_fea=False):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        text_features = self.model.text_fc(text_features)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        if return_fea:
            return logits_per_image, logits_per_text, image_features
        else:
            return logits_per_image, logits_per_text



class BC_MLP(torch.nn.Module):
    def __init__(self, device, resume=None, feature_dim=768, num_classes=2):
        super(BC_MLP, self).__init__()

        clipNet = CLIP_EXIF_Net(model_name="RN50", device=device).to(device)
        self.model = clipNet
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.model.visual.parameters():
            p.requires_grad = False

        self.fc1 = torch.nn.Linear(feature_dim, feature_dim * 2).to(device)
        self.fc2 = torch.nn.Linear(feature_dim * 2, num_classes).to(device)

        if resume is not None:
            if device == 'cpu':
                checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(resume)
            msg = self.load_state_dict(checkpoint['model'], strict=False)
            print(f"load pretraining model from {resume}...")
            print(msg)

    def MLP(self, fea):
        x = torch.nn.functional.relu(self.fc1(fea))
        hidden_out = x
        x = self.fc2(x)
        return x, hidden_out

    def forward(self, x, return_fea=False):
        features = self.model.encode_image(x)
        features = features / features.norm(dim=1, keepdim=True)
        output, hidden_fea = self.MLP(features)
        if return_fea:
            return output, hidden_fea
        return output