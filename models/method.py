import torch
import torch.nn as nn

class SelfTuning(nn.Module):
    """
    Build a Self-Tuning model with: a query encoder, a key encoder, and a queue list
    """
    def __init__(self, network, backbone, queue_size=32, projector_dim=256, feature_dim=256,
                       class_num=200, momentum=0.999, temp=0.07, pretrained=True, pretrained_path=None):
        """
        network: the network of the backbone
        backbone: the name of the backbone
        queue_size: the queue size for each class
        projector_dim: the dimension of the projector (default: 1024)
        feature_dim: the dimension of the output from the backbone
        class_num: the class number of the dataset
        pretrained: loading from pre-trained model or not (default: True)
        momentum: the momentum hyperparameter for moving average to update key encoder (default: 0.999)
        temp: softmax temperature (default: 0.07)
        pretrained_path: the path of the pre-trained model
        """
        super(SelfTuning, self).__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.class_num = class_num
        self.backbone = backbone
        self.pretrained = pretrained
        self.temp = temp
        self.pretrained_path = pretrained_path

        # create the encoders
        if 'efficientnet' in self.backbone:
            self.encoder_q = network(backbone=self.backbone, feature_dim=feature_dim, projector_dim=projector_dim)
            self.encoder_k = network(backbone=self.backbone, feature_dim=feature_dim, projector_dim=projector_dim)
        else:
            self.encoder_q = network(projector_dim=projector_dim)
            self.encoder_k = network(projector_dim=projector_dim)

        if backbone == 'MOCOv2':  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        self.load_pretrained(network)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # don't be updated by gradient

        # create the queue
        self.register_buffer("queue_list", torch.randn(projector_dim, queue_size * self.class_num))
        self.queue_list = nn.functional.normalize(self.queue_list, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(self.class_num, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, key_c, c):
        # gather keys before updating queue
        batch_size = key_c.shape[0]
        ptr = int(self.queue_ptr[c])
        real_ptr = ptr + c * self.queue_size
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_list[:, real_ptr:real_ptr + batch_size] = key_c.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[c] = ptr

    def forward(self, im_q, im_k, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        batch_size = im_q.size(0)

        # compute query features
        q_c, q_f = self.encoder_q(im_q)  # queries: q_c (N x projector_dim)
        q_c = nn.functional.normalize(q_c, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k_c, _ = self.encoder_k(im_k)  # keys: k_c (N x projector_dim)
            k_c = nn.functional.normalize(k_c, dim=1)

        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nl,nl->n', [q_c, k_c]).unsqueeze(-1)  # Einstein sum is more intuitive

        # cur_queue_list: queue_size * class_num
        cur_queue_list = self.queue_list.clone().detach()

        l_neg_list = torch.Tensor([]).cuda()
        l_pos_list = torch.Tensor([]).cuda()

        for i in range(batch_size):
            neg_sample = torch.cat([cur_queue_list[:,0:labels[i]*self.queue_size],
                                    cur_queue_list[:,(labels[i]+1)*self.queue_size:]],
                                   dim=1)
            pos_sample = cur_queue_list[:, labels[i]*self.queue_size: (labels[i]+1)*self.queue_size]
            ith_neg = torch.einsum('nl,lk->nk', [q_c[i:i+1], neg_sample])
            ith_pos = torch.einsum('nl,lk->nk', [q_c[i:i+1], pos_sample])
            l_neg_list = torch.cat((l_neg_list, ith_neg), dim = 0)
            l_pos_list = torch.cat((l_pos_list, ith_pos), dim = 0)
            self._dequeue_and_enqueue(k_c[i:i+1], labels[i])

        # logits: 1 + queue_size + queue_size * (class_num - 1)
        PGC_logits = torch.cat([l_pos, l_pos_list, l_neg_list], dim=1)
        # apply temperature
        PGC_logits = nn.LogSoftmax(dim=1)(PGC_logits / self.temp)

        PGC_labels = torch.zeros([batch_size, 1 + self.queue_size*self.class_num]).cuda()
        PGC_labels[:,0:self.queue_size+1].fill_(1.0/(self.queue_size+1))
        return PGC_logits, PGC_labels, q_f

    def load_pretrained(self, network):
        if self.backbone == 'MOCOv1' and self.pretrained:
            if self.pretrained_path is None:
                self.pretrained_path = "~/.torch/models/moco_v1_200ep_pretrain.pth.tar"
            ckpt = torch.load(self.pretrained_path)['state_dict']
            state_dict_cut = {}
            for k, v in ckpt.items():
                if not k.startswith("module.encoder_q."):
                    continue
                k = k.replace("module.encoder_q.", "")
                state_dict_cut[k] = v
            self.encoder_q.load_state_dict(state_dict_cut)
            print('Successfully load the pre-trained model of MOCOv1')
        elif self.backbone == 'MOCOv2' and self.pretrained:
            if self.pretrained_path is None:
                self.pretrained_path = '~/.torch/models/moco_v2_800ep_pretrain.pth.tar'
            ckpt = torch.load(self.pretrained_path)['state_dict']
            state_dict_cut = {}
            for k, v in ckpt.items():
                if not k.startswith("module.encoder_q."):
                    continue
                if 'fc.2' in k:
                    continue
                k = k.replace("module.encoder_q.", "")
                state_dict_cut[k] = v
            self.encoder_q.load_state_dict(state_dict_cut, strict=False)
            print('Successfully load the pre-trained model of MOCOv2')
        elif 'resnet' in self.backbone:
            q = network(projector_dim=1000, pretrained=self.pretrained)
            q.fc = self.encoder_q.fc
            self.encoder_q = q
        elif 'densenet' in self.backbone:
            q = network(projector_dim=1000, pretrained=self.pretrained)
            q.classifier = self.encoder_q.classifier
            self.encoder_q = q

    def inference(self, img):
        y, feat = self.encoder_q(img)
        return feat


