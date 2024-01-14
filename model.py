import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from transformers import BertModel
from torch_geometric.nn import GCNConv, GATConv
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from dataset_processing import build_graph
import matplotlib.pyplot as plt
import numpy as np
from configs import Config
cfg = Config()

from sklearn.preprocessing import MinMaxScaler

# class SelfAttention(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(SelfAttention, self).__init__()
#         self.W = nn.Linear(input_size, hidden_size, True)
#         self.u = nn.Linear(hidden_size, 1)

#     def forward(self, x): 
#         u = torch.tanh(self.W(x))
#         a = nn.softmax(self.u(u), dim=1)
#         x = a.mul(x)
#         x = x.sum(1)
#         return x

class MyResNet50(nn.Module):
    def __init__(self, cfg):
        super(MyResNet50, self).__init__()
        self.resnet50_pret = models.resnet50(pretrained=True)
        self.resnet50_pret.fc = nn.Linear(2048,768).to(cfg.device)
    
    def forward(self, img_id):
        # construct resnet50_pretrain model
        transform1 = transforms.Compose([  # 串联多个图片变换的操作
        transforms.Resize(256),  # 缩放
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor()])  # 转换成Tensor
        try:
            img = Image.open(f"{cfg.img_path}/{img_id}.jpg" )  # 打开图片
            # print(f"{cfg.img_path}/{filename}.jpg")

            if img.mode != "RGB":
                img = img.convert("RGB")
            img1 = transform1(img).to(cfg.device)  # 对图片进行transform1的各种操作
            resnet50_feature_extractor = self.resnet50_pret  # ResNet50的预训练模型
            # resnet50_feature_extractor.fc = nn.Linear(2048,768).to(cfg.device) # 全连接层的输出设为1024
            torch.nn.init.eye_(resnet50_feature_extractor.fc.weight)  # 将二维tensor初始化为单位矩阵

            for name, param in resnet50_feature_extractor.named_parameters():
                    param.requires_grad = False
            # x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=True)
            x = torch.unsqueeze(img1, dim=0).float()
            # print(x)

            y = resnet50_feature_extractor(x) # 图片特征向量
            return y
        except:
            return torch.zeros((1,cfg.input_dim)).to(cfg.device)

class BertPre1(nn.Module):
    def __init__(self, cfg):
        super(BertPre1, self).__init__()
        self.bert_pret1 = BertModel.from_pretrained(cfg.bert_model)
        unfreeze_layers = ['layer.10','layer.11', 'bert.pooler', 'out.']
        # frozen bert layersq
        for name, param in self.bert_pret1.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        
    def forward(self, sent):
        seman_vecs = self.bert_pret1(sent)
        seman_vecs = seman_vecs.pooler_output
        return seman_vecs


class BertPre2(nn.Module):
    def __init__(self, cfg):
        super(BertPre2, self).__init__()
        self.bert_pret2 = BertModel.from_pretrained(cfg.bert_model)
        unfreeze_layers = ['layer.10','layer.11', 'bert.pooler', 'out.']
        # frozen bert layersq
        for name, param in self.bert_pret2.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
                
    def forward(self, sent):
        seman_vecs = self.bert_pret2(sent)
        seman_vecs = seman_vecs.pooler_output
        return seman_vecs


class EdgeWeightCal1(nn.Module):
    def __init__(self, cfg):
        super(EdgeWeightCal1, self).__init__()
        self.Wl_weight = nn.Parameter(torch.rand(cfg.input_dim,cfg.input_dim))
        self.Wn_weight = nn.Parameter(torch.rand(cfg.input_dim,cfg.input_dim))
        self.Ws_weight = nn.Parameter(torch.rand(cfg.input_dim,cfg.input_dim))
        # self.hs_weight = nn.Parameter(torch.rand(1,cfg.output_dim))
        
                
    def forward(self, node_feature, sent_feature):
        node_feature = node_feature.transpose(0,1)    # (output_dim*2)*N
        sent_feature = sent_feature.transpose(0,1)    # output_dim*N

        F_matrix = torch.tanh(node_feature.transpose(0,1).matmul(self.Wl_weight.matmul(sent_feature)))   # N*N
        H_s = torch.tanh(self.Wn_weight.matmul(node_feature) + self.Ws_weight.matmul(sent_feature).matmul(F_matrix))   # input_dim*N

        H_s = H_s.transpose(0,1)
        node_feature = node_feature.transpose(0,1)

        edge_weights = torch.cosine_similarity(node_feature.unsqueeze(1), H_s.unsqueeze(0), dim=2)
        edge_weights = torch.clamp(edge_weights, min=0.0).view(-1)
        edge_weights = edge_weights.clone().detach().requires_grad_(True).to(dtype=torch.float32, device=cfg.device)
        return edge_weights

class Sequential_Encoder(nn.Module):
    def __init__(self, cfg):
        super(Sequential_Encoder, self).__init__()
        # self.lstm = nn.LSTM(cfg.input_dim, cfg.output_dim*2, batch_first=False)
        self.mlp = nn.Sequential(nn.Linear(cfg.input_dim, cfg.output_dim*2), # 200 for not weighted
                        nn.Linear(cfg.output_dim*2,cfg.output_dim))
        
        # self.hs_weight = nn.Parameter(torch.rand(1,cfg.output_dim))
        
                
    def forward(self, seq_features):
        # seq_features, _ = self.lstm(seq_features)
        seq_features = self.mlp(seq_features)
        return seq_features


class MyModel(nn.Module):
    def __init__(self,cfg):
        super(MyModel,self).__init__()
        self.bert_pret1 = BertPre1(cfg)
        self.bert_pret2 = BertPre2(cfg)

        self.resnet50 = MyResNet50(cfg)

        self.edge_weight1 = EdgeWeightCal1(cfg)

        self.conv1 = GCNConv(cfg.input_dim, cfg.output_dim)
        self.MLP_bert = nn.Sequential(nn.Linear(cfg.input_dim, cfg.output_dim*2), # 200 for not weighted
                        nn.Linear(cfg.output_dim*2,cfg.output_dim))

        self.MLP_pred = nn.Sequential(nn.Linear(cfg.output_dim*2, cfg.hidden_dim), # 200 for not weighted
                                nn.Linear(cfg.hidden_dim,cfg.final_dim))
                                


        

    def forward(self, sent, nodes_num, ids, Flag):
        pred_list = []
        kl_list = []
        var_weights = []
        for i, sent_i in enumerate(sent):
            sent_i = sent_i[0:nodes_num[i]]
            nodes_vecs = self.bert_pret1(sent_i)
            # print(nodes_vecs)
            semantic_vecs = self.bert_pret2(sent_i)
            # semantic_vecs = semantic_vecs[0:nodes_num[i]]
            # print(semantic_vecs)

            img_vecs = self.resnet50(ids[i])
            # print(img_vecs)
            len_img_vecs = len(img_vecs)
            len_cat_vecs = nodes_num[i]+len_img_vecs
            # len_cat_vecs = nodes_num[i]
            if len_img_vecs != 0:
                nodes_vecs = torch.cat((nodes_vecs, img_vecs), dim=0)
                semantic_vecs = torch.cat((semantic_vecs, img_vecs), dim=0)

            text_graph = build_graph(len_cat_vecs, nodes_vecs).to(cfg.device)
            node_fea, edge_index = text_graph.x, text_graph.edge_index

            edge_weight1 = self.edge_weight1(node_fea, semantic_vecs)
            edge_weights = edge_weight1.reshape(len_cat_vecs,len_cat_vecs)[:-1,:-1]
            # data = edge_weight1.reshape(len_cat_vecs,len_cat_vecs).cpu()
            data = edge_weights.cpu()
            transfer=MinMaxScaler(feature_range=[0,1])
            data=transfer.fit_transform(data)
            conv1_graph = self.conv1(node_fea, edge_index, edge_weight1)
            # # print(conv2_graph[0])

            # # 句子嵌入区分度
            node_feas = conv1_graph[:-1,:-1]
            # data = torch.cosine_similarity(node_feas.unsqueeze(1)*100, node_feas.unsqueeze(0)*100, dim=2).cpu()
            # # print(data)
            # # data = data.abs()
            # # print(ids[i])
            temp_id = ids[i]
            # # print(data)
            fig, ax = plt.subplots()
            im = ax.imshow(data, cmap='pink',vmin=0,vmax=1)

            # 显示数值
            for i in range(len(data)):
                for j in range(len(data)):
                    text = ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='#33A02C', fontsize=6)
            # 设置坐标轴标签为偶数
            even_ticks = np.arange(0, len(data[0]), 2)
            # ax.set_xticks(even_ticks)
            # ax.set_yticks(even_ticks)
            plt.yticks(even_ticks, fontsize=18)
            plt.xticks(even_ticks, fontsize=18) #15 in fig1

            # 设置坐标轴标签字体大小
            # ax.set_xticklabels(even_ticks + 1, fontsize=8)
            # ax.set_yticklabels(even_ticks + 1, fontsize=8) #19
            # 设置colorbar字体大小
            cbar = plt.colorbar(im)
            cbar.ax.tick_params(labelsize=18)
            # plt.show()
            # plt.switch_backend('pdfcairo')
            if temp_id == "politifact87":
                plt.savefig('/home/yinjunwei/bert_processing_data/visual/politifact87_edge.pdf', dpi=300, bbox_inches='tight')
                # plt.savefig('/home/yinjunwei/bert_processing_data/visual/politifact87.pdf', dpi=300, bbox_inches='tight')
            # plt.savefig('/home/yinjunwei/bert_processing_data/visual/'+temp_id+'.pdf', dpi=300, bbox_inches='tight')
            # plt.savefig('/home/yinjunwei/MBO_we_1/heatmap_weight.png', dpi=300, bbox_inches='tight')

            # edge_weight1 = self.edge_weight1(node_fea, semantic_vecs)
            # print(ids[i])
            # print(len_cat_vecs)
            # print(edge_weight1.reshape(len_cat_vecs,len_cat_vecs).sum(1))

            # conv1_graph = self.conv1(node_fea, edge_index)
            # if temp_id == "politifact14469":
            #     break
            semantic_vecs = self.MLP_bert(semantic_vecs)
            # if Flag=="outer":
            #     kldiv = torch.nn.functional.kl_div(semantic_vecs.softmax(dim=-1).log(),conv1_graph.softmax(dim=-1),reduction="sum")
            #     if (kldiv > 999):
            #         kldiv = torch.tensor(999).to(cfg.device)
            #     kl_list.append(kldiv)


            graph_seman = torch.cat((semantic_vecs,conv1_graph),dim=1)

            pred = self.MLP_pred(graph_seman)
            
            pred = torch.mean(pred,dim=0)
            # print(pred)
            pred_list.append(pred)

        pred = torch.stack(pred_list, 0)
        if Flag=="outer":
            # kldiv = torch.stack(kl_list, 0)
            # return pred, kldiv
            return pred
        else:
            return pred