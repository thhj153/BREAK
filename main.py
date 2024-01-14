from torch.nn import parameter
from configs import Config
from model_2 import MyModel
from dataset_processing import *
from mydataset import MyDataSet
from earlystopping import EarlyStopping

from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import DataLoader, RandomSampler, dataloader

import numpy as np
import re
import os
import torch
import time
import argparse


parser = argparse.ArgumentParser(description="code")
parser.add_argument('--dataset_name', '-dn', type=str, default=-1, help='dataset name')
parser.add_argument('--Epoch', '-epoch', type=int, default=100, help='epoch number')
parser.add_argument('--hyperpara', '-alpha', type=float, default=-1, help='hypterparamter alpha')
args = parser.parse_args()


# Random seed. 随机数种子
def set_seed(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Metircs
def metric_new(pred, label):
    p = metrics.precision_score(label, pred, zero_division=0,average='weighted')
    r = metrics.recall_score(label, pred,average='weighted')
    F1 = metrics.f1_score(label, pred,average='weighted')
    acc = metrics.accuracy_score(label, pred)
    # fpr, tpr, thresholds = metrics.roc_curve(label, pred_old.detach().numpy()[:,1], pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # if(mode=='test'):
    #     print(metrics.classification_report(label, pred, digits=4))
    # return p, r, F1, acc, auc
    return p, r, F1, acc

def cut_sent_c(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def cut_sent_chinese(data):
    data_sen = []
    for new in data:
        tp_new = cut_sent_c(new)
        data_sen.append(tp_new)
    return data_sen


if __name__ == "__main__":
    cfg = Config() # Instantiate the config class

    set_seed(cfg.Seed)
    # Setting the inputs
    if args.dataset_name != -1:
        cfg.dataset_name = args.dataset_name
        cfg.dataset_path = "/usr/gao/yinjunwei/MBO_data/"+ cfg.dataset_name + "/" + cfg.dataset_name +"_news.tsv"
        cfg.img_path = "/usr/gao/yinjunwei/MBO_data/"+ cfg.dataset_name + "/imgs"
        cfg.model_path = "/usr/gao/yinjunwei/MBO_data/"+ cfg.dataset_name + "/" + cfg.dataset_name +"_checkpoint.pt"
        cfg.news_list = "/usr/gao/yinjunwei/MBO_data/"+ cfg.dataset_name + "/" + cfg.dataset_name +"_news.npy"
        cfg.news_label = "/usr/gao/yinjunwei/MBO_data/"+ cfg.dataset_name + "/" + cfg.dataset_name +"_label.npy"
        cfg.news_id = "/usr/gao/yinjunwei/MBO_data/"+ cfg.dataset_name + "/" + cfg.dataset_name +"_id.npy"
        cfg.nodes_num_data = "/usr/gao/yinjunwei/MBO_data/"+ cfg.dataset_name + "/" + cfg.dataset_name +"_node.npy"

    if args.hyperpara != -1:
        cfg.alpha = args.hyperpara
    
    if args.Epoch != 100:
        cfg.num_epoch = args.Epoch
    # for i in cfg:
    print(vars(cfg))

    print("*" * 80)  
    # Setting random seed
    torch.manual_seed(cfg.Seed)
    torch.cuda.manual_seed_all(cfg.Seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.Seed)
    print("*" * 80)

    print("*" * 80)  
    print("Constructing Model...")
    TestModel = MyModel(cfg)
    # for name, param in TestModel.named_parameters():
    #     print(name, param.shape)
    # print("parameters1",TestModel.bert_pret1)
    # print("parameters2",TestModel.resnet50)
    print("device:",cfg.device)
    TestModel = TestModel.to(cfg.device)
    # TestModel.load_state_dict(torch.load(cfg.source_model_path)) # 不用，为0.811，用了，为0.813
    early_stopping = EarlyStopping()
    print("*" * 80)

    print("*" * 80)  
    print("Defining Loss Function...")
    loss_pred = torch.nn.CrossEntropyLoss()
    print("*" * 80)
    
    print("*" * 80)
    print("Taking Adam as the Ooptimizer...")
    lstm_params = []
    # for param in TestModel.MLP_bert.parameters():
    #     lstm_params.append(param)
    outer_optimizer = torch.optim.Adam([
        {'params': TestModel.bert_pret1.parameters()},
        {'params': TestModel.bert_pret2.parameters()},
        {'params': TestModel.resnet50.parameters()},
        {'params': TestModel.conv1.parameters()}, #111
        {'params': TestModel.MLP_bert.parameters()}, # MLP
        {'params': TestModel.MLP_pred.parameters()}], cfg.lr, weight_decay=cfg.weight_decay, eps=1e-6)
    # inner_optimizer = torch.optim.Adam([
    #     {'params': TestModel.edge_weight1.parameters()},
    #     ], cfg.lr_edge, weight_decay=cfg.weight_decay, eps=1e-6)
    print("*" * 80)

    # Some examples with four sentences and theirs labels
    # news_list = ["This is a test example, I want transfer it to a graph? This is the second sentences! It is being a long day, without you my friend. If you listen the whistle baloney, I am five hundred miles away from home."
    #             ,"This is the second sentences. It is being a long day, without you my friend. If you listen the whistle baloney, I am five hundred miles away from home.",]
    # labels = ["1","2","3"]

    # Get data from the dataset
    # news_tit_cont, news_label, news_id = get_data(cfg.dataset_path)


    # # Getting token ids and  the sentences (nodes) number 获得句子数量，即将要构建的节点数量, 将数据集从自然语言处理为token_id
    # print("convert_text_to_token......")
    # tokenizer = BertTokenizer.from_pretrained(cfg.bert_model) # Tokenizer
    # # news_list = [convert_text_to_token(tokenizer, tit, cfg.limit_num_sen, cfg.limit_num_words) for tit in tqdm(news_list)]
    # sents_list = []
    # nodes_num = []
    # for tit in tqdm(news_tit_cont):
    #     sent, node_num = convert_text_to_token(tokenizer, tit, cfg.limit_num_sen, cfg.limit_num_words)
    #     sents_list.append(sent)
    #     nodes_num.append(node_num)
    
    # # # Save the tokens 存储转换后的token，避免每次重复计算
    # # np.save(cfg.news_list,sents_list)
    # # np.save(cfg.news_label,news_label)
    # # np.save(cfg.nodes_num_data,nodes_num)
    # # np.save(cfg.news_id,news_id)

    sents_list = np.load(cfg.news_list,allow_pickle=True)
    news_label = np.load(cfg.news_label,allow_pickle=True)
    news_id = np.load(cfg.news_id,allow_pickle=True)
    nodes_num = np.load(cfg.nodes_num_data,allow_pickle=True)

    train_val_news_list, test_news_list, train_val_news_label, test_news_label, train_val_news_node, test_news_node, train_val_news_id, test_news_id \
        = train_test_split(sents_list, news_label, nodes_num, news_id, random_state=cfg.Seed, test_size=cfg.test_perc)

    train_news_list, val_news_list, train_news_label, val_news_label, train_news_node, val_news_node, train_news_id, val_news_id \
        = train_test_split(train_val_news_list, train_val_news_label, train_val_news_node, train_val_news_id, random_state=cfg.Seed, test_size=cfg.val_perc)
    # np.save(cfg.train_id, train_news_id)
    # np.save(cfg.test_id, test_news_id)
    # train_val_news_list, test_news_list, train_val_news_label, test_news_label, train_val_news_node, test_news_node \
    #     = train_test_split(sents_list, news_label, nodes_num, random_state=cfg.Seed, test_size=cfg.test_perc)

    # train_news_list, val_news_list, train_news_label, val_news_label, train_news_node, val_news_node \
    #     = train_test_split(train_val_news_list, train_val_news_label, train_val_news_node, random_state=cfg.Seed, test_size=cfg.val_perc)
    
    print(f"the length of train_news_list is: {len(train_news_list)}")
    print(f"the length of val_news_list is: {len(val_news_list)}")
    print(f"the length of test_news_list is: {len(test_news_list)}")

    # , train_news_id
    train_news = MyDataSet(list(zip(train_news_list,train_news_label, train_news_node, train_news_id)))
    val_news = MyDataSet(list(zip(val_news_list,val_news_label, val_news_node, val_news_id)))
    test_news = MyDataSet(list(zip(test_news_list,test_news_label, test_news_node, test_news_id)))
    # 
    
    # train_news = np.load(cfg.train_data)
    # val_news = np.load(cfg.val_data)
    # test_news = np.load(cfg.test_data)

    train_sampler = RandomSampler(train_news)
    val_sampler = RandomSampler(val_news)
    test_sampler = RandomSampler(test_news)
    
    train_dataloader = DataLoader(train_news, sampler=train_sampler,batch_size=cfg.batch_size)
    val_dataloader = DataLoader(val_news, sampler=val_sampler,batch_size=cfg.batch_size)
    test_dataloader = DataLoader(test_news, sampler=test_sampler,batch_size=cfg.batch_size)

    draw_train_loss, draw_val_loss = [], []
    min_val_loss = 10.0

    torch.set_num_threads(32)
    start_time = time.time()
    # for name, param in TestModel.named_parameters():
    #     param.requires_grad = False
    # for epoch in range(cfg.epoch):
    #     print("epoch:{}".format(epoch + 1))
    #     TestModel.train()
    #     epoch_loss_train = 0.0
        
    #     train_y_epoch = []
    #     start_time_epoch = time.time()
        
    #     for i, batch in enumerate(train_dataloader):
    #         train_news = batch[0].long().to(cfg.device)
    #         train_label = batch[1].long().to(cfg.device)
    #         train_node = batch[2].long().to(cfg.device)
    #         train_id = batch[3]
    #         train_y_epoch.append(train_label)
            
    #         # inner_optimizer.zero_grad()
    #         # pred_res = TestModel.forward(train_news, train_node, train_id, Flag="inner")
    #         # # pred_res = TestModel.forward(train_news, train_node)

    #         # batch_loss_train = loss_pred(pred_res, train_label)
    #         # # print(batch_loss_train)
    #         # epoch_loss_train += train_label.size()[0] * batch_loss_train.item()
    #         # # print(epoch_loss_train)
    #         # batch_loss_train.backward()
    #         # inner_optimizer.step()


    #         outer_optimizer.zero_grad()
    #         # pred_res, kldiv = TestModel.forward(train_news, train_node, train_id, Flag="outer")
    #         pred_res = TestModel.forward(train_news, train_node, train_id, Flag="outer")
    #         # pred_res = TestModel.forward(train_news, train_node)
    #         # kldiv = torch.mean(kldiv)
    #         # print(kldiv)

    #         # batch_loss_train = loss_pred(pred_res, train_label) + cfg.alpha*kldiv
    #         batch_loss_train = loss_pred(pred_res, train_label)
    #         # print(batch_loss_train)
    #         # print(loss_pred(pred_res, train_label))
    #         epoch_loss_train += train_label.size()[0] * batch_loss_train.item()
    #         # print(epoch_loss_train)
    #         batch_loss_train.backward()
    #         outer_optimizer.step()
    #     # print(epoch_loss_train)
        
    #     epoch_loss_train /= len(train_y_epoch)
    #     draw_train_loss.append(epoch_loss_train)
    #     print(f"epoch:{epoch+1}, train_loss: {epoch_loss_train:.4f}......")

    #     with torch.no_grad():
    #         TestModel.eval()
    #         epoch_loss_val = 0.0
    #         pred_epoch, pred_one_epoch, val_y_epoch = [], [], []

    #         for i, batch in enumerate(val_dataloader):
    #             # node_num = nodes_num[i]

    #             val_news = batch[0].long().to(cfg.device)
    #             val_label = batch[1].long().to(cfg.device)
    #             val_node = batch[2].long().to(cfg.device)
    #             val_id = batch[3]
    #             pred_res = TestModel.forward(val_news, val_node, val_id, Flag="outer")
    #             # pred_res = TestModel.forward(val_news, val_node)
    #             # kldiv = torch.mean(kldiv)


    #             batch_loss_val = loss_pred(pred_res, val_label)

    #             epoch_loss_val += val_label.size()[0]*batch_loss_val.item()

    #             pred_one = torch.argmax(pred_res, dim=1)
    #             # pred_epoch.append(pred_res)
    #             pred_one_epoch.append(pred_one)
    #             val_y_epoch.append(val_label)

    #         pred_one = torch.cat([i for i in pred_one_epoch],0).cpu()
    #         # pred = torch.cat([i for i in pred_epoch,0]).cpu()
    #         val_label = torch.cat([i for i in val_y_epoch],0).cpu()

    #         Pre, Rec, F1, Acc = metric_new(pred_one, val_label)
    #         epoch_loss_val /= len(val_label)
    #         draw_val_loss.append(epoch_loss_val)
    #         print(f"********epoch:{epoch+1}, val_loss:{epoch_loss_val:.4f},  Acc:{Acc:.3f}, Pre:{Pre:.3f}, Rec:{Rec:.3f}, F1:{F1:.3f}")

    #         # 早停
    #         early_stopping(epoch_loss_val, TestModel)
    #         if early_stopping.early_stop:
    #             print('Early stopping!')
    #             break
    #     print(f"训练一个epoch耗时：{(time.time()-start_time_epoch)}s")
    #     print("*" * 80)
    print("*" * 80)
    print(f"训练总耗时：{(time.time()-start_time)/60}mins, 开始测试模型性能...")
    # 加载模型
    
    Model = MyModel(cfg).to(cfg.device)
    
    Model.load_state_dict(torch.load(cfg.model_path))

    pred_epoch, pred_one_batch, test_y_batch, GRU_embed = [], [], [] ,[]

    with torch.no_grad():
        Model.eval()
        for i, batch in enumerate(test_dataloader):
            # node_num = nodes_num[i]

            test_news = batch[0].long().to(cfg.device)
            test_label = batch[1].long().to(cfg.device)
            test_node = batch[2].long().to(cfg.device)
            test_id = batch[3]

            pred_res = Model.forward(test_news, test_node, test_id, Flag="inner")
            # pred_res = Model.forward(test_news, test_node)
            
            # pred_epoch.append(pred_all)
            # GRU_embed.append(pred_GRU)  # MLP之前的嵌入

            pred_one = torch.argmax(pred_res, dim=1)

            pred_one_batch.append(pred_one)
            test_y_batch.append(test_label)

        pred_one = torch.cat([i for i in pred_one_batch],0).cpu()
        # pred = torch.cat([i for i in pred_epoch],0).cpu()
        test_label = torch.cat([i for i in test_y_batch],0).cpu()

        # GRU_embed = torch.cat([i for i in GRU_embed],0).cpu() # MLP之前的嵌入

        # np.save('T-SNE/politic_label_2.npy',np.asarray(test_y_data_epoch))
        # np.save('T-SNE/politic_embed_2.npy',np.asarray(GRU_embed))
        # print(np.asarray(test_y_data_epoch),np.asarray(GRU_embed).shape)

        Pre, Rec, F1, Acc = metric_new(pred_one, test_label)
        print(f"Acc:{Acc:.3f}, Pre:{Pre:.3f}, Rec:{Rec:.3f}, F1:{F1:.3f}")





