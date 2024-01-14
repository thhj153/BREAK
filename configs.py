import torch

class Config():
    def __init__(self):
        # some hyperparameters
        self.limit_num_sen = 25
        self.limit_num_words = 13
        self.lr = 0.00001 # 0.00001 for weighted, 0.0001 for not weighted
        self.lr_edge = 0.10 # 0.00001 for weighted, 0.0001 for not weighted
        self.Seed = 1998
        self.weight_decay = 0.0005
        self.hidden_dim_tsne = 100 # dimension of tsne
        self.hidden_dim = 100  # dimension of MLP
        self.nodes_num = 5
        self.nodes_feature = 5
        self.input_dim = 768
        self.output_dim = 128 # 128 for weighted
        self.final_dim = 2
        self.test_perc = 0.1 # 0.8
        self.val_perc =  0.11 # 0.5
        self.alpha = 0.1

        self.epoch = 100
        self.batch_size = 8 

        # set devices and pre_trained model
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.bert_model = '/usr/gao/yinjunwei/bert-base-uncased'

        # data path
        self.dataset_name = "politic"
        self.sr_dataset_name = "politic"
        self.dataset_path = "/usr/gao/yinjunwei/MBO_data/"+ self.dataset_name + "/" + self.dataset_name +"_news.tsv"
        self.img_path = "/usr/gao/yinjunwei/MBO_data/"+ self.dataset_name + "/imgs"
        self.model_path = "/usr/gao/yinjunwei/MBO_data/"+ self.dataset_name + "/" + self.dataset_name +"_checkpoint_op.pt" # politic: checkpoint_op, gossip: checkpoint_op2
        self.source_model_path = "/usr/gao/yinjunwei/MBO_data/"+ self.sr_dataset_name + "/" + self.sr_dataset_name +"_checkpoint_variant.pt" # politic: checkpoint_op, gossip: checkpoint_op2

        self.news_list = "/usr/gao/yinjunwei/MBO_data/"+ self.dataset_name + "/" + self.dataset_name +"_news.npy"
        self.news_label = "/usr/gao/yinjunwei/MBO_data/"+ self.dataset_name + "/" + self.dataset_name +"_label.npy"
        self.news_id = "/usr/gao/yinjunwei/MBO_data/"+ self.dataset_name + "/" + self.dataset_name +"_id.npy"
        self.nodes_num_data = "/usr/gao/yinjunwei/MBO_data/"+ self.dataset_name + "/" + self.dataset_name +"_node.npy"

        self.test_id = "/usr/gao/yinjunwei/MBO_data/"+ self.dataset_name + "/" + self.dataset_name +"_test_id.npy"
        self.train_id = "/usr/gao/yinjunwei/MBO_data/"+ self.dataset_name + "/" + self.dataset_name +"_train_id.npy"
        self.val_id = "/usr/gao/yinjunwei/MBO_data/"+ self.dataset_name + "/" + self.dataset_name +"_val_id.npy"
