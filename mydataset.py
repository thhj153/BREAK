from torch.utils.data import Dataset, DataLoader
import numpy as np

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from transformers import data  #三维绘图

# from data_processing import gen_npy

# def load_data(cfg):
#     # 加载数据
#     # news_X, body_X, comment_X, img_X, img_ela_X, data_Y = gen_npy(cfg)
#     print('loss_fea:'+str(cfg.loss_Fea))
#     print("读取新闻标题:"+cfg.news_npy_path)
#     news_X = np.load(cfg.news_npy_path)
#     # print(np.asarray(news_X).shape)
#     # print(news_X.sum(1).sum(1).shape)
#     print("读取新闻正文"+cfg.body_npy_path)
#     body_X = np.load(cfg.body_npy_path)
#     # print(np.asarray(body_X).shape)

#     print("读取新闻评论"+cfg.comment_npy_path)
#     comment_X = np.load(cfg.comment_npy_path)
#     print("读取新闻图片"+cfg.img_ela_npy_path)
#     img_X = np.load(cfg.img_npy_path)
#     img_ela_X = np.load(cfg.img_ela_npy_path)
#     print("读取新闻标签"+cfg.label_npy_path)
#     data_Y = np.load(cfg.label_npy_path)
    
#     # print(np.asarray(body_X).shape)
#     # print(np.asarray(comment_X).shape)


#     # print(data_Y)

#     # news_X = news_X.sum(1).sum(1)
#     # # body_X = body_X.sum(1).sum(1)
#     # comment_X = comment_X.sum(1).sum(1)
#     # img_X = img_X.sum(1).sum(1)
#     # img_ela_X = img_ela_X.sum(1).sum(1)

#     # np.save('pca/0_1/title_0_1.npy',np.asarray(news_X))
#     # np.save('pca/0_1/label_0_1.npy',np.asarray(data_Y))

#     # # np.save('pca/0_1/body_0_1.npy',np.asarray(body_X))
#     # np.save('pca/0_1/comment_0_1.npy',np.asarray(comment_X))
#     # np.save('pca/0_1/img_0_1.npy',np.asarray(img_X))
#     # np.save('pca/0_1/img_ela_0_1.npy',np.asarray(img_ela_X))

#     # pca = PCA(n_components=3)
#     # pca = pca.fit(comment_X)
#     # news_X_dec = pca.transform(comment_X)
#     # print(news_X_dec.shape)

#     # fig = plt.figure()
#     # # plt.scatter(news_X_dec[data_Y==0,0],news_X_dec[data_Y==0,1],c='red',label=0)
#     # # plt.scatter(news_X_dec[data_Y==1,0],news_X_dec[data_Y==1,1],c='black',label=1)
#     # plt.legend()
#     # ax = Axes3D(fig)
#     # ax.scatter(news_X_dec[data_Y==0,0],news_X_dec[data_Y==0,1],news_X_dec[data_Y==0,2],c="red")
#     # ax.scatter(news_X_dec[data_Y==1,0],news_X_dec[data_Y==1,1],news_X_dec[data_Y==1,2],c="blue")
#     # ax.view_init(0,-30)

#     # plt.title("PCA of news titles")
#     # plt.show()
    
#     # plt.savefig("news_title_3d_4545")
#     # print(news_X_dec.shape)



#     # print(img_ela_X)

#     # 划分训练集、测试集、验证集
#     train_size = int(len(data_Y)*(1-cfg.percent_of_test-cfg.percent_of_val))
#     test_size = int(len(data_Y)*cfg.percent_of_test)

#     train_news_X = news_X[0:train_size]
#     test_news_X = news_X[train_size:train_size+test_size]
#     val_news_X = news_X[train_size+test_size:len(news_X)]

#     train_body_X = body_X[0:train_size]
#     test_body_X = body_X[train_size:train_size+test_size]
#     val_body_X = body_X[train_size+test_size:len(body_X)]

#     train_comment_X = comment_X[0:train_size]
#     test_comment_X = comment_X[train_size:train_size+test_size]
#     val_comment_X = comment_X[train_size+test_size:len(comment_X)]

#     train_img_X = img_X[0:train_size]
#     test_img_X = img_X[train_size:train_size+test_size]
#     val_img_X = img_X[train_size+test_size:len(img_X)]

#     train_img_ela_X = img_ela_X[0:train_size]
#     test_img_ela_X = img_ela_X[train_size:train_size+test_size]
#     val_img_ela_X = img_ela_X[train_size+test_size:len(img_ela_X)]

#     train_Y = data_Y[0:train_size]
#     test_Y = data_Y[train_size:train_size+test_size]
#     val_Y = data_Y[train_size+test_size:len(data_Y)]

#     train_data = list(zip(train_Y,train_news_X,train_body_X,train_comment_X,train_img_X,train_img_ela_X))
#     test_data = list(zip(test_Y,test_news_X,test_body_X,test_comment_X,test_img_X,test_img_ela_X))
#     val_data = list(zip(val_Y,val_news_X,val_body_X,val_comment_X,val_img_X,val_img_ela_X))

#     # train_data = list(zip(train_Y,train_news_X,train_comment_X,train_img_X,train_img_ela_X))
#     # test_data = list(zip(test_Y,test_news_X,test_comment_X,test_img_X,test_img_ela_X))
#     # val_data = list(zip(val_Y,val_news_X,val_comment_X,val_img_X,val_img_ela_X))

#     # train_data = list(zip(train_Y,train_news_X,train_comment_X,train_img_X))
#     # test_data = list(zip(test_Y,test_news_X,test_comment_X,test_img_X))
#     # val_data = list(zip(val_Y,val_news_X,val_comment_X,val_img_X))

#     # print("max of sents is:{}, max of words is:{}".format(max_sents,max_words))
#     # print("max of comment sents is:{}, max of comment words is:{}".format(comment_max_sents,comment_max_words))
#     max_sents = 9
#     comment_max_sents = 15 # 这里的这两个怎么设置处理的
#     print("*" * 80)
#     return train_data, test_data, val_data, max_sents, comment_max_sents
    # ---------------------------------------------数据-------------------------------------------------------

class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        assert item < len(self.data)
        data = self.data[item]
        title = data[0]
        label = data[1]
        nodes = data[2]
        ids = data[3]

        # return title, assignee, abstract, labels
        return title, label, nodes, ids





