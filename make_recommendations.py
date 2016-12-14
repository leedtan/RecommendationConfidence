import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
print (sys.version)
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import six
from sklearn.decomposition import PCA
import time
from sklearn.model_selection import KFold
#import tflearn
import seaborn as sns

import itertools
from collections import defaultdict

from sklearn.metrics import confusion_matrix

#datestr = "sample"
#datestr = "'2016-03-30'and'2016-04-30'"
productratings = pd.read_csv('data/ratings.csv', nrows = 1e6)
#productratings = pd.read_csv('data/productratings/' + str(datestr) + '.csv')
#feat_df = pd.read_csv('data/feat_df/' + str(datestr) + '.csv')
#df_u_by_y = pd.read_csv('data/df_u_by_y/' + str(datestr) + '.csv')
#featMat = pd.read_csv('data/featMat/' + str(datestr) + '.csv').as_matrix()

def map2idx(productratings):

    users = productratings['userId'].values
    products = productratings['movieId'].values

    # unique users / products
    uni_users = productratings['userId'].unique()
    uni_products = productratings['movieId'].unique()

    # dict mapping the id to an index
    user_map = dict(zip(uni_users, range(len(uni_users))))
    product_map = dict(zip(uni_products, range(len(uni_products))))
    inverse_user_map = dict(zip(range(len(uni_users)), uni_users))
    inverse_product_map = dict(zip(range(len(uni_products)), uni_products))

    pairs = []
    for user, product, rating in zip(users, products, productratings['rating']):
        if product in product_map:
            pairs.append((user_map[user], product_map[product], rating))

    return np.array(pairs), len(uni_users), len(uni_products), user_map, product_map, inverse_user_map, inverse_product_map

   


def split_all_train_test(users,products,ratings,split=.2):

    shuffle  = np.random.permutation(len(users))

    partition = np.floor(len(users) * (1-split))

    train_idx = shuffle[:partition]
    test_idx = shuffle[partition:]

    users_train = users[train_idx]
    users_test = users[test_idx]

    products_train = products[train_idx]
    products_test = products[test_idx]

    ratings_train = ratings[train_idx]
    ratings_test = ratings[test_idx]

    return users_train,products_train,ratings_train , users_test,products_test,ratings_test


class HybridCollabFilter():

    def __init__(self, numUsers, numProducts, input_product_dim = None, reg_l = 1, conf_dim = 1, conf_l = 1,
                 edim_latent_prod = 3, edim_user = 2, hidden_product_size = None):
                
        
        # hyper parameters
        self.batch_size = 512
        self.numUsers = numUsers
        self.numProducts = numProducts
        self.init_var =.01
        self.reg_l = reg_l
        self.edim_latent_prod = edim_latent_prod
        self.edim_user = edim_user
        self.hidden_product_size = hidden_product_size
        self.input_product_dim = input_product_dim
        self.conf_l = conf_l
        numSongs = numProducts
        
        # input tensors for products, usres, ratings
        self.users = tf.placeholder(tf.int32, shape=(None))
        self.products = tf.placeholder(tf.int32, shape=(None))
        self.rating = tf.placeholder(tf.float32, shape=(None))
        self.mu = tf.Variable(tf.random_normal([1]))
        # embedding matricies for users
        self.userMat = tf.Variable(self.init_var*tf.random_normal([numUsers, edim_user]))
        self.userBias = tf.Variable(self.init_var*tf.random_normal([numUsers,]))
  
        # embedding matrices for movies
        self.movieMat = tf.Variable(self.init_var*tf.random_normal([numProducts, edim_user]))
        self.movieBias = tf.Variable(self.init_var*tf.random_normal([numProducts,]))
        
        # map each user/product to its feature vector
        self.U = tf.abs(tf.nn.embedding_lookup(self.userMat, self.users))
        self.u_b = tf.abs(tf.nn.embedding_lookup(self.userBias, self.users))
        self.V = tf.abs(tf.nn.embedding_lookup(self.movieMat, self.products))
        self.v_b = tf.abs(tf.nn.embedding_lookup(self.movieBias, self.products))

        
        self.reg = self.reg_l * ( \
                    tf.reduce_mean(tf.square(self.U) ) + \
                    tf.reduce_mean(tf.square(self.u_b) ) + \
                    tf.reduce_mean(tf.square(self.V) ) + \
                    tf.reduce_mean(tf.square(self.v_b) ) + \
                    tf.reduce_mean(tf.square(self.mu) ) \
                                            )
        
        #final answer
        self.yhat = tf.reduce_sum(tf.mul(self.U, self.V) , 1) + self.u_b + self.v_b + self.mu
        
        self.error = tf.reduce_mean(tf.nn.l2_loss(self.yhat - self.rating))
        self.true_loss = self.error
        self.training_sse = self.error
        if conf_dim > 0:
            self.C_user = tf.Variable(.5*tf.ones([numUsers, conf_dim]))
            self.C_song = tf.Variable(.5*tf.ones([numSongs, conf_dim]))
            self.C_ui = tf.clip_by_value(tf.nn.embedding_lookup(self.C_user, self.users), 1e-20, 1)
            self.C_sj = tf.clip_by_value(tf.nn.embedding_lookup(self.C_song, self.products), 1e-20, 1)
            self.confidence_reg = self.conf_l * tf.reduce_sum(1/self.C_ui + 1/self.C_sj)
            self.reg = self.reg + self.confidence_reg
            self.error *= tf.reduce_mean(tf.reduce_sum(self.C_ui * self.C_sj, 1))
            
        self.cost = (self.error + self.reg)/1e7
        
        self.learning_rate = 1e-2

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())





    def train(self, users, products, ratings, 
              eval_type = 'MSE', val_freq=5, epochs = 5, verb = 0, add_noise = 1e-5):

        users_train, products_train, ratings_train, users_test, products_test, ratings_test = \
            split_all_train_test(users,products,ratings)
            
        self.num_train = products_train.shape[0]
        self.num_test = products_test.shape[0]
        self.num_batches = self.num_train // self.batch_size
        t = time.time()
        for i in range(epochs):
            self.adjust_from_quick_to_stable_training()
            avg_cost = 0
            for b_idx in range(self.num_batches):

                ratings_batch  = ratings_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                users_batch = users_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                product_ids = products_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                
                if b_idx == self.num_batches - 1:
                    ratings_batch  = ratings_train[self.batch_size * b_idx:]
                    users_batch = users_train[self.batch_size * b_idx:]
                    product_ids = products_train[self.batch_size * b_idx:]

                product_batch = product_ids
                
                if add_noise > 0:
                    ratings_batch = ratings_batch + np.random.randn(ratings_batch.shape[0]) * add_noise
                
                avg_cost +=  (self.session.run([self.true_loss, self.optimizer],
                                   {self.users: users_batch, 
                                    self.products: product_batch,
                                    self.rating: ratings_batch})[0] ) / self.num_train
            if (verb > 0):
                print ("Epoch: ", i, " Average Training Cost: ",avg_cost / self.num_batches, ' time = ', time.time() - t)
            
            if (i+1) % val_freq ==0 or i == epochs - 1:
                err = self.validate_model(users_test, products_test, ratings_test, eval_type, users_train = users_train)
            
        return err 
    
    def create_learning_curve(self, users, products, ratings,
                              eval_type = 'MSE', val_freq=5, epochs = 5, verb = 0, add_noise = 1e-5):

        users_train, products_train, ratings_train, users_test, products_test, ratings_test = \
            split_all_train_test(users,products,ratings)
            
        self.num_train = products_train.shape[0]
        self.num_test = products_test.shape[0]
        self.num_batches = self.num_train // self.batch_size
        t = time.time()
        mse_train = []
        mse_test = []
        for i in range(epochs):
            self.adjust_from_quick_to_stable_training()
            avg_cost = 0
            for b_idx in range(self.num_batches):

                ratings_batch  = ratings_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                users_batch = users_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                product_ids = products_train[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                
                if b_idx == self.num_batches - 1:
                    ratings_batch  = ratings_train[self.batch_size * b_idx:]
                    users_batch = users_train[self.batch_size * b_idx:]
                    product_ids = products_train[self.batch_size * b_idx:]

                product_batch = product_ids
                
                if add_noise > 0:
                    ratings_batch = ratings_batch + np.random.randn(ratings_batch.shape[0]) * add_noise
                
                avg_cost +=  (self.session.run([self.training_sse, self.optimizer],
                                   {self.users: users_batch, 
                                    self.products: product_batch,
                                    self.rating: ratings_batch})[0] ) / self.num_train
            if verb > 0:
                print ("Epoch: ", i, " Average Training Cost: ",avg_cost / self.num_batches, ' time = ', time.time() - t)
            
            
            test_err = self.validate_model(users_test, products_test, ratings_test, eval_type= 'MSE', users_train = users_train)
            
            mse_train += [avg_cost]
            mse_test += [test_err]
        self.validate_model(users_test, products_test, ratings_test, eval_type= 'err_bars', users_train = users_train)
        self.validate_model(users_test, products_test, ratings_test, eval_type= 'confusion', users_train = users_train)
        return (mse_train, mse_test) 
    
    def validate_model(self, users_test, products_test, ratings_test, eval_type, users_train= None):
        if eval_type == 'AUC':
            auc_mean = 0
            uni_users = np.unique(users_test)
            for usr in uni_users:
                usr_idxes = users_test == usr
                usr_idxes = np.where(usr_idxes)
                usr_u = users_test[usr_idxes]
                product_u = products_test[usr_idxes]
                rtg_u = ratings_test[usr_idxes]
                if len(usr_u) < 3:
                    continue
                yhat = (self.session.run([self.yhat],
                                         {self.users: usr_u, self.products: product_u,
                                          self.rating: rtg_u})[0] )
                auc_mean += sklearn.metrics.auc(yhat, rtg_u, reorder = True) / len(uni_users)

            print ("Testing AUC mean: " , auc_mean)
            err = auc_auc
            
        if eval_type == 'err_bars':
            err_mean = {}
            uni_users = np.unique(users_test)
            for usr in uni_users:
                usr_train_idxes = users_train == usr
                usr_train_idxes = np.where(usr_train_idxes)
                usr_train_u = users_train[usr_train_idxes]
                user_train_len = len(usr_train_u)
                position = np.log2(user_train_len)
                if position < 0:
                    continue
                usr_idxes = users_test == usr
                usr_idxes = np.where(usr_idxes)
                usr_u = users_test[usr_idxes]
                product_u = products_test[usr_idxes]
                rtg_u = ratings_test[usr_idxes]
                mse = (self.session.run([self.training_sse],
                                         {self.users: usr_u, 
                                          self.products: product_u,
                                          self.rating: rtg_u})[0] ) / len(usr_u)
                
                usr_position = int(np.floor(position))
                if not usr_position in err_mean.keys():
                    err_mean[usr_position] = []
                err_mean[usr_position] += [mse]
            errors = [err_mean[key] for key in sorted(err_mean)]
            means = [np.mean(errors[idx]) for idx in range(len(errors))]
            medians = [np.median(errors[idx]) for idx in range(len(errors))]
            stds = [np.std(errors[idx]) for idx in range(len(errors))]
            q25s = [np.percentile(errors[idx], 25) for idx in range(len(errors))]
            q75s = [np.percentile(errors[idx], 75) for idx in range(len(errors))]
            keys = [key for key in sorted(err_mean)]
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
            ax.set_xlim([-0.5, len(errors)+0.5])
            ax.set(xlabel='log2(number of ratings per user)', \
                   ylabel='IQR (or mse if i changed back) per user validation values', \
                   title = "Error Bars of user types")

            # Add std deviation bars to the previous plot
            ax.errorbar(keys, medians, yerr=[q25s, q75s], fmt='-o')
            plt.show()
            err = 449

        if eval_type == 'MSE':
            self.test_batch_size = np.min((10000, self.num_test))
            self.test_num_batches = self.num_test // self.test_batch_size
            mse = 0
            
            for b_idx in range(self.test_num_batches):
                mse += self.get_sse(ratings_test, users_test, products_test, b_idx) / self.num_test

            print ("Testing MSE: ", mse)
            err = mse

        if eval_type == 'confusion':
            
            self.test_batch_size = self.num_test#np.min((10000, self.num_test))
            self.test_num_batches = self.num_test // self.test_batch_size
            
            #yhat = []
            #for b_idx in range(self.test_num_batches):
            b_idx = 0
            yhat = self.get_yhat(users_test, products_test, b_idx)
            yhat_rounded = np.round(yhat*2)/2
            ratings_test_rounded = np.round(ratings_test * 2 ) / 2 
            
            draw_confusion(yhat_rounded, ratings_test_rounded)
            
            ### and do MSE
            self.test_batch_size = np.min((10000, self.num_test))
            self.test_num_batches = self.num_test // self.test_batch_size
            mse = 0
            
            for b_idx in range(self.test_num_batches):
                mse += self.get_sse(ratings_test, users_test, products_test, b_idx) / self.num_test

            print ("Testing MSE: ", mse)
            err = mse
        return err
                    
    def get_yhat(self, users_test, products_test, b_idx):
        
        users_batch = users_test[self.test_batch_size * b_idx:self.test_batch_size * (b_idx + 1)]
        product_ids = products_test[self.test_batch_size * b_idx:self.test_batch_size * (b_idx + 1)]

        if b_idx == self.test_num_batches - 1:
            users_batch = users_test[self.test_batch_size * b_idx:]
            product_ids = products_test[self.test_batch_size * b_idx:]
            
        product_batch = product_ids

        yhat = self.session.run(self.yhat,
                               {self.users: users_batch, 
                                self.products: product_batch})
        return yhat

    
    def get_sse(self, ratings_test, users_test, products_test, b_idx):
        
        ratings_batch  = ratings_test[self.test_batch_size * b_idx:self.test_batch_size * (b_idx + 1)]
        users_batch = users_test[self.test_batch_size * b_idx:self.test_batch_size * (b_idx + 1)]
        product_ids = products_test[self.test_batch_size * b_idx:self.test_batch_size * (b_idx + 1)]

        if b_idx == self.test_num_batches - 1:
            ratings_batch  = ratings_test[self.test_batch_size * b_idx:]
            users_batch = users_test[self.test_batch_size * b_idx:]
            product_ids = products_test[self.test_batch_size * b_idx:]
            
        product_batch = product_ids

        sse = self.session.run(self.training_sse,
                               {self.users: users_batch, 
                                self.products: product_batch,
                                self.rating: ratings_batch})
        return sse

                    
    
    def adjust_from_quick_to_stable_training(self):
        self.learning_rate = self.learning_rate * .99



def plot_confusion_matrix(cm, classesyhat, classesY,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marksyhat = np.arange(len(classesyhat))
    tick_marksY = np.arange(len(classesY))
    plt.xticks(tick_marksyhat, classesyhat, rotation=45)
    plt.yticks(tick_marksY, classesY)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def draw_confusion(yhat, Y):
    
    uniq_yhat = np.unique(yhat)
    uniq_Y = np.unique(Y)
    data = np.zeros((len(uniq_Y), len(uniq_yhat)))
    
    yhat_map = dict(zip(uniq_yhat, range(len(uniq_yhat))))
    inv_yhat_map = dict(zip(range(len(uniq_yhat)), uniq_yhat))
    
    Y_map = dict(zip(uniq_Y, range(len(uniq_Y))))
    inv_Y_map = dict(zip(range(len(uniq_Y)), uniq_Y))
    
    for predict, gt in zip(yhat, Y):
        data[Y_map[gt], yhat_map[predict]] += 1 
    #datadf = pd.DataFrame(data, columns = uniq_yhat, index = uniq_Y)
    ax = sns.heatmap(data, yticklabels = uniq_Y, xticklabels = uniq_yhat, cmap="YlGnBu") 
    ax.set_ylabel('True Labels')
    ax.set_xlabel('Predicted Labels')
    plt.show()
    """
    non-control version
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y, yhat)
    #np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classesyhat = np.unique(yhat), classesY = np.unique(Y),
                          title='Confusion matrix, without normalization')
    
    plt.show()
    """



# List of products in order
#productLenseproducts = pd.read_csv('products.csv')

#feat_df = df_u_by_y[['product', 'cost', 'uber_category']]
#featureMatrix(scrapedproductData)

#feat_df.drop_duplicates(subset='product', inplace=True)

#User and product ids mapped to be on continuous interval
#productratings['product'].unique()
triples, num_users, num_product, user_map, product_map, inverse_user_map, inverse_product_map = \
        map2idx(productratings)

user_idx = triples[:,0]
product_idx = triples[ :,1]
ratings = triples[:, 2]
'''
feat_df.loc[:,'cost2'] = 0
feat_df.loc[feat_df['cost'] < 50, 'cost2'] = 0
feat_df.loc[feat_df['cost'] > 250, 'cost2'] = 2
feat_df['cost'] = feat_df['cost2']
feat_df = feat_df.drop('cost2', 1)
product_idx = product_idx.astype(int)
uni_categories = feat_df['uber_category'].unique()
num_categories = len(uni_categories)
category_map = dict(zip(uni_categories, range(len(uni_categories))))
inverse_category_map = dict(zip(range(len(uni_categories)), uni_categories))
feat_df['uber_category'] = feat_df['uber_category'].map(category_map)
feat_df.reset_index(drop=True, inplace=True)
featMat = np.zeros([num_product, 1 + num_categories])
for idx, feat in enumerate(feat_df['product']):
    feat_df.loc[idx, 'cost']
    featMat[product_map[feat],0] = feat_df.ix[idx, 'cost']
    featMat[product_map[feat],1+feat_df.ix[idx, 'uber_category']] = 1
'''    

conf_dims = [0, 1, 2]
edims_user = [1, 1e-2, 1e-4]
add_noises = [1, 1e-2, 1e-4]
edim_user = 10
add_noise = 0
conf_l = 1
reg_l = 1
errmat = np.zeros([len(conf_dims), len(edims_user), len(add_noises)])

#input_product_dim = featMat.shape[1]
USE_EXISTING_MODEL = True
GRAPH_LEARNING_CURVE = False
if GRAPH_LEARNING_CURVE:
    USE_EXISTING_MODEL = False

ModelPath = "./data/models/tfgraph_uber_test_dates_.ckpt"
print("num_users: ", num_users)
print("num_product: ", num_product)
for conf_dim_idx, conf_dim in enumerate(conf_dims):
    for edim_user_idx, conf_l in enumerate(edims_user):
        for add_noise_idx, reg_l in enumerate(add_noises):
            ModelPath = "./data/models/users= " + str(num_users) + "conf_dim=" + str(conf_dim) + \
                        "edim_user = " + str(edim_user) + "addnoise= " + str(add_noise) +  \
                        "reg_l = " + str(reg_l) + "conf_l = " + str(conf_l) + ".ckpt"

            productModel = HybridCollabFilter(num_users, num_product, conf_dim = conf_dim,
                                              edim_user = edim_user)
            if os.path.isfile(ModelPath) and USE_EXISTING_MODEL == True:
                Saver = tf.train.Saver()
                Saver.restore(productModel.session, ModelPath)
                print("Used existing Model")
            else:
                print("New Model Used")
            if GRAPH_LEARNING_CURVE:
                mse = productModel.create_learning_curve(user_idx,product_idx, ratings,
                                                         epochs = 30, add_noise = add_noise)
                plt.plot(mse[0], label='Training Error')
                plt.plot(mse[1], label = 'Testing Error')
                plt.legend()
                plt.show()
            else:
                errmat[conf_dim_idx, edim_user_idx, add_noise_idx] = \
                productModel.train(user_idx,product_idx, ratings, eval_type = "MSE",
                                   epochs = 5, val_freq = 5, add_noise = add_noise)
            Saver = tf.train.Saver()
            Saver.save(productModel.session, ModelPath)
            print(errmat)





