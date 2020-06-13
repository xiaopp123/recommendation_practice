#encoding=utf-8
import os
import math

class UserCF(object):
    def __init__(self):
        #从相似的20个用户中推荐5个电影
        self.n_sim_user = 20
	self.n_rec_movie = 5
	#训练集和测试集
	self.train_set = {}
	self.test_set = {}
	#用户相似矩阵
	self.user_sim_matrix = {}
	#电影个数，为了计算覆盖率
	self.movice_cnt = 0

    def load_data(self, file_name):
        with open(file_name, 'r') as fr:
	    for i, line in enumerate(fr):
	        yield line.strip('\n')

    def get_data(self, file_path):
	#加载训练集
        for line in self.load_data(os.path.join(file_path, 'u1.base')):
	    user, item, rating, timestamp = line.split('\t')
	    if user not in self.train_set:
	        self.train_set[user] = {}
	    self.train_set[user][item] = rating
	#加载测试集
	for line in self.load_data(os.path.join(file_path, 'u1.test')):
	    user, item, rating, timestamp = line.split('\t')
	    if user not in self.test_set:
	        self.test_set[user] = {}
	    self.test_set[user][item] = rating
	print('len(train_set) is %d' % len(self.train_set))
        print('len(test_set) is %d' % len(self.test_set))

    def cal_user_sim(self):
        #建立movie-user倒排索引表
        self.movie_user = {}
	for user, movie in self.train_set.items():
	    for mv in movie:
	        if mv not in self.movie_user:
	            self.movie_user[mv] = []
	        self.movie_user[mv].append(user)
	self.movice_count = len(self.movie_user)
	print('here are all %d movie' % self.movice_count)

        #计算用户相似度矩阵
	for movie, users in self.movie_user.items():
	    for u_a in users:
	        for u_b in users:
		    if u_a == u_b:
		        continue
		    if u_a not in self.user_sim_matrix:
		        self.user_sim_matrix[u_a] = {}
		    if u_b not in self.user_sim_matrix[u_a]:
		        self.user_sim_matrix[u_a][u_b] = 0
		    self.user_sim_matrix[u_a][u_b] += 1

        for u_a, relate_user in self.user_sim_matrix.items():
	    for u_b, cnt in relate_user.items():
	        self.user_sim_matrix[u_a][u_b] /= math.sqrt(\
		    len(self.train_set[u_a]) * len(self.train_set[u_b]))
        print('cal user sim matrix success!')

    def recommand(self, user):
        user_nearest = sorted(self.user_sim_matrix[user].items(),\
	                      key=lambda item: item[1],\
			      reverse=True)
	watched_movice = self.train_set[user]
	rank = {}
	for u_b, weight in user_nearest[:self.n_sim_user]:
	    for rec_movice in self.train_set[u_b]:
	        if rec_movice in watched_movice:
		    continue
		if rec_movice not in rank:
		    rank[rec_movice] = 0
		rank[rec_movice] += weight
	return sorted(rank.items(), key=lambda item: item[1],\
	              reverse=True)[:self.n_rec_movie]

    def evaluate(self):
        hit = 0
	rec_count = 0
	test_count = 0
	all_rec_movice = set()
        for user, test_movice in self.test_set.items():
	    if user not in self.train_set:
	        print('user %s not in train_set' % user)
		continue
	    #print(test_movice)
	    rec_movice = self.recommand(user)
	    for movice, weight in rec_movice:
	        if movice in test_movice:
		    hit += 1
		all_rec_movice.add(movice)
            rec_count += len(rec_movice)
	    test_count += len(test_movice)
	precision = hit / (1.0 + rec_count)
	recall = hit / (1.0 + test_count)
	coverage = len(all_rec_movice) / (1.0 + self.movice_count)
	print('precision is %lf, recall is %lf, coverage is %lf' \
	    % (precision, recall, coverage))

	    
if __name__ == '__main__':
    user_cf = UserCF()
    file_path = '/home/zhaopei.666/recommendation_practice/ml-100k/'
    user_cf.get_data(file_path)
    user_cf.cal_user_sim()
    user_cf.evaluate()
