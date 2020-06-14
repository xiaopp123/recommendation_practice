#encoding=utf-8
import os
import math

#基于物品的协同过滤
class ItemCF(object):
    def __init__(self):
        #根据20个相似电影，为目标用户推荐5部
        self.n_sim_movie = 20
	self.n_rec_movie = 5
	#训练集和测试集
	self.train_set = {}
	self.test_set = {}

	#电影相似矩阵
	self.movie_sim_matrix = {}
	#电影热度表
	self.movie_popular = {}

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

    def cal_movie_sim(self):
        for user, movies in self.train_set.items():
	    for movie in movies:
	        if movie not in self.movie_popular:
		    self.movie_popular[movie] = 0
		self.movie_popular[movie] += 1

	#构建相似度矩阵
	for user, movies in self.train_set.items():
	    for m1 in movies:
	        for m2 in movies:
		     if m1 == m2:
		         continue
		     if m1 not in self.movie_sim_matrix:
		         self.movie_sim_matrix[m1] = {}
		     if m2 not in self.movie_sim_matrix[m1]:
		         self.movie_sim_matrix[m1][m2] = 0
		     self.movie_sim_matrix[m1][m2] += 1
	for m1, related_movies in self.movie_sim_matrix.items():
	   for m2, count in related_movies.items():
	       self.movie_sim_matrix[m1][m2] /= math.sqrt(\
	           self.movie_popular[m1] * self.movie_popular[m2])
	print('build sim matrix success!')

    def recommand(self, user):
        rank = {}
        watched_movies = self.train_set[user]
	for movie, rating in watched_movies.items():
	     sorted_movies = sorted(self.movie_sim_matrix[movie].items(), key=lambda item: item[1], reverse=True)
	     for related_movie, w in sorted_movies[:self.n_rec_movie]:
	         if related_movie in watched_movies:
		     continue
		 if related_movie not in rank:
		     rank[related_movie] = 0
		 rank[related_movie] += w * float(rating)
        return sorted(rank.items(), key=lambda item: item[1], reverse=True)[:self.n_rec_movie]


    def evaluate(self):
        hit = 0
	rec_count = 0
	test_count = 0
	all_rec_movies = set()
        for user, movies in self.test_set.items():
	    rec_movies = self.recommand(user)
	    watched_movies = self.test_set[user]
	    for movie, w in rec_movies:
	        if movie in watched_movies:
		    hit += 1
		all_rec_movies.add(movie)
	    rec_count += len(rec_movies)
	    test_count += len(watched_movies)
	precision = hit / (1.0 + rec_count)
	recall = hit / (1.0 + test_count)
	coverage = hit / (1.0 + len(self.movie_popular))

	print('precison is %lf, recall is %lf, coverage is %lf' % \
	      (precision, recall, coverage))


if __name__ == '__main__':
    item_cf = ItemCF()
    file_path = '/home/zhaopei.666/recommendation_practice/ml-100k/'
    item_cf.get_data(file_path)
    item_cf.cal_movie_sim()
    item_cf.evaluate()
