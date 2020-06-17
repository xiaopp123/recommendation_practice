#encoding=utf-8
import numpy as np

def read_rating(file_name, num_user, num_item, num_total_rating, train_ratio):
    """
    @file_name: 文件名
    @num_user: 用户个数
    @num_item: 物品个数
    @num_total_rating: 文件中数据个数
    @train_ratio: 划分数据的比例
    return
    @train_rating:训练数据
    @train_mask_rating:对训练数据中没用的部分进行遮盖
    @test_rating:测试数据
    @test_mask_rating:对测试数据中没用的部分进行遮盖
    @user_train_set: 训练数据中的用户id
    @item_train_set:训练数据中的物品id
    @user_test_set: 测试数据中的用户id
    @item_test_set: 测试数据中的物品id
    """
    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()
    
    #user-item rating map
    rating_map = np.zeros((num_user, num_item))
    #mask unused rating 
    mask_rating_map = np.zeros((num_user, num_item))
    #user-item click map 
    click_map = np.zeros((num_user, num_item))

    train_rating = np.zeros((num_user, num_item))
    test_rating = np.zeros((num_user, num_item))
    
    #将user没有买过的item遮蔽,训练时计算损失需要
    train_mask_rating = np.zeros((num_user, num_item))
    test_mask_rating = np.zeros((num_user, num_item))

    random_per_ids = np.random.permutation(np.arange(num_total_rating))
    num_train = int(num_total_rating * train_ratio)
    num_test = num_total_rating - num_train
    train_ids = random_per_ids[: num_train]
    test_ids = random_per_ids[num_train:]
    
    all_datas = []

    with open(file_name, 'r') as fr:
        for line in fr:
            user, item, rating, _ = line.split('\t')
            user = int(user) - 1
            item = int(item) - 1
            rating = int(rating)
            rating_map[user][item] = rating
            mask_rating_map[user][item] = 1
            click_map[user][item] = 1
            all_datas.append([user, item, rating])
        for idx in train_ids:
            data = all_datas[idx]
            user, item, rating = data[0], data[1], data[2]
            train_rating[user][item] = rating
            train_mask_rating[user][item] = 1
            user_train_set.add(user)
            item_train_set.add(item)
        for idx in test_ids:
            data = all_datas[idx]
            user, item, rating = data[0], data[1], data[2]
            test_rating[user][item] = rating
            test_mask_rating[user][item] = 1
            user_test_set.add(user)
            item_test_set.add(item)

    return train_rating, train_mask_rating, test_rating, test_mask_rating, user_train_set, item_train_set, user_test_set, item_test_set

if __name__ == '__main__':
    file_name = '/home/zhaopei.666/recommendation_practice/ml-100k/u.data'
    read_rating(file_name, 944, 1683, 100000, 0.8)
