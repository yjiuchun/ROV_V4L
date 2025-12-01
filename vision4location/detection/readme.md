detector需要实现extract_feature_points方法：
self.box=np.eye(4,4) # 4x4的单位矩阵
def extract_feature_points(self,image):
    
    return np.array(feature_points, dtype=np.float32), image, self.box  