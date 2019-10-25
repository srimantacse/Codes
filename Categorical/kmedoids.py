# pip install pyclustering
from pyclustering.cluster.kmedoids import kmedoids

class KMedoids:
    def __init__(self, data, initial_medoids):
        self.kmedoids_ = kmedoids(data, initial_medoids)
        
    def fit():
        self.kmedoids_.process()
        
    def get_clusters():
        return self.kmedoids_.get_clusters()
        