import numpy as np
from scipy.spatial.distance import cdist


class DDBClustering:
    def __init__(self, d_c, n_clusters):
        # Очев
        self.d_c = d_c
        self.n_clusters = n_clusters
        self._x_ = None
        self._distances_ = None
        self._filter_mask_ = None
        self._centroids_ = None
        self.min_distances = None
        self.density = None
        self.labels = None

    def fit(self, x):
        self._x_ = x
        self._distances_ = cdist(self._x_, self._x_)
        self.density = np.sum(self._distances_ < self.d_c, axis=1)
        self._filter_mask_ = np.array([self.density > self.density[i] for i in range(len(self._x_))])
        self.min_distances = self._min_distance_to_centroid_()
        self._centroids_ = set(np.argsort(-self.density * self.min_distances)[:self.n_clusters])
        self._clusterize_points_()

        return self

    def _clusterize_points_(self):
        # Каждому вектору назначаем метку, соответствующую ближайшему центроиду
        labels = np.full(shape=len(self._x_), fill_value=-1, dtype=int)
        cluster_id = 0
        centroids_list = list(self._centroids_)

        for index in np.argsort(-self.density):
            if index in self._centroids_:
                labels[index] = cluster_id
                cluster_id += 1
                continue

            if np.any(self._filter_mask_[index]):
                near_centroid_index = np.argmin(self._distances_[index, self._filter_mask_[index]])
                labels[index] = labels[np.where(self._filter_mask_[index])[0][near_centroid_index]]
                continue

            closest_centroid_idx = np.argmin(self._distances_[index, centroids_list])
            labels[index] = labels[centroids_list][closest_centroid_idx]

        self.labels = labels

    def _min_distance_to_centroid_(self):
        # Считаем минимальное расстояние до ближайшего центра масс
        min_distances = np.zeros(len(self._x_))

        for i in range(len(self._x_)):
            if np.any(self._filter_mask_[i]):
                min_distances[i] = np.min(self._distances_[i, self._filter_mask_[i]])
            else:
                min_distances[i] = np.max(self._distances_[i, :])

        return min_distances
