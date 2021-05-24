from sklearn.cluster import AgglomerativeClustering


def run(x_train, y_train,
        n_clusters, affinity, memory, connectivity, compute_full_tree, linkage, distance_threshold
        ):
    reg = AgglomerativeClustering(n_clusters=n_clusters,
                                  affinity=affinity,
                                  memory=memory,
                                  connectivity=connectivity,
                                  compute_full_tree=compute_full_tree,
                                  linkage=linkage,
                                  distance_threshold=distance_threshold).fit(x_train, y_train)
    return {'n_clusters_': reg.n_clusters_,
            'labels_': reg.labels_.tolist(),
            'n_leaves_': reg.n_leaves_,
            'n_connected_components_': reg.n_connected_components_,
            'children_': reg.children_.tolist()}
