from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator



def elbowMethod(dataSet):
    inertia = []
    maxK = 10  # Numero massimo di cluster da testare

    for i in range(1, maxK + 1):
        kmeans = KMeans(n_clusters=i, n_init=10, init='random')
        kmeans.fit(dataSet)
        inertia.append(kmeans.inertia_)

    kl = KneeLocator(range(1, maxK + 1), inertia, curve="convex", direction="decreasing")

    # Visualizzazione del metodo del gomito
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, maxK + 1), inertia, 'bx-')
    plt.scatter(kl.elbow, inertia[kl.elbow - 1], c='red', label=f'Miglior k: {kl.elbow}')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Inertia')
    plt.title('Metodo del Gomito')
    plt.legend()
    plt.show()
    

    return kl.elbow

def calcolaCluster(dataSet):
    """Esegue il clustering con KMeans usando il numero ottimale di cluster determinato dal metodo del gomito."""
    k = elbowMethod(dataSet)
    km = KMeans(n_clusters=k, n_init=10, init='random', random_state=42)
    km.fit(dataSet)
    return km.labels_, km.cluster_centers_
