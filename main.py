import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from apprendimentoNonSupervisionato import calcolaCluster
from apprendimentoSupervisionato import apprendimento_supervisionato

def visualizeAspectRatioChart(dataSet, differentialColumn, title):
    """Genera un grafico a torta per visualizzare la distribuzione della colonna di riferimento."""
    counts = dataSet[differentialColumn].value_counts()
    labels = counts.index.tolist()
    colors = plt.cm.Paired.colors[:len(labels)]  # Palette di colori dinamica

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.legend(labels, loc='lower left', fontsize='small')
    plt.title(title)
    plt.show()



   

if __name__ == "__main__":
     #NON SUPERVISIONATO

    # Caricamento dati
    file_name = os.path.join(os.path.dirname(__file__), "players_games.csv")
    df = pd.read_csv(file_name, sep=";", header=0)

    # Colonne categoriche da escludere dalla normalizzazione
    cols_categoriche = ['id', 'd_spell', 'f_spell', 'champion', 'side', 'role', 'result']
    cols_numeriche = df.columns.difference(cols_categoriche)

    # Normalizzazione solo delle feature numeriche
    scaler = MinMaxScaler()
    df[cols_numeriche] = scaler.fit_transform(df[cols_numeriche])

    # Creo un dataframe completo (con colonne stringa mantenute)
    df_cluster = df.drop(columns=['id'])  # Rimuovo solo ID

    # Seleziono solo le colonne numeriche per il clustering, escludendo anche 'result'
    X_for_clustering = df_cluster.select_dtypes(include=['float64', 'int64']).drop(columns=['result'])

    # Visualizzazione ruoli prima del clustering
    visualizeAspectRatioChart(df_cluster, "role", "Distribuzione dei Ruoli Prima del Clustering")
 

    # Calcolo dei cluster
    etichette_cluster, centroidi = calcolaCluster(X_for_clustering)
    df_cluster['clusterIndex'] = etichette_cluster  # Aggiunta dei cluster

    # Visualizzazione distribuzione dei cluster
    visualizeAspectRatioChart(df_cluster, "clusterIndex", "Clustering dei Playstyle rilevati")

    
    #print(pd.crosstab(df_cluster['clusterIndex'], df_cluster['result'], normalize='index')) #winrate in ogni cluster
    #print(pd.crosstab(df_cluster['clusterIndex'], df['champion'])) #champ in ogni cluster
    #print(df.groupby('role')['result'].mean()) #winrate ruoli
    #print(df.groupby('champion')['result'].mean()) #winrate champ




    output_path = os.path.join(os.path.dirname(__file__), "newDataset.csv")
    df_cluster.to_csv(output_path, index=False, sep=";", encoding="utf-8-sig")


    #SUPERVISIONATO
    apprendimento_supervisionato(df_cluster)


    file_name = os.path.join(os.path.dirname(__file__), "newDataset.csv")
    df_cluster_bayes = pd.read_csv(file_name, sep=";", header=0)
    cols_categoriche = ['d_spell', 'f_spell', 'champion', 'side', 'role']
    cols_numeriche = df_cluster_bayes.columns.difference(cols_categoriche + ['result'])

    #Encoding delle variabili categoriche
    label_encoders = {}
    for col in cols_categoriche:
        le = LabelEncoder()
        df_cluster_bayes[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Normalizzazione delle feature numeriche
    scaler = MinMaxScaler()
    df_cluster_bayes[cols_numeriche] = scaler.fit_transform(df_cluster_bayes[cols_numeriche])

    # Definizione delle feature (X) e della variabile target (y)
    X = df_cluster_bayes.drop(columns=['result'])  # Escludiamo la variabile target
    y = df_cluster_bayes['result']

    # Divisione in training e test set (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Inizializzazione e addestramento del modello Naïve Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Predizione sui dati di test
    y_pred = nb_model.predict(X_test)

    # Valutazione del modello
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)