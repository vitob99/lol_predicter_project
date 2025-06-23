import matplotlib
matplotlib.use('Agg')  # Evita problemi con Tkinter in ambienti senza interfaccia grafica
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def apprendimento_supervisionato(df):
    # Separazione input/target
    X = df.drop(columns=['result'])
    y = df['result']

    # Codifica delle variabili categoriche
    X_encoded = pd.get_dummies(X, columns=['role', 'champion', 'side', 'd_spell', 'f_spell'], drop_first=True)

    # Definizione della cross-validation esterna (5-fold)
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Inizializzazione dei modelli
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear')
    }

    # Definizione degli iperparametri per Grid Search
    param_grid = {
        "Decision Tree": {
            'max_depth': [5, 10, 15, 20, 30],  
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 5, 10],
            'criterion': ['gini', 'entropy', 'log_loss']
        },
        "Random Forest": {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, 20],  
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 5, 10]  
        },
        "Logistic Regression": {
            'C': [0.1, 0.5, 1, 5, 10],  
            'solver': ['liblinear', 'saga', 'lbfgs'],
            'penalty': ['l2']
        },
    }

    # Dizionari per memorizzare i risultati
    model_scores = {}
    model_stds = {}

    # Creazione di una lista per memorizzare i report di classificazione
    classification_reports = []

    def calcola_metriche(y_test, y_pred_test):
        """Calcola le metriche di performance del modello."""
        report_test = classification_report(y_test, y_pred_test, output_dict=True)
        f1_macro = report_test['macro avg']['f1-score']
        precision_macro = report_test['macro avg']['precision']
        recall_macro = report_test['macro avg']['recall']
        return f1_macro, precision_macro, recall_macro

    for train_idx, test_idx in outer_cv.split(X_encoded, y):
        X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for model_name, model in models.items():
            try:
                print(f"\nModello: {model_name}")

                # Cross-validation interna per la ricerca degli iperparametri
                inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid[model_name], 
                                           cv=inner_cv, n_jobs=-1, verbose=1)
                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Migliori iperparametri per {model_name}: {best_params}")

                # Previsioni sul test set esterno
                y_pred_test = best_model.predict(X_test)
                acc_test = accuracy_score(y_test, y_pred_test)
                f1_macro, precision_macro, recall_macro = calcola_metriche(y_test, y_pred_test)

                # Stampa risultati test set esterno
                print(f"Accuracy Test: {acc_test:.4f}")
                print(f"F1-score Macro: {f1_macro:.4f}")
                print(f"Precision Macro: {precision_macro:.4f}")
                print(f"Recall Macro: {recall_macro:.4f}")

                # Cross-validation per ottenere l'accuratezza di ogni fold
                cv_scores = cross_val_score(best_model, X_encoded, y, cv=outer_cv)
                model_scores[model_name] = cv_scores.mean()

                # Calcolare la deviazione standard delle prestazioni del modello
                model_stds[model_name] = cv_scores.std()

                # Aggiungi l'accuratezza per ogni fold e la deviazione standard
                print("\nRisultati per ogni fold:")
                for i, score in enumerate(cv_scores):
                    print(f"Fold {i+1}: Accuracy = {score:.4f}")

                # Stampa la deviazione standard per ogni modello
                print(f"Deviazione Standard: {model_stds[model_name]:.4f}")

                # Salva il classification report per ogni modello
                classification_reports.append({
                    'Model': model_name,
                    'Accuracy': acc_test,
                    'F1 Macro': f1_macro,
                    'Precision Macro': precision_macro,
                    'Recall Macro': recall_macro,
                })

            except Exception as e:
                print(f"Errore con il modello {model_name}: {e}")



    return model_scores, model_stds

