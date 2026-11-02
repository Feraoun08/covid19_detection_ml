# Prédiction du COVID-19 à partir de tests sanguins et viraux

Projet de Machine Learning : prédire le résultat d'un test **SARS-Cov-2** (`positive` / `negative`) à partir des données cliniques de patients (taux sanguins, tests viraux, âge).

Le travail est découpé en **3 phases** : Exploratory Data Analysis → Preprocessing → Modélisation.

---

## 1. Exploratory Data Analysis

**Objectifs :**
- Comprendre du mieux possible nos données (un petit pas en avant vaut mieux qu'un grand pas en arrière)
- Développer une première stratégie de modélisation

### Analyse de forme
- **Variable target** : `SARS-Cov-2 exam result`
- **Lignes et colonnes** : 5644 × 111
- **Types de variables** : quantitatives : 70, qualitatives : 41
- **Valeurs manquantes** : beaucoup de NaN (la moitié des variables ont > 90 % de NaN)
  - 2 groupes de données : **76 % de NaN → tests viraux**, **89 % de NaN → taux sanguins**

### Analyse de fond
- **Target** : ~10 % de positifs (558 / 5644), jeu de données déséquilibré
- **Signification des variables** :
  - variables continues standardisées, asymétriques (skewed) : tests sanguins
  - `Patient age quantile` : donnée déjà transformée, l'âge réel est inconnu (peu important)
  - variables qualitatives : binaires (0/1) ou viral (`detected` / `not_detected`), le Rhinovirus/Enterovirus semble très élevé
- **Relations variables / target** :
  - **target / blood** : Monocytes, Platelets, Leukocytes semblent liés au COVID-19 → hypothèse à tester
  - **target / âge** : les individus de faible âge sont très peu contaminés ? → prudence, l'âge exact et la date du dataset sont inconnus ; variable intéressante à comparer avec les tests sanguins
  - **target / viral** : les doubles maladies sont très rares ; Rhinovirus/Enterovirus positif et COVID-19 négatif ? → hypothèse à tester (possible épidémie régionale, et on peut avoir 2 virus en même temps, sans lien avec le COVID-19)

### Analyse plus détaillée
- **blood / blood** : certaines variables sont très corrélées (+0.9), à surveiller
- **blood / âge** : très faible corrélation
- **viral / viral** : le test rapide *Influenza* donne de mauvais résultats, il faudra peut-être l'abandonner
- **maladie / blood** : les taux sanguins entre malades (autre virus) et non malades sont différents
- **hospitalisation / blood** : intéressant pour prédire dans quel service un patient devrait aller
- **Analyse des NaN** : viral : 1350 (92 / 8), blood : 600 (87 / 13), both : 90

### Tests d'hypothèses (t-test, α = 0.02)
- **H0** : les taux moyens sont **ÉGAUX** chez les individus positifs et négatifs
- **Résultat** : H0 **rejetée** pour **Platelets, Leukocytes, Monocytes** (et Eosinophils) → ces variables diffèrent significativement chez les patients COVID-19
- Les individus atteints d'une quelconque maladie ont également des taux sanguins significativement différents

---

## 2. Preprocessing

- Garde uniquement les colonnes utiles : âge, target, **14 variables sanguines** et **tests viraux** (colonnes très vides supprimées)
- **Train / test split** : 80 % / 20 % (`random_state=0`)
- **Encodage** : `positive`/`detected` → 1, `negative`/`not_detected` → 0
- **Feature engineering** : création de la variable `est malade` (au moins un virus détecté), puis suppression des colonnes virales
- **Nettoyage des NaN** : `dropna()` sur les lignes
- **Jeu final** :
  - train : 487 échantillons (422 négatifs / 65 positifs)
  - test : 111 échantillons (95 négatifs / 16 positifs)

---

## 3. Modélisation

**Pipeline** : `PolynomialFeatures(2)` → `SelectKBest(f_classif, k=10)` → (`StandardScaler`) → modèle

### Modèles comparés (classe positive)

| Modèle       | Precision | Recall | F1   |
|--------------|-----------|--------|------|
| RandomForest | 0.56      | 0.31   | 0.40 |
| AdaBoost     | 0.50      | 0.38   | 0.43 |
| SVM          | 0.67      | 0.38   | 0.48 |
| KNN          | 0.53      | 0.50   | 0.52 |

### Optimisation (SVM, `RandomizedSearchCV`, scoring = recall)
- Meilleurs hyperparamètres : `C=1000`, `gamma=0.0005`, `PolynomialFeatures degree=3`, `SelectKBest k=53`
- Résultat : **Precision = 0.62 | Recall = 0.50 | F1 = 0.55** (accuracy 0.88)

### Courbe Precision / Recall et seuil de décision
- Abaissement du seuil de décision à **-1** (au lieu de 0) pour privilégier le recall
- Résultat final : **Recall = 0.69 | F1 = 0.54**
- On accepte plus de faux positifs pour mieux détecter les vrais cas COVID-19

---

## Source

Ce projet est basé sur la série de **30 vidéos de la chaîne YouTube Machine Learnia**.
