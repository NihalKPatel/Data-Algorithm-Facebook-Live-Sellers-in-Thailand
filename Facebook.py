import time
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from yellowbrick.cluster import SilhouetteVisualizer

# ------------------------------- preprocessing of dataset Facebook -------------------------------#
# Dataset shape


data = pd.read_csv("Live_20210128.csv", encoding='unicode_escape')
data.head(10)
data.head()

data.rename({"num_reactions": "reactions",
             "num_comments": "comments",
             "num_shares": "shares",
             "num_likes": "likes",
             "num_loves": "loves",
             "num_wows": "wows",
             "num_hahas": "hahas",
             "num_sads": "sads",
             "num_angrys": "angrys"}, axis=1, inplace=True)

data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
nrows, ncols = data.shape
data.info()
data.count()
data.head(10)

# ------------------------------- exploration of dataset Facebook -------------------------------#
data.isnull().sum()
data.duplicated().sum()
data.isna().sum()

data.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)

data['status_id'].unique()
len(data['status_id'].unique())
data['status_published'].unique()
len(data['status_published'].unique())
data['status_type'].unique()
len(data['status_type'].unique())
print(data.nunique())

data.drop('status_id', axis=1, inplace=True)
data['status_type_isvideo'] = data['status_type'].map(lambda x: 1 if (x == 'video') else 0)
data['status_published'] = pd.to_datetime(data['status_published'])

data['status_published'] = pd.to_datetime(data['status_published'])
data['year'] = data['status_published'].dt.year
data['month'] = data['status_published'].dt.month
data['dayofweek'] = data['status_published'].dt.dayofweek  # 0 is Monday, 7 is Sunday.
data['hour'] = data['status_published'].dt.hour

st_ax = data.status_type.value_counts().plot(kind='bar', figsize=(10, 5), title="Status Type")
st_ax.set(xlabel="Status Type", ylabel="Count")
plt.show()

file = 'Live_20210128_Cleaned_data.csv'
data.to_csv(file, index=False)

reaction = ['reactions', 'comments', 'shares', 'likes', 'loves', 'wows', 'hahas',
            'sads', 'angrys']

# -------------------------------Feature Selection Facebook -------------------------------#
# Export to csv and save cleaned dataset
file = 'Live_20210128_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file)

le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)

# ANOVA FEATURE SELECTION
X = data.iloc[:, 1:]
y = data.iloc[:, 1]

from sklearn.preprocessing import MinMaxScaler

# Split data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Chi-squared feature selection
fs = SelectKBest(score_func=chi2, k=10)
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)

# Scores for each feature
print('Feature scores using Chi-square: ', '\n')
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))

# Plot scores
labels = data.iloc[:, 1:]

ms = MinMaxScaler()

labels = ms.fit_transform(labels)
fig, ax = plt.subplots()
ax.set_xlabel("Features")
ax.set_ylabel("Chi-Square Scores")
ax.set_title("Facebook Feature Selection ")
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()

# Export to csv and save cleaned dataset
file = 'Live_20210128_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file)

# data = data[['num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
# 'num_sads', 'num_angrys']]

data.drop(['status_published', 'year', 'month', 'dayofweek', 'hour', 'status_type_isvideo'], axis=1, inplace=True)

file = 'Live_20210128_Cleaned_data.csv'
data.to_csv(file, index=False)
print(data.info())
le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)

# ANOVA FEATURE SELECTION
X = data.iloc[:, 1:]
y = data.iloc[:, 1]

from sklearn.preprocessing import MinMaxScaler

# Split data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Chi-squared feature selection
fs = SelectKBest(score_func=chi2, k=8)
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)

# Scores for each feature
print('Completeed Feature scores using Chi-square: ', '\n')
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))

# Plot scores
labels = data.iloc[:, 1:]

ms = MinMaxScaler()

labels = ms.fit_transform(labels)
fig, ax = plt.subplots()
ax.set_xlabel("Features")
ax.set_ylabel("Chi-Square Scores")
ax.set_title("Facebook Feature Selection ")
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()
# ------------------------------- K Means Facebook  ------------------------------- #

file = 'Live_20210128_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

X = data.loc[:, ['shares', 'comments']]

# image size
plt.figure(figsize=(10, 5))

# ploting scatered graph
plt.scatter(x=X['shares'], y=X['comments'])
plt.xlabel('shares')
plt.ylabel('comments')
plt.show()
# ------------------------------- K VALUE Facebook  ------------------------------- #

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 5))
sns.lineplot(range(1, 11), wcss, marker='o', color='green')
print(wcss)
plt.title('The Elbow Method Facebook')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
Kmean_n_clusters_Facebook = 2

# ------------------------------- Kmeans Facebook with clusters ------------------------------- #
# Elbow Method Shows 3 is the optimal number of clusters
start = time.time()
file = 'Live_20210128_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

X = data.loc[:, ['shares', 'comments']]

km = KMeans(Kmean_n_clusters_Facebook)
km.fit(X)

# ploting the graph of the clusters
plt.figure(figsize=(10, 5))
scatter = plt.scatter(x=X.iloc[:, 0], y=X.iloc[:, 1], c=km.labels_, cmap="Set2")
plt.xlabel('shares')
plt.ylabel('comments')
plt.legend(handles=scatter.legend_elements()[0], labels=[0, 1, 2, 3])
plt.show()
finish = time.time()

# ------------------------------- Kmeans Facebook with Time Taken ------------------------------- #
Kmean_time_taken = {finish - start}

# ------------------------------- Kmeans Facebook with Davis-Bouldin score  ------------------------------- #
file = 'Live_20210128_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['comments']
x = data[['shares']]

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=30).fit(x)
labels = kmeans.fit_predict(x)

kmeans_Davis_Bouldin_score = davies_bouldin_score(X, labels)
# ------------------------------- Kmeans Facebook CSM ------------------------------- #
file = 'Live_20210128_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['comments']
x = data[['shares']]
km = KMeans(n_clusters=2, random_state=42)

km.fit_predict(X)

kmeans_silhouette_avg = silhouette_score(X, km.labels_, metric='euclidean')

fig, ax = plt.subplots(2, 2, figsize=(15, 8))
for i in [2, 3, 4, 5]:
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    plt.title("$k={}$".format(i), fontsize=16)

    plt.ylabel("Cluster")

    plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlabel("Silhouette Coefficient")
    q, mod = divmod(i, 2)

    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q - 1][mod])
    visualizer.fit(x)
plt.show()

# ------------------------------- Agglomerative Bike with clusters ------------------------------- #
start = time.time()

file = 'Live_20210128_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

X = data.loc[:, ['shares', 'comments']]
aggloclust = AgglomerativeClustering(n_clusters=2).fit(X)
plt.figure(figsize=(10, 5))
labels = aggloclust.labels_
plt.scatter(x=X.iloc[:, 0], y=X.iloc[:, 1], c=labels, cmap="Set2")
plt.xlabel('shares')
plt.ylabel('comments')
plt.title("2 Cluster Agglomerative")
plt.show()
finish = time.time()
# ------------------------------- Agglomerative Bike with Time Taken ------------------------------- #

Agglomerative_time_taken = finish - start

# ------------------------------- Agglomerative Bike with Davis-Bouldin score  ------------------------------- #
file = 'Live_20210128_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['comments']
x = data[['shares']]

n_clusters = 2
model = AgglomerativeClustering(n_clusters=2)
# fit model and predict clusters
yhat_2 = model.fit_predict(x)

Agglomerative_Davis_Bouldin_score = davies_bouldin_score(x, yhat_2)

# ------------------------------- Agglomerative with CSM ------------------------------- #
file = 'Live_20210128_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['comments']
x = data[['shares']]

agg_avg = AgglomerativeClustering(linkage='average', n_clusters=2)
as_avg = agg_avg.fit(x)
Agglomerative_silhouette_avg = silhouette_score(x, as_avg.labels_, metric='euclidean')
# ------------------------------- DBSCAN Bike with clusters ------------------------------- #
start = time.time()

file = 'Live_20210128_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

X = data[['shares', 'comments']]

X = np.nan_to_num(X)
X = np.array(X, dtype=np.float64)
X = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.4, min_samples=5).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
data['Clus_Db'] = db.labels_

realClusterNum = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
clusterNum = len(set(db.labels_))

plt.figure(figsize=(15, 10))
unique_labels = set(db.labels_)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
class_member_mask = (db.labels_ == k)
xy = X[class_member_mask & core_samples_mask]
plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
         markeredgecolor='k', markersize=14)
xy = X[class_member_mask & ~core_samples_mask]
plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
         markeredgecolor='k', markersize=6)
plt.title('Estimated Number of Clusters: %d' % realClusterNum, fontweight='bold', fontsize=20)
plt.legend(fontsize=20)
n_noise_ = list(db.labels_).count(-1)
print('number of noise(s): ', n_noise_)
plt.xlabel('shares')
plt.ylabel('comments')
plt.show()
finish = time.time()
# ------------------------------- DBSCAN Bike with Time Taken ------------------------------- #

DBSCAN_time_taken = finish - start

# ------------------------------- DBSCAN Bike with Davis-Bouldin score  ------------------------------- #
file = 'Live_20210128_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['comments']
x = data[['shares']]

db = DBSCAN(eps=0.3, min_samples=10).fit(x)

DBSCAN_Davis_Bouldin_score = davies_bouldin_score(x, db.labels_)

# ------------------------------- DBSCAN with CSM ------------------------------- #
file = 'Live_20210128_Cleaned_data.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['comments']
x = data[['shares']]

db = DBSCAN(eps=0.3, min_samples=10).fit(x)

DBSCAN_silhouette_avg = metrics.silhouette_score(x, db.labels_)

# ------------------------------- Tabulate Form ------------------------------- #

all_data = [["CSM", "Davis Bouldin Score", "TimeTaken"],
            [("No.Cluster", Kmean_n_clusters_Facebook, "Avg", kmeans_silhouette_avg), kmeans_Davis_Bouldin_score,
             (Kmean_time_taken, "Secs in Kmean")],
            [("No.Cluster", Kmean_n_clusters_Facebook, "Avg", Agglomerative_silhouette_avg),
             Agglomerative_Davis_Bouldin_score,
             (Agglomerative_time_taken, "Secs in Agglomerative")],
            [("No.Cluster", Kmean_n_clusters_Facebook, "Avg", DBSCAN_silhouette_avg),
             DBSCAN_Davis_Bouldin_score,
             (DBSCAN_time_taken, "Secs in DBscan")]
            ]

print(tabulate(all_data, headers='firstrow', tablefmt='grid'))
