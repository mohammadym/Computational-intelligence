#write your code here
import numpy as np
def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)
def kmeans(X, k):
    clusters = np.random.choice(np.squeeze(X), size=k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False
    while not converged:
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
        closestCluster = np.argmin(distances, axis=1)
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)
        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()
    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)
    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = X[closestCluster == i]
        if len(pointsForCluster) < 2:
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(X[closestCluster == i])
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))
    return clusters, stds
class RBFNet(object):
    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds
 
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)
    def fit(self, X, y):
      if self.inferStds:
        self.centers, self.stds = kmeans(X, self.k)
      else:
        self.centers, _ = kmeans(X, self.k)
        dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
        self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)
      for epoch in range(self.epochs):
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            loss = (y[i] - F).flatten() ** 2
            error = -(y[i] - F).flatten()
            self.w = self.w - self.lr * a * error
            self.b = self.b - self.lr * error
    def predict(self, X):
      y_pred = []
      for i in range(X.shape[0]):
        a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
        F = a.T.dot(self.w) + self.b
        y_pred.append(F)
      return np.array(y_pred)
if __name__ == '__main__':
  i = -3
  a = []
  while(i <= 3):
    a.append(i)
    i += 0.01
  x = asarray(a)
  y = asarray([math.sin(i) for i in x])
  x = x.reshape((len(x), 1))
  y = y.reshape((len(y), 1))
  scale_x = MinMaxScaler()
  x = scale_x.fit_transform(x)
  scale_y = MinMaxScaler()
  y = scale_y.fit_transform(y)
  model = Sequential()
  model.add(Dense(10, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(1))
  model.compile(loss='mse', optimizer='adam')
  model.fit(x, y, epochs=500, batch_size=10, verbose=0)
  yhat = model.predict(x)
  x_plot = scale_x.inverse_transform(x)
  y_plot = scale_y.inverse_transform(y)
  yhat_plot = scale_y.inverse_transform(yhat)
  NUM_SAMPLES = 100
  X = np.random.uniform(-4., 4., NUM_SAMPLES)
  X = np.sort(X, axis=0)
  noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
  y = np.sin(X)  + noise
  rbfnet = RBFNet(lr=1e-2, k=2)
  rbfnet.fit(X, y)
  y_pred = rbfnet.predict(X)
  pyplot.scatter(x_plot,y_plot, label='Actual')
  pyplot.scatter(x_plot,yhat_plot, label='MLP Predict')
  pyplot.title('Input (x) versus Output (y)')
  pyplot.xlabel('Input Variable (x)')
  pyplot.ylabel('Output Variable (y)')
  pyplot.scatter(X,y_pred, label=' RBF Predict')
  pyplot.legend()
  pyplot.tight_layout()
  pyplot.show()

