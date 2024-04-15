import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

# ======================================================================
#                    Numpy를 이용한 MLP 모델 구현
# ======================================================================

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    
    def loss(self, Y, Y_hat):
        return np.mean(np.power(Y - Y_hat, 2))
    
    def backward(self, X, Y, learning_rate):
        m = Y.shape[0]
        dZ2 = (self.A2 - Y) * self.A2 * (1 - self.A2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.A1 * (1 - self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        
    def train(self, X, Y, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]
                
                Y_hat = self.forward(X_batch)
                loss = self.loss(Y_batch, Y_hat)
                self.backward(X_batch, Y_batch, learning_rate)
            
            # 10 epoch마다 loss를 출력하려면 아래 코드 주석 해제 및 기존 print문 주석 처리
            # if epoch % 10 == 0:
            #     print(f'Epoch {epoch}/{epochs}, Loss: {loss}')
            print(f'Epoch {epoch}/{epochs}, Loss: {loss}')
            
    def evaluate(self, X_test, Y_test):
        Y_hat = self.forward(X_test)
        predictions = np.argmax(Y_hat, axis=1)
        labels = np.argmax(Y_test, axis=1)
        accuracy = np.mean(predictions == labels)
        print(f'Accuracy: {accuracy * 100:.2f}%')


# ======================================================================
#           MNIST 데이터셋을 이용하여 MLP 모델 학습 및 평가
# ======================================================================

mnist = datasets.fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target

X = X / 255.0
y = y.astype(int)
y = np.eye(10)[y]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

model = MLP(input_size=784, hidden_size=100, output_size=10)
model.train(X_train, y_train, epochs=100, batch_size=64, learning_rate=0.01)
model.evaluate(X_test, y_test)