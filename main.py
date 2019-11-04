# Ümit Küçük - 150101027
# 11.04.2019

from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):

        random.seed(1)

        # 3 girdili baglanti ve 1 ciktili baglanti ile tek bir noron modelle
        # Degerleri -1 ve 1 arasinda olan agirliliklari 3 x 1 matrisine ata
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # Sigmoid fonksiyonu, S-tipi egri
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Sigmoid derivative
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Deneme yanilmayla sinir aglarini egit
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):

            output = self.think(training_set_inputs)

            # Hatayi hesapla
            error = training_set_outputs - output

            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Agirliklari ayarla
            self.synaptic_weights += adjustment

    def think(self, inputs):
        # Noronlardan girdileri gecir
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    # Agi initialize et
    neural_network = NeuralNetwork()

    print ("Rastgele: ")
    print (neural_network.synaptic_weights)

    # 4 ornegimiz var, her birinde 3 girdi degeri
    # bir cikti deger
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Set ile neural network'u egit
    # 10.000 kere yap ve her seferinde ayarlama yap
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ("Yeni: ")
    print (neural_network.synaptic_weights)

    # Yeni durumda neural network'u test et
    print ("Yeni durum [1, 0, 0] -> ?: ")
    print (neural_network.think(array([1, 0, 0])))