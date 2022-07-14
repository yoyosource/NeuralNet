package de.yoyosource.neuralnet

fun neuralNet(builder: NeuralNetBuilder.() -> Unit): NeuralNet {
    val neuralNetBuilder = NeuralNetBuilder()
    neuralNetBuilder.builder()

    if (neuralNetBuilder.inputSize <= 0) {
        throw IllegalArgumentException("Input size must be greater than 0")
    }

    val layers = mutableListOf<NeuralLayer>()
    for (layer in neuralNetBuilder.layers) {
        layers.add(NeuralLayer(layer.second, layer.first, if (layers.isNotEmpty()) layers.last().neurons.size else neuralNetBuilder.inputSize))
    }

    return NeuralNet(layers, neuralNetBuilder.learningRate)
}

class NeuralNetBuilder {
    var learningRate: Double = 0.1
    var inputSize: Int = 0

    internal val layers = mutableListOf<Pair<Int, ActivationFunction>>()

    fun layer(activationFunction: ActivationFunction, neurons: Int): NeuralNetBuilder {
        layers.add(neurons to activationFunction)
        return this
    }
}