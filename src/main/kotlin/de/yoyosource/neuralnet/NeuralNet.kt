package de.yoyosource.neuralnet

class NeuralNet(var layers: List<NeuralLayer>, var learningRate: Double = 0.1) {

    init {
        if (layers.size < 1) {
            throw IllegalArgumentException("NeuralNet must have at least 2 layers")
        }
    }

    fun calculate(input: DoubleArray): DoubleArray {
        var output = input
        for (layer in layers) {
            layer.calculate(output)
            output = layer.outputs()
        }
        return output
    }

    fun train(expected: DoubleArray): Double {
        if (layers.last().neurons.size != expected.size) {
            throw IllegalArgumentException("Expected output size does not match output layer size")
        }
        var errors = layers.last().error(true, expected, emptyArray())
        val currentError = errors.sum()
        layers.last().adjustWeights(learningRate, errors)
        for (i in layers.size - 2 downTo 0) {
            errors = layers[i].error(false, errors, layers[i + 1].neurons)
            layers[i].adjustWeights(learningRate, errors)
        }
        return currentError
    }
}