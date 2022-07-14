package de.yoyosource.neuralnet

class NeuralLayer(private val activationFunction: ActivationFunction, size: Int, previousSize: Int) {

    var neurons: Array<Neuron> = Array(size) { Neuron(activationFunction, previousSize) }

    fun calculate(input: DoubleArray) {
        for (neuron in neurons) {
            neuron.calculate(input)
        }
    }

    fun outputs(): DoubleArray {
        val outputs = DoubleArray(neurons.size)
        for (i in neurons.indices) {
            outputs[i] = neurons[i].output
        }
        return outputs
    }

    fun adjustWeights(learningRate: Double, error: DoubleArray) {
        for (i in neurons.indices) {
            neurons[i].adjustWeights(learningRate, error[i])
        }
    }

    fun error(last: Boolean, data: DoubleArray, next: Array<Neuron>) : DoubleArray {
        val errors = DoubleArray(neurons.size)
        if (last) {
            for (i in neurons.indices) {
                errors[i] = neurons[i].calculateError(data[i], null, null)
            }
        } else {
            for (i in neurons.indices) {
                val weights = DoubleArray(data.size)
                for (j in next.indices) {
                    weights[j] = next[j].weights[i]
                }
                errors[i] = neurons[i].calculateError(null, data, weights)
            }
        }
        return errors
    }
}