package de.yoyosource.neuralnet

class Neuron(private val activationFunction: ActivationFunction, size: Int) {

    private var _input: DoubleArray = DoubleArray(0)
    private var _output: Double = 0.0
    val output: Double
        get() = _output
    private var sum: Double = 0.0
    internal var weights: DoubleArray = DoubleArray(size) { Math.random() }

    fun calculate(input: DoubleArray) {
        if (input.size != weights.size) {
            throw IllegalArgumentException("Input and weights size must be equal")
        }
        _input = input
        sum = 0.0
        for (i in input.indices) {
            sum += input[i] * weights[i]
        }
        _output = activationFunction.activate(sum)
    }

    fun adjustWeights(learningRate: Double, error: Double) {
        for (i in weights.indices) {
            weights[i] += learningRate * error * _input[i]
        }
    }

    fun calculateError(expected: Double?, errors: DoubleArray?, weights: DoubleArray?): Double {
        val derivative = activationFunction.derivative(sum)
        return if (expected != null) {
            derivative * (expected - _output)
        } else {
            var sum = 0.0
            for (i in errors!!.indices) {
                sum += errors[i] * weights!![i]
            }
            derivative * sum
        }
    }
}