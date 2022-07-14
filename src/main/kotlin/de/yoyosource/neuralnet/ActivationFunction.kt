package de.yoyosource.neuralnet

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.tanh

interface ActivationFunction {
    fun activate(input: Double): Double
    fun derivative(input: Double): Double
}

object Identity : ActivationFunction {
    override fun activate(input: Double): Double = input
    override fun derivative(input: Double): Double = 1.0
}

object Logistic : ActivationFunction {
    override fun activate(input: Double): Double = 1.0 / (1.0 + exp(-input))
    override fun derivative(input: Double): Double = activate(input) * (1.0 - activate(input))
}

object Sigmoid : ActivationFunction {
    override fun activate(input: Double): Double = 1.0 / (1.0 + exp(-input))
    override fun derivative(input: Double): Double = activate(input) * (1.0 - activate(input))
}

object SoftStep : ActivationFunction {
    override fun activate(input: Double): Double = 1.0 / (1.0 + exp(-input))
    override fun derivative(input: Double): Double = activate(input) * (1.0 - activate(input))
}

object HyperbolicTangent : ActivationFunction {
    override fun activate(input: Double): Double = tanh(input)
    override fun derivative(input: Double): Double = 1.0 - activate(input).pow(2)
}

object SoftPlus : ActivationFunction {
    override fun activate(input: Double): Double = ln(1.0 + exp(input))
    override fun derivative(input: Double): Double = 1.0 / (1.0 + exp(-input))
}

object Gaussian : ActivationFunction {
    override fun activate(input: Double): Double = exp(-input.pow(2))
    override fun derivative(input: Double): Double = -2.0 * input * activate(input)
}