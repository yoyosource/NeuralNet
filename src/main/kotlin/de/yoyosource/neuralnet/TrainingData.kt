package de.yoyosource.neuralnet

import java.math.BigDecimal
import kotlin.math.absoluteValue

data class TrainingData(val trainingData: List<Pair<DoubleArray, DoubleArray>>)

class TrainingDataBuilder {
    val trainingData: MutableList<Pair<DoubleArray, DoubleArray>> = mutableListOf()
    private val current = this

    fun input(vararg input: Double): TrainingDataInput {
        return TrainingDataInput(DoubleArray(input.size) { input[it] })
    }

    fun input(s: String, padding: Int = s.length): TrainingDataInput {
        return TrainingDataInput(DoubleArray(padding) {
            if (it < s.length) s[it].code.toDouble() else 0.0
        })
    }

    inner class TrainingDataInput(val input: DoubleArray) {
        fun output(vararg output: Double): TrainingDataBuilder {
            trainingData.add(Pair(input, DoubleArray(output.size) { output[it] }))
            return current
        }
    }
}

fun trainingData(builder: TrainingDataBuilder.() -> Unit): TrainingData {
    val builder = TrainingDataBuilder()
    builder.builder()
    return TrainingData(builder.trainingData)
}

fun TrainingData.train(neuralNet: NeuralNet, epochs: Int = -1, maxError: Double = 0.001, logging: Boolean = false) {
    var time = System.currentTimeMillis()
    var currentError = BigDecimal.valueOf(maxError) + BigDecimal.ONE
    val maxError = BigDecimal.valueOf(maxError)
    var epoch = 0
    while (currentError > maxError || (epochs > 0 && epoch < epochs)) {
        epoch++
        currentError = BigDecimal.ZERO
        for (element in this.trainingData) {
            neuralNet.calculate(element.first)
            currentError += BigDecimal.valueOf(neuralNet.train(element.second).absoluteValue)
        }
        if (System.currentTimeMillis() - time > 1000) {
            time = System.currentTimeMillis()
            if (logging) print("\rEpoch: $epoch, Error: $currentError")
        }
    }
    if (logging) print("\rEpoch: $epoch, Error: $currentError")
}