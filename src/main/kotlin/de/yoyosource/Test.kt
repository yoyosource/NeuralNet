package de.yoyosource

import de.yoyosource.neuralnet.*
import java.math.BigDecimal
import kotlin.math.absoluteValue

fun main() {
    test()
}

fun test() {
    // I, you, he, she, it
    val neuralNet = neuralNet {
        inputSize = 3
        layer(Sigmoid, 3)
        layer(Sigmoid, 3)
        layer(Sigmoid, 3)
        layer(Sigmoid, 3)
        layer(Sigmoid, 3)
        layer(Sigmoid, 3)
        layer(Sigmoid, 3)
    }

    val trainingData = trainingData {
        input("I", 3).output(1.0, 0.0, 0.0)
        input("you", 3).output(0.0, 1.0, 0.0)
        input("he", 3).output(0.0, 0.0, 1.0)
        input("she", 3).output(0.0, 0.0, 1.0)
        input("it", 3).output(0.0, 0.0, 1.0)
    }

    trainingData.train(neuralNet, logging = true)

    println(neuralNet.calculate(doubleArrayOf('I'.code.toDouble(), 0.0, 0.0)).contentToString())
}

fun xorGaussian() {
    val neuralNet = neuralNet {
        inputSize = 3
        layer(Sigmoid, 2)
        layer(Sigmoid, 4)
        layer(Sigmoid, 4)
        layer(Sigmoid, 1)
    }

    val trainingData = trainingData {
        input(0.0, 0.0, 0.0).output(0.0)
        input(1.0, 0.0, 0.0).output(1.0)
        input(0.0, 1.0, 0.0).output(1.0)
        input(0.0, 0.0, 1.0).output(1.0)
        input(1.0, 1.0, 0.0).output(0.0)
        input(0.0, 1.0, 1.0).output(0.0)
        input(1.0, 0.0, 1.0).output(0.0)
        input(1.0, 1.0, 1.0).output(1.0)
    }

    trainingData.train(neuralNet, logging = true)
}

fun xor() {
    val neuralNet = neuralNet {
        inputSize = 2
        layer(Sigmoid, 2)
        layer(Sigmoid, 1)
    }

    val inputs = listOf(
        doubleArrayOf(0.0, 0.0),
        doubleArrayOf(0.0, 1.0),
        doubleArrayOf(1.0, 0.0),
        doubleArrayOf(1.0, 1.0)
    )
    val outputs = listOf(
        doubleArrayOf(0.0),
        doubleArrayOf(1.0),
        doubleArrayOf(1.0),
        doubleArrayOf(0.0)
    )

    neuralNet.train(inputs, outputs, 0.0001)
}

fun and() {
    val neuralNet = neuralNet {
        inputSize = 2
        layer(Sigmoid, 2)
        layer(Sigmoid, 2)
        layer(Sigmoid, 1)
    }

    val inputs = listOf(
        doubleArrayOf(0.0, 0.0),
        doubleArrayOf(0.0, 1.0),
        doubleArrayOf(1.0, 0.0),
        doubleArrayOf(1.0, 1.0)
    )
    val outputs = listOf(
        doubleArrayOf(0.0),
        doubleArrayOf(0.0),
        doubleArrayOf(0.0),
        doubleArrayOf(1.0)
    )

    neuralNet.train(inputs, outputs)
}

fun or() {
    val neuralNet = neuralNet {
        inputSize = 2
        layer(Sigmoid, 2)
        layer(Sigmoid, 1)
    }

    val inputs = listOf(
        doubleArrayOf(0.0, 0.0),
        doubleArrayOf(0.0, 1.0),
        doubleArrayOf(1.0, 0.0),
        doubleArrayOf(1.0, 1.0)
    )
    val outputs = listOf(
        doubleArrayOf(0.0),
        doubleArrayOf(1.0),
        doubleArrayOf(1.0),
        doubleArrayOf(1.0)
    )

    neuralNet.train(inputs, outputs)
}

fun not() {
    val neuralNet = neuralNet {
        inputSize = 1
        layer(Sigmoid, 1)
        layer(Sigmoid, 2)
        layer(Sigmoid, 1)
    }

    val inputs = listOf(
        doubleArrayOf(0.0),
        doubleArrayOf(1.0)
    )
    val outputs = listOf(
        doubleArrayOf(1.0),
        doubleArrayOf(0.0)
    )

    neuralNet.train(inputs, outputs)
}

fun binaryHalfAdder() {
    val neuralNet = neuralNet {
        inputSize = 3
        layer(Sigmoid, 2)
        layer(Sigmoid, 4)
        layer(Sigmoid, 2)
    }

    val inputs = listOf(
        doubleArrayOf(0.0, 0.0, 0.0),
        doubleArrayOf(0.0, 1.0, 0.0),
        doubleArrayOf(1.0, 0.0, 0.0),
        doubleArrayOf(1.0, 1.0, 0.0),
        doubleArrayOf(0.0, 0.0, 1.0),
        doubleArrayOf(0.0, 1.0, 1.0),
        doubleArrayOf(1.0, 0.0, 1.0),
        doubleArrayOf(1.0, 1.0, 1.0),
    )
    val outputs = listOf(
        doubleArrayOf(0.0, 0.0),
        doubleArrayOf(1.0, 0.0),
        doubleArrayOf(1.0, 0.0),
        doubleArrayOf(0.0, 1.0),
        doubleArrayOf(1.0, 0.0),
        doubleArrayOf(0.0, 1.0),
        doubleArrayOf(0.0, 1.0),
        doubleArrayOf(1.0, 1.0)
    )

    neuralNet.train(inputs, outputs, 0.001)
}

fun sin() {
    val neuralNet = neuralNet {
        inputSize = 1
        layer(Sigmoid, 1)
        layer(Sigmoid, 10)
        layer(Sigmoid, 10)
        layer(Sigmoid, 10)
        layer(Sigmoid, 1)
    }

    val inputs = mutableListOf<DoubleArray>()
    for (i in -100..100) {
        inputs.add(doubleArrayOf(i / 10.0))
    }
    val outputs = inputs.map { doubleArrayOf(kotlin.math.sin(it[0])) }

    neuralNet.train(inputs, outputs, 0.001)
}

private fun NeuralNet.train(inputs: List<DoubleArray>, outputs: List<DoubleArray>) {
    var time = System.currentTimeMillis()
    for (epoch in 0..10000000) {
        var currentError = 0.0
        for (i in inputs.indices) {
            this.calculate(inputs[i])
            currentError += this.train(outputs[i]).absoluteValue
        }
        if (System.currentTimeMillis() - time > 1000) {
            time = System.currentTimeMillis()
            print("\rEpoch: $epoch, Error: $currentError")
        }
    }
}

private fun NeuralNet.train(inputs: List<DoubleArray>, outputs: List<DoubleArray>, maxError: Double) {
    var time = System.currentTimeMillis()
    var currentError = BigDecimal.valueOf(maxError) + BigDecimal.ONE
    val maxError = BigDecimal.valueOf(maxError)
    var epoch = 0
    while (currentError > maxError) {
        epoch++
        currentError = BigDecimal.ZERO
        for (i in inputs.indices) {
            this.calculate(inputs[i])
            currentError += BigDecimal.valueOf(this.train(outputs[i]).absoluteValue)
        }
        if (System.currentTimeMillis() - time > 1000) {
            time = System.currentTimeMillis()
            print("\rEpoch: $epoch, Error: $currentError")
        }
    }
    print("\rEpoch: $epoch, Error: $currentError")
}