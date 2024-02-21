package main

import (
	"fmt"
	"math"
	"math/rand"
	//"time"
	//"slices"
)

//var r = rand.New(rand.NewSource(time.Now().UnixNano()))

var r = rand.New(rand.NewSource(100))

var training_NAND = [][3]float64{
	{0, 0, 1},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 0},
}

var training_OR = [][3]float64{
	{0, 0, 0},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 1},
}

var training_AND = [][3]float64{
	{0, 0, 0},
	{1, 0, 0},
	{0, 1, 0},
	{1, 1, 1},
}

func simple_cost(parTrainingset *[][3]float64, weight float64, bias float64) float64 {
	var result float64 = 0
	var x1, y, d float64
	for _, dummy := range *parTrainingset {
		x1 = dummy[0]
		y = x1 * weight
		d = y - dummy[1]
		result += math.Pow(d, 2)
	}
	return result / float64(len(*parTrainingset))
}

func simple_twoimput_cost(parTrainingset *[][3]float64, weight1 float64, weight2 float64, bias float64) float64 {
	var result float64
	var x1, x2, y, d float64
	for _, dummy := range *parTrainingset {
		x1 = dummy[0]
		x2 = dummy[1]
		y = sigmoid(x1*weight1 + x2*weight2 + bias)
		d = y - dummy[2]
		result += d * d
	}
	return (result / float64(len(*parTrainingset)))
}

func sigmoid(parInputFloat float64) float64 {
	return float64(1) / (float64(1) + math.Exp(-parInputFloat))
}

func train(parTrainingset *[][3]float64, weight1 *float64, weight2 *float64, bias *float64, eps float64, rate float64) {
	for i := 0; i < 200000; i++ {
		dummy := simple_twoimput_cost(parTrainingset, *weight1, *weight2, *bias)
		dW := (simple_twoimput_cost(parTrainingset, *weight1+eps, *weight2, *bias) - dummy) / eps
		dw2 := (simple_twoimput_cost(parTrainingset, *weight1, *weight2+eps, *bias) - dummy) / eps
		dB := (simple_twoimput_cost(parTrainingset, *weight1, *weight2, *bias+eps) - dummy) / eps

		*weight1 -= dW * rate
		*weight2 -= dw2 * rate
		*bias -= dB * rate
	}

}

func main() {
	var weight, weight2 float64
	var bias float64
	var rate float64 = 0.1
	var eps float64 = 0.1
	weight = r.Float64()
	weight2 = r.Float64()
	bias = r.Float64()

	var currOperation *[][3]float64
	currOperation = &training_NAND

	train(currOperation, &weight, &weight2, &bias, eps, rate)

	for h := 0; h < 2; h++ {
		for t := 0; t < 2; t++ {
			fmt.Println(sigmoid(float64(t)*weight + weight2*float64(h) + bias))
		}
	}

}
