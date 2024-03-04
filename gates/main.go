package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
	//"slices"
)

var rate float64 = 0.1
var eps float64 = 0.1
var numberOfLoops int = 1000000

type simpleModel struct {
	weight, weight2, bias [3]float64
}

var training_XOR = [][3]float64{
	{0, 0, 0},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 0},
}

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
		d = dummy[2] - y
		result += d * d
	}
	return (result / float64(len(*parTrainingset)))
}

func simple_cost_Model(simpleModel *simpleModel, parTrainingset *[][3]float64) float64 {
	var result float64
	var x1, x2, y, d float64
	for _, dummy := range *parTrainingset {
		x1 = dummy[0]
		x2 = dummy[1]
		y = forward(simpleModel, x1, x2)
		d = y - dummy[2]
		result += d * d
	}
	return (result / float64(len(*parTrainingset)))
}

func sigmoid(parInputFloat float64) float64 {
	return float64(1) / (float64(1) + math.Exp(-parInputFloat))
}

func forward(parModel *simpleModel, x1 float64, x2 float64) float64 {
	var a = sigmoid(x1*(parModel.weight[0]) + (parModel.weight2[0])*x2 + parModel.bias[0])
	var b = sigmoid(x1*(parModel.weight[1]) + (parModel.weight2[1])*x2 + parModel.bias[1])
	return sigmoid(a*(parModel.weight[2]) + (parModel.weight2[2])*b + parModel.bias[2])
}

func finite_diff(parModel *simpleModel) *simpleModel {
	var c float64 = simple_cost_Model(parModel, &training_XOR)
	newModel := &simpleModel{}
	staging := parModel.weight
	stagin2 := parModel.weight2
	staginBias := parModel.bias

	for j := 0; j < 3; j++ {
		parModel.weight[j] += eps
		newModel.weight[j] = (simple_cost_Model(parModel, &training_XOR) - c) / eps
		parModel.weight[j] = staging[j]

		parModel.weight2[j] += eps
		newModel.weight2[j] = (simple_cost_Model(parModel, &training_XOR) - c) / eps
		parModel.weight2[j] = stagin2[j]

		parModel.bias[j] += eps
		newModel.bias[j] = (simple_cost_Model(parModel, &training_XOR) - c) / eps
		parModel.bias[j] = staginBias[j]
	}

	return newModel
}

func train(parTrainingset *[][3]float64, weight1 *float64, weight2 *float64, bias *float64) {
	for i := 0; i < numberOfLoops; i++ {
		dummy := simple_twoimput_cost(parTrainingset, *weight1, *weight2, *bias)
		dW := (simple_twoimput_cost(parTrainingset, *weight1+eps, *weight2, *bias) - dummy) / eps
		dw2 := (simple_twoimput_cost(parTrainingset, *weight1, *weight2+eps, *bias) - dummy) / eps
		dB := (simple_twoimput_cost(parTrainingset, *weight1, *weight2, *bias+eps) - dummy) / eps

		*weight1 -= dW * rate
		*weight2 -= dw2 * rate
		*bias -= dB * rate
	}

}

func trainModel(parModel *simpleModel) *simpleModel {
	gradient := &simpleModel{}
	for i := 0; i < numberOfLoops; i++ {

		gradient = finite_diff(parModel)
		for j := 0; j < 3; j++ {
			parModel.weight[j] -= gradient.weight[j] * rate
			parModel.weight2[j] -= gradient.weight2[j] * rate
			parModel.bias[j] -= gradient.bias[j] * rate
		}
	}
	return parModel

}

func newRandModel() *simpleModel {
	modelPtr := &simpleModel{}
	for h := 0; h < 3; h++ {
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		*&modelPtr.weight[h] = r.Float64()
		*&modelPtr.weight2[h] = r.Float64()
		*&modelPtr.bias[h] = r.Float64()
	}

	return modelPtr

}

func main() {

	var a, b float64
	s := "----------------------------------------\n"
	ss := "%d | %d ---> %g\n"

	modelPrt := newRandModel()

	//var currOperation *[][3]float64
	//currOperation = &training_NAND

	train(&training_AND, &modelPrt.weight[0], &modelPrt.weight2[0], &modelPrt.bias[0])
	train(&training_NAND, &modelPrt.weight[1], &modelPrt.weight2[1], &modelPrt.bias[1])
	train(&training_OR, &modelPrt.weight[2], &modelPrt.weight2[2], &modelPrt.bias[2])

	for h := 0; h < 2; h++ {
		for t := 0; t < 2; t++ {
			fmt.Printf(ss, h, t, sigmoid(float64(t)*(modelPrt.weight[0])+(modelPrt.weight2[0])*float64(h)+modelPrt.bias[0]))
		}
	}
	fmt.Print(s)
	for h := 0; h < 2; h++ {
		for t := 0; t < 2; t++ {
			fmt.Printf(ss, h, t, sigmoid(float64(t)*(modelPrt.weight[1])+(modelPrt.weight2[1])*float64(h)+modelPrt.bias[1]))
		}
	}
	fmt.Print(s)
	for h := 0; h < 2; h++ {
		for t := 0; t < 2; t++ {
			fmt.Printf(ss, h, t, sigmoid(float64(t)*(modelPrt.weight[2])+(modelPrt.weight2[2])*float64(h)+modelPrt.bias[2]))
		}
	}
	fmt.Print(s)
	for h := 0; h < 2; h++ {
		for t := 0; t < 2; t++ {
			a = sigmoid(float64(t)*(modelPrt.weight[2]) + (modelPrt.weight2[2])*float64(h) + modelPrt.bias[2])
			b = sigmoid(float64(t)*(modelPrt.weight[1]) + (modelPrt.weight2[1])*float64(h) + modelPrt.bias[1])

			fmt.Printf(ss, h, t, sigmoid(a*(modelPrt.weight[0])+(modelPrt.weight2[0])*b+modelPrt.bias[0]))
		}
	}
	fmt.Print(s)
	modelPrt2 := newRandModel()
	modelPrt2 = trainModel(modelPrt2)
	for h := 0; h < 2; h++ {
		for t := 0; t < 2; t++ {
			fmt.Printf(ss, h, t, forward(modelPrt2, float64(t), float64(h)))
		}
	}

}
