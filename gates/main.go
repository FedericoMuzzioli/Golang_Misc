package main

import (
	"math"

	matrix "./libraries/Matrix"
	//gates "./data/gates"
	//"slices"
)

type SimpleModel struct {
	a0         matrix.SimpleMatrix
	w1, b1, a1 matrix.SimpleMatrix
	w2, b2, a2 matrix.SimpleMatrix
}

func forward_xor(xor SimpleModel) {
	matrix.Mult(xor.a1, xor.a0, xor.w1)
	matrix.Sum(xor.a1, xor.b1)
	matrix.ApplyFunction(xor.a1, sigmoid)

	matrix.Mult(xor.a2, xor.a1, xor.w2)
	matrix.Sum(xor.a2, xor.b2)
	matrix.ApplyFunction(xor.a2, sigmoid)
}

func cost(m SimpleModel, input matrix.SimpleMatrix, output matrix.SimpleMatrix) float64 {
	var c, d float64

	for i := 0; i < input.Row; i++ {
		matrix.Copy(m.a0, *matrix.TakeRow(input, i))
		y := *matrix.TakeRow(output, i)
		forward_xor(m)

		for t := 0; t < output.Col; t++ {
			c = (*m.a2.Values)[0][t] - (*y.Values)[0][t]
			c += d * d
		}
	}
	return c / float64(input.Row)
}

/*
func simple_cost(parTrainingset *[][]float64, weight float64, bias float64) float64 {
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

func simple_twoimput_cost(parTrainingset *[][]float64, weight1 float64, weight2 float64, bias float64) float64 {
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

func simple_cost_Model(simpleModel *simpleModel, parTrainingset *[][]float64) float64 {
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
}*/

func sigmoid(parInputFloat float64) float64 {
	return float64(1) / (float64(1) + math.Exp(-parInputFloat))
}

/*
func forward(parModel *simpleModel, x1 float64, x2 float64) float64 {
	var a = sigmoid(x1*(parModel.weight[0]) + (parModel.weight2[0])*x2 + parModel.bias[0])
	var b = sigmoid(x1*(parModel.weight[1]) + (parModel.weight2[1])*x2 + parModel.bias[1])
	return sigmoid(a*(parModel.weight[2]) + (parModel.weight2[2])*b + parModel.bias[2])
}*

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

/*
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

func trainMat(parTrainingset matrix.SimpleMatrix, weight1 *float64, weight2 *float64, bias *float64) {
	for i := 0; i < numberOfLoops; i++ {
		dummy := simple_twoimput_cost(parTrainingset.Values, *weight1, *weight2, *bias)
		dW := (simple_twoimput_cost(parTrainingset.Values, *weight1+eps, *weight2, *bias) - dummy) / eps
		dw2 := (simple_twoimput_cost(parTrainingset.Values, *weight1, *weight2+eps, *bias) - dummy) / eps
		dB := (simple_twoimput_cost(parTrainingset.Values, *weight1, *weight2, *bias+eps) - dummy) / eps

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
*/

func main() {
	/*
		var a, b float64
		s := "----------------------------------------\n"
		ss := "%d | %d ---> %g\n"
	*/
	//var y0 float64
	var xorModel SimpleModel
	xorModel.a0 = (*matrix.NewMemMatrix(1, 2))
	xorModel.w1 = (*matrix.NewRandMatrix(2, 2))
	xorModel.b1 = (*matrix.NewRandMatrix(1, 2))
	xorModel.a1 = (*matrix.NewMemMatrix(1, 2))
	xorModel.w2 = (*matrix.NewRandMatrix(2, 1))
	xorModel.b2 = (*matrix.NewRandMatrix(1, 1))
	xorModel.a2 = (*matrix.NewMemMatrix(1, 1))

	for i := 0; i < 2; i++ {

		for j := 0; j < 2; j++ {
			(*xorModel.a0.Values)[0][0] = float64(i)
			(*xorModel.a0.Values)[0][1] = float64(j)
			forward_xor(xorModel)
			//y0 = (*xorModel.a2.Values)[0][0]
		}
	}

}
