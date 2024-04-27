package main

import (
	"fmt"
	"math"

	gates "./data/gates"
	matrix "./libraries/Matrix"
	//"slices"
)

type SimpleModel struct {
	a0         matrix.SimpleMatrix
	w1, b1, a1 matrix.SimpleMatrix
	w2, b2, a2 matrix.SimpleMatrix
}

type ActualModel struct {
	count   int
	weights []matrix.SimpleMatrix
	biases  []matrix.SimpleMatrix
	active  []matrix.SimpleMatrix
}

func createNNModel(parSruct []int, parCount int) ActualModel {
	var newModel ActualModel
	newModel.count = parCount - 1 /*
		newModel.weights = make([]matrix.SimpleMatrix, newModel.count )
		newModel.biases = make([]matrix.SimpleMatrix, newModel.count )
		newModel.active = make([]matrix.SimpleMatrix, parCount)
	*/

	newModel.active = append(newModel.active, *matrix.NewMemMatrix(1, parSruct[0], parSruct[0]))
	for i := 1; i < parCount; i++ {
		newModel.weights = append(newModel.weights, *matrix.NewMemMatrix(newModel.active[i-1].Col, parSruct[i], parSruct[i]))
		newModel.biases = append(newModel.biases, *matrix.NewMemMatrix(1, parSruct[i], parSruct[i]))
		newModel.active = append(newModel.active, *matrix.NewMemMatrix(1, parSruct[i+1], parSruct[i+1]))
	}

	return newModel
}

func createModel() SimpleModel {
	var xorModel SimpleModel
	xorModel.a0 = (*matrix.NewMemMatrix(1, 2, 2))
	xorModel.w1 = (*matrix.NewMemMatrix(2, 2, 2))
	xorModel.b1 = (*matrix.NewMemMatrix(1, 2, 2))
	xorModel.a1 = (*matrix.NewMemMatrix(1, 2, 2))
	xorModel.w2 = (*matrix.NewMemMatrix(2, 1, 1))
	xorModel.b2 = (*matrix.NewMemMatrix(1, 1, 1))
	xorModel.a2 = (*matrix.NewMemMatrix(1, 1, 1))
	return xorModel
}

func sigmoid(parInputFloat float64) float64 {
	return float64(1) / (float64(1) + math.Exp(-parInputFloat))
}

func forward_xor_actual(xor ActualModel) {

	for h := 0; h < len(xor.weights); h++ {

		matrix.Mult(xor.active[h+1], xor.active[h], xor.weights[h+1])
		matrix.Sum(xor.active[h+1], xor.biases[h+1])
		matrix.ApplyFunction(xor.active[h+1], sigmoid)

		matrix.Mult(xor.active[h+2], xor.active[h+1], xor.weights[h+2])
		matrix.Sum(xor.active[h+2], xor.biases[h+2])
		matrix.ApplyFunction(xor.active[h+2], sigmoid)
	}
}

func cost_actual(m ActualModel, input matrix.SimpleMatrix, output matrix.SimpleMatrix) float64 {
	var c, d float64

	for i := 0; i < input.Row; i++ {
		matrix.Copy(m.active[0], *matrix.TakeRow(input, i))
		y := *matrix.TakeRow(output, i)
		forward_xor_actual(m)

		for t := 0; t < output.Col; t++ {
			d = (*m.active[len(m.active)-1].Values)[0][t] - (*y.Values)[0][t]
			c += d * d
		}
	}
	return c / float64(input.Row)
}

func finite_diff_actual(m ActualModel, g ActualModel, ti matrix.SimpleMatrix, to matrix.SimpleMatrix) {
	var saved float64
	c := cost_actual(m, ti, to)

	for h := 0; h < len(m.weights); h++ {

		for i := 0; i < m.weights[h].Row; i++ {
			for j := 0; j < m.weights[h].Col; j++ {
				saved = (*m.weights[h].Values)[i][j]
				(*m.weights[h].Values)[i][j] += gates.Eps
				(*g.weights[h].Values)[i][j] = (cost_actual(m, ti, to) - c) / gates.Eps
				(*m.weights[h].Values)[i][j] = saved
			}
		}
		for i := 0; i < m.biases[h].Row; i++ {
			for j := 0; j < m.biases[h].Col; j++ {
				saved = (*m.biases[h].Values)[i][j]
				(*m.biases[h].Values)[i][j] += gates.Eps
				(*g.biases[h].Values)[i][j] = (cost_actual(m, ti, to) - c) / gates.Eps
				(*m.biases[h].Values)[i][j] = saved
			}
		}
	}
}

func learn_actual(m ActualModel, g ActualModel) {

	for j := 0; j < len(m.weights); j++ {
		for i := 0; i < m.weights[j].Row; i++ {
			for h := 0; h < m.weights[j].Col; h++ {
				(*m.weights[h].Values)[i][h] -= gates.Rate * (*g.weights[h].Values)[i][h]
			}
		}
	}

	for j := 0; j < len(m.biases); j++ {
		for i := 0; i < m.biases[j].Row; i++ {
			for h := 0; h < m.biases[j].Col; h++ {
				(*m.biases[h].Values)[i][h] -= gates.Rate * (*g.biases[h].Values)[i][h]
			}
		}

	}

}

/*
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
				d = (*m.a2.Values)[0][t] - (*y.Values)[0][t]
				c += d * d
			}
		}
		return c / float64(input.Row)
	}

	func finite_diff(m SimpleModel, g SimpleModel, ti matrix.SimpleMatrix, to matrix.SimpleMatrix) {
		var saved float64
		c := cost(m, ti, to)

		for i := 0; i < m.w1.Row; i++ {
			for j := 0; j < m.w1.Col; j++ {
				saved = (*m.w1.Values)[i][j]
				(*m.w1.Values)[i][j] += gates.Eps
				(*g.w1.Values)[i][j] = (cost(m, ti, to) - c) / gates.Eps
				(*m.w1.Values)[i][j] = saved
			}
		}
		for i := 0; i < m.b1.Row; i++ {
			for j := 0; j < m.b1.Col; j++ {
				saved = (*m.b1.Values)[i][j]
				(*m.b1.Values)[i][j] += gates.Eps
				(*g.b1.Values)[i][j] = (cost(m, ti, to) - c) / gates.Eps
				(*m.b1.Values)[i][j] = saved
			}
		}
		for i := 0; i < m.w2.Row; i++ {
			for j := 0; j < m.w2.Col; j++ {
				saved = (*m.w2.Values)[i][j]
				(*m.w2.Values)[i][j] += gates.Eps
				(*g.w2.Values)[i][j] = (cost(m, ti, to) - c) / gates.Eps
				(*m.w2.Values)[i][j] = saved
			}
		}
		for i := 0; i < m.b2.Row; i++ {
			for j := 0; j < m.b2.Col; j++ {
				saved = (*m.b2.Values)[i][j]
				(*m.b2.Values)[i][j] += gates.Eps
				(*g.b2.Values)[i][j] = (cost(m, ti, to) - c) / gates.Eps
				(*m.b2.Values)[i][j] = saved
			}
		}
	}

	func learn(m SimpleModel, g SimpleModel) {
		for i := 0; i < m.w1.Row; i++ {
			for j := 0; j < m.w1.Col; j++ {
				(*m.w1.Values)[i][j] -= gates.Rate * (*g.w1.Values)[i][j]
			}
		}
		for i := 0; i < m.w2.Row; i++ {
			for j := 0; j < m.w2.Col; j++ {
				(*m.w2.Values)[i][j] -= gates.Rate * (*g.w2.Values)[i][j]
			}
		}
		for i := 0; i < m.b1.Row; i++ {
			for j := 0; j < m.b1.Col; j++ {
				(*m.b1.Values)[i][j] -= gates.Rate * (*g.b1.Values)[i][j]
			}
		}
		for i := 0; i < m.b2.Row; i++ {
			for j := 0; j < m.b2.Col; j++ {
				(*m.b2.Values)[i][j] -= gates.Rate * (*g.b2.Values)[i][j]
			}
		}
	}
*/
func main() {
	/*
		var a, b float64
	*/
	var y0 float64
	s := "----------------------------------------\n"
	ss := "%d | %d ---> %g\n"

	//var y0 float64

	var xorModel ActualModel
	/*
		xorModel.active = (*matrix.NewMemMatrix(1, 2, 2))
		xorModel.w1 = (*matrix.NewRandMatrix(2, 2))
		xorModel.b1 = (*matrix.NewRandMatrix(1, 2))
		xorModel.a1 = (*matrix.NewMemMatrix(1, 2, 2))
		xorModel.w2 = (*matrix.NewRandMatrix(2, 1))
		xorModel.b2 = (*matrix.NewRandMatrix(1, 1))
		xorModel.a2 = (*matrix.NewMemMatrix(1, 1, 1))
	*/
	gradModel := createNNModel()

	stride := 3
	n := len(gates.Training_OR)
	xormatrix := (*matrix.NewMemMatrix(n, 3, stride))
	xormatrix.Values = &gates.Training_OR
	ti := (*matrix.NewMemMatrix(n, 2, stride))
	ti.Values = &gates.Training_OR
	to := (*matrix.TakeColumn(xormatrix, 2))

	fmt.Println(cost_actual(xorModel, ti, to))

	for times := 0; times < gates.NumberOfLoops; times++ {
		finite_diff_actual(xorModel, gradModel, ti, to)
		learn_actual(xorModel, gradModel)
	}
	fmt.Println(cost_actual(xorModel, ti, to))

	fmt.Printf(s)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			(*xorModel.a0.Values)[0][0] = float64(i)
			(*xorModel.a0.Values)[0][1] = float64(j)
			forward_xor_actual(xorModel)
			y0 = (*xorModel.a2.Values)[0][0]
			fmt.Printf(ss, i, j, y0)
		}
	}

}
