package matrix

import (
	"fmt"
	"math/rand"
	//"time"
)

//"math"
//"math/rand"
//"time"
//"slices"
/*
var aa = [][]float64{
	{1, 0},
	{0, 3},
}

var bb = [][]float64{
	{4, 1},
	{-2, 2},
}
*/
type SimpleMatrix struct {
	Values *[][]float64
	Col    int
	Row    int
	stride int
}

func NewMatrix(parValues [][]float64) *SimpleMatrix {
	return &SimpleMatrix{
		Values: &parValues,
		Row:    len(parValues),
		Col:    len(parValues[0]),
	}
}

func NewRandMatrix(parRows int, parColumns int) *SimpleMatrix {
	resultVal := make([][]float64, parRows)
	for i := 0; i < parRows; i++ {
		resultVal[i] = make([]float64, parColumns)

		for j := 0; j < parColumns; j++ {
			//r := rand.New(rand.NewSource(time.Now().UnixNano()))
			resultVal[i][j] = rand.Float64()
		}
	}
	return &SimpleMatrix{
		Values: &resultVal,
		Row:    parRows,
		Col:    parColumns,
		stride: parColumns,
	}
}

func NewMemMatrix(parRows int, parColumns int, stride int) *SimpleMatrix {
	resultVal := make([][]float64, parRows)
	for i := 0; i < parRows; i++ {
		resultVal[i] = make([]float64, parColumns)
	}
	return &SimpleMatrix{
		Values: &resultVal,
		Row:    parRows,
		Col:    parColumns,
		stride: stride,
	}
}

func Sum(result SimpleMatrix, b SimpleMatrix) {
	for i := 0; i < result.Row; i++ {
		for j := 0; j < result.Col; j++ {
			(*result.Values)[i][j] += (*b.Values)[i][j]
		}
	}
}

func Mult(result SimpleMatrix, a SimpleMatrix, b SimpleMatrix) {

	for i := 0; i < a.Row; i++ {
		for j := 0; j < b.Col; j++ {
			(*result.Values)[i][j] = 0
			for k := 0; k < a.Col; k++ {
				(*result.Values)[i][j] += (*a.Values)[i][k] * (*b.Values)[k][j]
			}
		}
	}
}

func ApplyFunction(result SimpleMatrix, toApply func(par float64) float64) {
	for i := 0; i < result.Row; i++ {
		for j := 0; j < result.Col; j++ {
			(*result.Values)[i][j] = toApply((*result.Values)[i][j])
		}
	}
}

func Print(a SimpleMatrix) {
	fmt.Println("[")
	ss := "%g    "
	for i := 0; i < a.Row; i++ {
		for j := 0; j < a.Col; j++ {
			fmt.Printf(ss, (*a.Values)[i][j])
		}
		fmt.Print("\n")
	}
	fmt.Print("]")

}

func TakeRow(a SimpleMatrix, parRow int) *SimpleMatrix {
	result := NewMemMatrix(1, a.Col, a.stride)
	for j := 0; j < a.Col; j++ {
		(*result.Values)[0][j] = (*a.Values)[parRow][j]
	}
	return result
}

func TakeColumn(a SimpleMatrix, parColumn int) *SimpleMatrix {
	result := NewMemMatrix(a.Row, 1, a.stride)
	for j := 0; j < a.Row; j++ {
		(*result.Values)[j][0] = (*a.Values)[j][parColumn]
	}
	return result
}

func Copy(a SimpleMatrix, b SimpleMatrix) {

	for i := 0; i < a.Row; i++ {
		for j := 0; j < a.Col; j++ {
			(*a.Values)[i][j] = (*b.Values)[i][j]
		}
	}
}

/*
func main() {
	a := NewMatrix(aa)
	b := NewMatrix(bb)

	//fmt.Println((*sum(*a, *b)).Values)
	fmt.Println((*mult(*b, *a)).Values)

}
*/
