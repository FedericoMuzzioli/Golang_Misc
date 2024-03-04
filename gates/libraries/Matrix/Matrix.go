package matrix

import (
	"math/rand"
	"time"
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
	col    int
	row    int
}

func NewMatrix(parValues [][]float64) *SimpleMatrix {
	return &SimpleMatrix{
		Values: &parValues,
		row:    len(parValues),
		col:    len(parValues[0]),
	}
}

func NewRandMatrix(parRows int, parColumns int) *SimpleMatrix {
	resultVal := make([][]float64, parRows)
	for i := 0; i < parRows; i++ {
		resultVal[i] = make([]float64, parColumns)

		for j := 0; j < parColumns; j++ {
			r := rand.New(rand.NewSource(time.Now().UnixNano()))
			resultVal[i][j] = r.Float64()
			resultVal[i][j] = r.Float64()
			resultVal[i][j] = r.Float64()
		}
	}
	return &SimpleMatrix{
		Values: &resultVal,
		row:    parRows,
		col:    parColumns,
	}
}

func sum(a SimpleMatrix, b SimpleMatrix) *SimpleMatrix {

	result := &SimpleMatrix{}
	resultVal := make([][]float64, a.row)
	for i := 0; i < a.col; i++ {
		resultVal[i] = make([]float64, a.col)
		for j := 0; j < a.row; j++ {
			resultVal[i][j] += (*a.Values)[i][j] + (*b.Values)[i][j]
		}
	}
	result.Values = &resultVal
	return result
}

func mult(a SimpleMatrix, b SimpleMatrix) *SimpleMatrix {

	result := &SimpleMatrix{}
	resultVal := make([][]float64, a.row)
	for i := 0; i < a.row; i++ {

		resultVal[i] = make([]float64, b.col)

		for j := 0; j < b.col; j++ {

			for k := 0; k < a.col; k++ {
				resultVal[i][j] += (*a.Values)[i][k] * (*b.Values)[k][j]
			}
		}
	}
	result.Values = &resultVal
	return result
}

/*
func main() {
	a := NewMatrix(aa)
	b := NewMatrix(bb)

	//fmt.Println((*sum(*a, *b)).Values)
	fmt.Println((*mult(*b, *a)).Values)

}
*/
