package matrix

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
type simpleMatrix struct {
	values *[][]float64
	col    int
	row    int
}

func NewMatrix(parValues [][]float64) *simpleMatrix {
	return &simpleMatrix{
		values: &parValues,
		row:    len(parValues),
		col:    len(parValues[0]),
	}
}

func sum(a simpleMatrix, b simpleMatrix) *simpleMatrix {

	result := &simpleMatrix{}
	resultVal := make([][]float64, a.row)
	for i := 0; i < a.col; i++ {
		resultVal[i] = make([]float64, a.col)
		for j := 0; j < a.row; j++ {
			resultVal[i][j] += (*a.values)[i][j] + (*b.values)[i][j]
		}
	}
	result.values = &resultVal
	return result
}

func mult(a simpleMatrix, b simpleMatrix) *simpleMatrix {

	result := &simpleMatrix{}
	resultVal := make([][]float64, a.row)
	for i := 0; i < a.row; i++ {

		resultVal[i] = make([]float64, b.col)

		for j := 0; j < b.col; j++ {

			for k := 0; k < a.col; k++ {
				resultVal[i][j] += (*a.values)[i][k] * (*b.values)[k][j]
			}
		}
	}
	result.values = &resultVal
	return result
}

/*
func main() {
	a := NewMatrix(aa)
	b := NewMatrix(bb)

	//fmt.Println((*sum(*a, *b)).values)
	fmt.Println((*mult(*b, *a)).values)

}
*/
