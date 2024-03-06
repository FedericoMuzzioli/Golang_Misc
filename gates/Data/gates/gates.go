package gates

var Rate float64 = 0.1
var Eps float64 = 0.1
var NumberOfLoops int = 10000

var Training_XOR = [][]float64{
	{0, 0, 0},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 0},
}

var Training_NAND = [][]float64{
	{0, 0, 1},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 0},
}

var Training_OR = [][]float64{
	{0, 0, 0},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 1},
}

var Training_AND = [][]float64{
	{0, 0, 0},
	{1, 0, 0},
	{0, 1, 0},
	{1, 1, 1},
}
