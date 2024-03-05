package gates

var rate float64 = 0.1
var eps float64 = 0.1
var numberOfLoops int = 1000000

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
