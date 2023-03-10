// Copyright 2023 The Zero Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"compress/gzip"
	"encoding/gob"
	"flag"
	"fmt"
	"io"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tc128"
	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/pagerank"
)

// TODO: don't use random values in the attention network
// TODO: use kmeans on output of attention network

const (
	// Size is the size of the square matrix
	Size = 10
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.9
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.999
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
	// Eta is the learning rate
	Eta = .001
	// Epochs is the number of epochs
	Epochs = 64
	// EtaT is the learning rate for transform based model
	EtaT = .0001
	// EpochsT is the number of epochs for transform based model
	EpochsT = 16 * 1024
	// Width is the width of the model
	Width = 300
	// Length is the length of the model
	// 128
	// x 12.5
	// y 9.875
	// 64
	// x 6.8125
	// y 3.625
	// 32
	// x 3.375
	// y 3.5
	Length = 128
	// Offset is the offset for the parameters to learn
	Offset = Width * Length / 2
	// Words is the number of words per language
	Words = Length / 4
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// SphericalSoftmax is the spherical softmax function
// https://arxiv.org/abs/1511.05042
func SphericalSoftmax(k tc128.Continuation, node int, a *tc128.V, options ...map[string]interface{}) bool {
	const E = complex(0, 0)
	c, size, width := tc128.NewV(a.S...), len(a.X), a.S[0]
	values, sums, row := make([]complex128, width), make([]complex128, a.S[1]), 0
	for i := 0; i < size; i += width {
		sum := complex(0, 0)
		for j, ax := range a.X[i : i+width] {
			values[j] = ax*ax + E
			sum += values[j]
		}
		for _, cx := range values {
			c.X = append(c.X, (cx+E)/sum)
		}
		sums[row] = sum
		row++
	}
	if k(&c) {
		return true
	}
	// (2 a (b^2 + c^2 + d^2 + 0.003))/(a^2 + b^2 + c^2 + d^2 + 0.004)^2
	for i, d := range c.D {
		ax, sum := a.X[i], sums[i/width]
		//a.D[i] += d*(2*ax*(sum-(ax*ax+E)))/(sum*sum) - d*cx*2*ax/sum
		a.D[i] += d * (2 * ax * (sum - (ax*ax + E))) / (sum * sum)
	}
	return false
}

func SphericalSoftmaxReal(k tf32.Continuation, node int, a *tf32.V, options ...map[string]interface{}) bool {
	const E = 0
	c, size, width := tf32.NewV(a.S...), len(a.X), a.S[0]
	values, sums, row := make([]float32, width), make([]float32, a.S[1]), 0
	for i := 0; i < size; i += width {
		sum := float32(0)
		for j, ax := range a.X[i : i+width] {
			values[j] = ax*ax + E
			sum += values[j]
		}
		for _, cx := range values {
			c.X = append(c.X, (cx+E)/sum)
		}
		sums[row] = sum
		row++
	}
	if k(&c) {
		return true
	}
	// (2 a (b^2 + c^2 + d^2 + 0.003))/(a^2 + b^2 + c^2 + d^2 + 0.004)^2
	for i, d := range c.D {
		ax, sum := a.X[i], sums[i/width]
		//a.D[i] += d*(2*ax*(sum-(ax*ax+E)))/(sum*sum) - d*cx*2*ax/sum
		a.D[i] += d * (2 * ax * (sum - (ax*ax + E))) / (sum * sum)
	}
	return false
}

// Uses page rank to do zero shot learning
func Rank() {
	data := []float64{
		0, 1, 0, 1, 1, 0, 0, 0, 0, 0,
		1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
		0, 1, 0, 1, 1, 0, 0, 0, 0, 0,
		1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
		0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
		0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
		0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
		0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
	}
	adjacency := mat.NewDense(Size, Size, data)
	var eig mat.Eigen
	ok := eig.Factorize(adjacency, mat.EigenRight)
	if !ok {
		panic("Eigendecomposition failed")
	}

	values := eig.Values(nil)
	for i, value := range values {
		fmt.Println(i, value, cmplx.Abs(value), cmplx.Phase(value))
	}
	fmt.Printf("\n")

	vectors := mat.CDense{}
	eig.VectorsTo(&vectors)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			fmt.Printf("%f ", vectors.At(i, j))
		}
		fmt.Printf("\n")
	}
	fmt.Printf("\n")

	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			fmt.Printf("(%f, %f) ", cmplx.Abs(vectors.At(i, j)), cmplx.Phase(vectors.At(i, j)))
		}
		fmt.Printf("\n")
	}
	fmt.Printf("\n")

	type Rank struct {
		Rank float64
		Node int
	}
	merged := make(map[int]bool)
	graph, ranks := pagerank.NewGraph64(), make([]Rank, Size)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			graph.Link(uint64(i), uint64(j), adjacency.At(i, j))
		}
	}
	graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
		ranks[node] = Rank{
			Rank: rank,
			Node: int(node),
		}
	})
	sort.Slice(ranks, func(i, j int) bool {
		return ranks[i].Rank > ranks[j].Rank
	})
	for {
		a, b, found := -1, -1, 0
		for i := 0; i < Size; i++ {
			if !merged[ranks[i].Node] {
				fmt.Println(ranks[i].Node)
				if ranks[i].Node < 5 {
					if a == -1 {
						a = ranks[i].Node
						found++
					}
				} else {
					if b == -1 {
						b = ranks[i].Node
						found++
					}
				}
			}
			if found == 2 {
				break
			}
		}
		if found != 2 {
			break
		}
		adjacency.Set(a, b, 1)
		adjacency.Set(b, a, 1)
		merged[a] = true
		merged[b] = true
		fmt.Println(ranks)
	}

	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			fmt.Printf(" %.0f", adjacency.At(i, j))
		}
		fmt.Printf("\n")
	}
	fmt.Printf("\n")
}

// Neural uses neural network to do zero shot learning
func Neural() {
	data := []float64{
		0, 1, 0, 1, 1, 0, 0, 0, 0, 0,
		1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
		0, 1, 0, 1, 1, 0, 0, 0, 0, 0,
		1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
		0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
		0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
		0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
		0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
	}

	set := tc128.NewSet()
	set.Add("A", Size, Size)
	a := set.ByName["A"]
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			forward, back := float64(data[i*Size+j]), float64(data[j*Size+i])
			// https://www.sciencedirect.com/science/article/pii/S0972860019300945
			a.X = append(a.X, complex((forward+back)/2, (forward-back)/2))
		}
	}

	x := tc128.NewV(Size)
	for i := 0; i < Size; i++ {
		x.X = append(x.X, complex(rand.NormFloat64()/Size, rand.NormFloat64()/Size))
	}

	deltas := make([]complex128, len(x.X))

	spherical := tc128.U(SphericalSoftmax)
	l1 := spherical(tc128.Mul(set.Get("A"), x.Meta()))
	cost := tc128.Avg(tc128.Quadratic(x.Meta(), l1))

	iterations := 128
	points := make(plotter.XYs, 0, iterations)
	phase := make(plotter.XYs, 0, iterations)
	alpha, eta := complex(.3, 0), complex(.3, 0)
	for i := 0; i < iterations; i++ {
		set.Zero()
		x.Zero()

		total := tc128.Gradient(cost).X[0]
		norm := complex128(0)
		for _, d := range x.D {
			norm += d * d
		}
		norm = cmplx.Sqrt(norm)
		scaling := complex(1, 0)
		if cmplx.Abs(norm) > 1 {
			scaling = 1 / norm
		}
		for l, d := range x.D {
			deltas[l] = alpha*deltas[l] - eta*d*scaling
			x.X[l] += deltas[l]
		}
		if cmplx.Abs(total) < 1e-6 {
			break
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(cmplx.Abs(total))})
		phase = append(phase, plotter.XY{X: float64(i), Y: float64(cmplx.Phase(total))})
		fmt.Println(i, cmplx.Abs(total))
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	scatter, err = plotter.NewScatter(phase)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
	if err != nil {
		panic(err)
	}

	type Result struct {
		Value complex128
		Index int
	}
	results := make([]Result, 0, 10)
	for i := 0; i < Size; i++ {
		value := x.X[i]
		fmt.Printf("%d %f %f\n", i, cmplx.Abs(value), cmplx.Phase(value))
		results = append(results, Result{
			Value: value,
			Index: i,
		})
	}
	sort.Slice(results, func(i, j int) bool {
		return cmplx.Abs(results[i].Value) > cmplx.Abs(results[j].Value)
	})
	fmt.Printf("\n")
	for _, result := range results {
		fmt.Printf("%d %f %f\n", result.Index, cmplx.Abs(result.Value), cmplx.Phase(result.Value))
	}
}

// Vector is a word vector
type Vector struct {
	Word   string
	Vector []float64
}

// Vectors is a set of word vectors
type Vectors struct {
	List       []Vector
	Dictionary map[string]Vector
}

// NewVectors creates a new word vector set
func NewVectors(file string) Vectors {
	vectors := Vectors{
		Dictionary: make(map[string]Vector),
	}
	in, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer in.Close()

	gzipReader, err := gzip.NewReader(in)
	if err != nil {
		panic(err)
	}
	reader := bufio.NewReader(gzipReader)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
		}
		parts := strings.Split(line, " ")
		values := make([]float64, 0, len(parts)-1)
		for _, v := range parts[1:] {
			n, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
			if err != nil {
				panic(err)
			}
			values = append(values, float64(n))
		}
		sum := 0.0
		for _, v := range values {
			sum += v * v
		}
		length := math.Sqrt(sum)
		for i, v := range values {
			values[i] = v / length
		}
		word := strings.ToLower(strings.TrimSpace(parts[0]))
		vector := Vector{
			Word:   word,
			Vector: values,
		}
		vectors.List = append(vectors.List, vector)
		vectors.Dictionary[word] = vector
		if len(vector.Vector) == 0 {
			fmt.Println(vector)
		}
	}
	return vectors
}

var (
	// FlagRank runs the program in page rank mode
	FlagRank = flag.Bool("rank", false, "page rank mode")
	// FlagNeural runs the program in neural mode
	FlagNeural = flag.Bool("neural", false, "neural mode")
	// FlagGradient is the gradient descent mode
	FlagGradient = flag.Bool("gradient", false, "gradient descent mode")
)

// Entropy is the output self entropy of the model
type Entropy struct {
	Index   int
	Entropy float32
}

func process(rnd *rand.Rand, iteration int, dictionary map[string]string, words []string, vectors []float64) ([]Entropy, []Entropy) {
	debug, err := os.Create(fmt.Sprintf("%d_output.txt", iteration))
	if err != nil {
		panic(err)
	}
	defer debug.Close()

	dropout := tf32.U(func(k tf32.Continuation, node int, a *tf32.V, options ...map[string]interface{}) bool {
		size, width := len(a.X), a.S[0]
		c, drops, factor := tf32.NewV(a.S...), make([]int, width), float32(1)/(1-.5)
		for i := range drops {
			if rnd.Float64() > .5 {
				drops[i] = 1
			}
		}
		c.X = c.X[:cap(c.X)]
		for i := 0; i < size; i += width {
			for j, ax := range a.X[i : i+width] {
				if drops[j] == 1 {
					c.X[i+j] = ax * factor
				}
			}
		}
		if k(&c) {
			return true
		}
		for i := 0; i < size; i += width {
			for j := range a.D[i : i+width] {
				if drops[j] == 1 {
					a.D[i+j] += c.D[i+j]
				}
			}
		}
		return false
	})

	set := tf32.NewSet()
	set.Add("words", Width, Length)
	w := set.ByName["words"]
	for _, w := range set.Weights {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		size := cap(w.X)
		for _, value := range vectors {
			w.X = append(w.X, float32(value))
		}
		for i := Offset; i < size; i++ {
			w.X = append(w.X, float32(rnd.NormFloat64()*factor))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}
	set.Add("inputs", Width, Length)
	inputs := set.ByName["inputs"]
	inputs.X = append(inputs.X, w.X...)
	inputs.States = make([][]float32, StateTotal)
	for i := range inputs.States {
		inputs.States[i] = make([]float32, len(inputs.X))
	}

	order := func() {
		buffer := make([]float32, Width)
		rand.Shuffle(Length/2, func(i, j int) {
			copy(buffer, w.X[i*Width:i*Width+Width])
			copy(w.X[i*Width:i*Width+Width], w.X[j*Width:j*Width+Width])
			copy(w.X[j*Width:j*Width+Width], buffer)
			copy(buffer, inputs.X[i*Width:i*Width+Width])
			copy(inputs.X[i*Width:i*Width+Width], inputs.X[j*Width:j*Width+Width])
			copy(inputs.X[j*Width:j*Width+Width], buffer)
			words[i], words[j] = words[j], words[i]
		})
	}
	_ = order
	//order()

	//spherical := tf32.U(SphericalSoftmaxReal)
	a := tf32.Mul(set.Get("words"), set.Get("inputs"))
	l1 := dropout(tf32.Softmax(a))
	aa := tf32.Mul(tf32.T(set.Get("words")), l1)
	l2 := tf32.Softmax(aa)
	b := tf32.Entropy(l2)
	cost := tf32.Avg(b)

	i := 1
	pow := func(x float32) float32 {
		y := math.Pow(float64(x), float64(i))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return float32(y)
	}
	points := make(plotter.XYs, 0, 8)
	// The stochastic gradient descent loop
	for i < Epochs+1 {
		start := time.Now()
		// Calculate the gradients
		total := tf32.Gradient(cost).X[0]

		// Update the point weights with the partial derivatives using adam
		b1, b2 := pow(B1), pow(B2)

		for k, d := range w.D[Offset:] {
			k += Offset
			g := d
			m := B1*w.States[StateM][k] + (1-B1)*g
			v := B2*w.States[StateV][k] + (1-B2)*g*g
			w.States[StateM][k] = m
			w.States[StateV][k] = v
			mhat := m / (1 - b1)
			vhat := v / (1 - b2)
			w.X[k] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
		}
		for k, d := range inputs.D[Offset:] {
			k += Offset
			g := d
			m := B1*inputs.States[StateM][k] + (1-B1)*g
			v := B2*inputs.States[StateV][k] + (1-B2)*g*g
			inputs.States[StateM][k] = m
			inputs.States[StateV][k] = v
			mhat := m / (1 - b1)
			vhat := v / (1 - b2)
			w.X[k] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
		}
		copy(inputs.X, w.X)

		// Housekeeping
		end := time.Since(start)
		fmt.Fprintln(debug, i, total, end)
		set.Zero()

		if math.IsNaN(float64(total)) {
			fmt.Println(total)
			break
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		i++
	}

	// Plot the cost
	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%d_cost.png", iteration))
	if err != nil {
		panic(err)
	}

	a = tf32.Mul(set.Get("words"), set.Get("inputs"))
	l1 = tf32.Softmax(a)
	aa = tf32.Mul(tf32.T(set.Get("words")), l1)
	l2 = tf32.Softmax(aa)
	b = tf32.Entropy(l2)
	cost = tf32.Avg(b)

	output, err := os.Create(fmt.Sprintf("%d_output.html", iteration))
	if err != nil {
		panic(err)
	}
	defer output.Close()

	type Graph struct {
		Value  float32
		Row    int
		Column int
	}

	graphs := make([]Graph, 0, 8)
	graphsA := make([]Graph, 0, 8)
	graphsB := make([]Graph, 0, 8)
	l1(func(a *tf32.V) bool {
		for i := Length / 2; i < Length; i++ {
			for j := Length / 2; j < Length; j++ {
				value := a.X[i*Length+j]
				graphs = append(graphs, Graph{
					Value:  value,
					Row:    i,
					Column: j,
				})
			}
		}
		sort.Slice(graphs, func(i, j int) bool {
			return graphs[i].Value > graphs[j].Value
		})

		first, second := graphs[0].Column, 0
		for _, graph := range graphs[1:] {
			if graph.Column != first {
				second = graph.Column
				break
			}
		}

		pairs := make(plotter.XYs, 0, 8)
		for i := Length / 2; i < Length; i++ {
			x, y := a.X[i*Length+first], a.X[i*Length+second]
			pairs = append(pairs, plotter.XY{X: float64(x), Y: float64(y)})
		}
		// Plot the cost
		p := plot.New()

		p.Title.Text = "x vs y"
		p.X.Label.Text = fmt.Sprintf("x %d", first)
		p.Y.Label.Text = fmt.Sprintf("y %d", second)

		scatter, err := plotter.NewScatter(pairs)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%d_pairs.png", iteration))
		if err != nil {
			panic(err)
		}

		for i := Length / 2; i < Length; i++ {
			for j := 0; j < Length/2; j++ {
				value := a.X[i*Length+j]
				graphsA = append(graphsA, Graph{
					Value:  value,
					Row:    i,
					Column: j,
				})
			}
		}
		sort.Slice(graphsA, func(i, j int) bool {
			return graphsA[i].Value > graphsA[j].Value
		})

		for i := 0; i < Length/2; i++ {
			for j := Length / 2; j < Length; j++ {
				value := a.X[i*Length+j]
				graphsB = append(graphsB, Graph{
					Value:  value,
					Row:    i,
					Column: j,
				})
			}
		}
		sort.Slice(graphsB, func(i, j int) bool {
			return graphsB[i].Value > graphsB[j].Value
		})
		for i, graph := range graphs {
			fmt.Fprintf(debug, "% 7.7f %2d %2d % 7.7f %2d %2d % 7.7f %2d %2d\n",
				graph.Value, graph.Row, graph.Column,
				graphsA[i].Value, graphsA[i].Row, graphsA[i].Column,
				graphsB[i].Value, graphsB[i].Row, graphsB[i].Column)
		}

		fmt.Fprintf(output, "<html>")
		fmt.Fprintf(output, "<head><title>Adjacency Matrix</title></head>")
		fmt.Fprintf(output, "<body>")
		fmt.Fprintf(output, `<style>
 table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
 }
</style>
`)
		fmt.Fprintf(output, "<table>\n")
		fmt.Fprintf(output, "<tr>\n")
		fmt.Fprintf(output, " <th></th>\n")
		for i := 0; i < Length; i++ {
			fmt.Fprintf(output, " <th>%d</th>\n", i)
		}
		fmt.Fprintf(output, "</tr>\n")
		for i := 0; i < Length; i++ {
			fmt.Fprintf(output, "<tr>\n")
			fmt.Fprintf(output, " <th>%d</th>\n", i)
			for j := 0; j < Length; j++ {
				fmt.Fprintf(output, " <td>%.7f</td>", a.X[i*Length+j])
			}
			fmt.Fprintf(output, "</tr>\n")
		}
		fmt.Fprintf(output, "</table>")
		fmt.Fprintf(output, "</body>")
		fmt.Fprintf(output, "</html>")
		return true
	})

	x, y := make([]Entropy, 0, 8), make([]Entropy, 0, 8)
	b(func(a *tf32.V) bool {
		for key, value := range a.X {
			if key < Length/2 {
				x = append(x, Entropy{
					Index:   key,
					Entropy: value,
				})
			} else {
				y = append(y, Entropy{
					Index:   key - Length/2,
					Entropy: value,
				})
			}
		}
		return true
	})
	sort.Slice(x, func(i, j int) bool {
		return x[i].Entropy > x[j].Entropy
	})
	sort.Slice(y, func(i, j int) bool {
		return y[i].Entropy > y[j].Entropy
	})
	for _, entropy := range x {
		word := words[entropy.Index]
		fmt.Fprintln(debug, word, dictionary[word], entropy.Entropy)
	}
	fmt.Fprintln(debug)
	for _, entropy := range y {
		word := words[entropy.Index]
		fmt.Fprintln(debug, word, dictionary[word], entropy.Entropy)
	}

	type Rank struct {
		Index int
		Rank  float32
	}
	ranks := make([]Rank, 0, 8)
	aa(func(a *tf32.V) bool {
		for i := Length / 2; i < Length; i++ {
			sum := float32(0)
			aa := float32(0)
			bb := float32(0)
			for j := 0; j < Width; j++ {
				a, b := a.X[(Length/2)*Width+j], a.X[i*Width+j]
				aa += a * a
				bb += b * b
				sum += a * b
			}
			ranks = append(ranks, Rank{
				Index: i - Length/2,
				Rank:  sum / (float32(math.Sqrt(float64(aa)) * math.Sqrt(float64(bb)))),
			})
		}
		return true
	})
	sort.Slice(ranks, func(i, j int) bool {
		return ranks[i].Rank > ranks[j].Rank
	})
	for _, rank := range ranks {
		fmt.Fprintln(debug, words[rank.Index], rank)
	}

	accuracy := func(x []Entropy) float64 {
		correctness := 0
		for i := 0; i < Length/2; i++ {
			start := words[x[i].Index]
			target := dictionary[start]
			for j := i + 1; j < Length/2; j++ {
				if words[x[j].Index] == target {
					correctness += j - i - 1
					break
				}
			}
		}
		return 2 * float64(correctness) / Length
	}
	fmt.Fprintln(debug, "input", accuracy(x))
	fmt.Fprintln(debug, "learned", accuracy(y))

	average, squared := 0.0, 0.0
	for i := 0; i < 256; i++ {
		w := make([]string, len(words))
		copy(w, words)
		rand.Shuffle(len(words), func(i, j int) {
			w[i], w[j] = w[j], w[i]
		})
		correctness := 0
		for i := 0; i < Length/2; i++ {
			start := w[i]
			target := dictionary[start]
			for j := i + 1; j < Length/2; j++ {
				if w[j] == target {
					correctness += j - i - 1
					break
				}
			}
		}
		x := 2 * float64(correctness) / Length
		average += x
		squared += x * x
	}
	average /= 256
	squared /= 256
	stddev := math.Sqrt(squared - average*average)
	fmt.Fprintln(debug, "-3 std", average-3*stddev)
	fmt.Fprintln(debug, "-2 std", average-2*stddev)
	fmt.Fprintln(debug, "-1 std", average-1*stddev)
	fmt.Fprintln(debug, "avg", average)
	fmt.Fprintln(debug, "+1 std", average+1*stddev)
	fmt.Fprintln(debug, "+2 std", average+2*stddev)
	fmt.Fprintln(debug, "+3 std", average+3*stddev)

	return x, y
}

// GradientDescent is the gradient descent model
func GradientDescent(dictionary map[string]string, words []string, vectors []float64) {
	type Word struct {
		Word  string
		Value float64
		Index int
	}
	project := func(words []string, vectors []float64) []Word {
		projection := make([]Word, 0, 8)
		dense := mat.NewDense(Words, Width, vectors)
		var pc stat.PC
		ok := pc.PrincipalComponents(dense, nil)
		if !ok {
			panic("PrincipalComponents failed")
		}
		k := 1
		var proj mat.Dense
		var vec mat.Dense
		pc.VectorsTo(&vec)
		proj.Mul(dense, vec.Slice(0, Width, 0, k))
		for i := 0; i < Words; i++ {
			projection = append(projection, Word{
				Word:  words[i],
				Value: proj.At(i, 0),
				Index: i,
			})
		}
		sort.Slice(projection, func(i, j int) bool {
			return projection[i].Value < projection[j].Value
		})
		return projection
	}
	projectionEnglish := project(words[:Words], vectors[:Words*Width])
	projectionGerman := project(words[Words:], vectors[Words*Width:2*Words*Width])
	vectorsNew := make([]float64, len(vectors))
	wordsNew := make([]string, len(words))
	for i, value := range projectionEnglish {
		copy(vectorsNew[Width*i:Width*i+Width], vectors[Width*value.Index:Width*value.Index+Width])
		wordsNew[i] = words[value.Index]
	}
	for i, value := range projectionGerman {
		i += Words
		value.Index += Words
		copy(vectorsNew[Width*i:Width*i+Width], vectors[Width*value.Index:Width*value.Index+Width])
		wordsNew[i] = words[value.Index]
	}
	words, vectors = wordsNew, vectorsNew

	rnd := rand.New(rand.NewSource(1))

	for i := 0; i < 3; i++ {
		_, y := process(rnd, i, dictionary, words, vectors)

		vectorsNew := make([]float64, len(vectors))
		wordsNew := make([]string, len(words))
		for i, value := range y {
			copy(vectorsNew[Width*i:Width*i+Width], vectors[Width*value.Index:Width*value.Index+Width])
			wordsNew[i] = words[value.Index]
		}
		words, vectors = wordsNew, vectorsNew
	}
	return
}

// State is the state of the sampler
type State struct {
	Points  plotter.XYs
	Rnd     *rand.Rand
	Dropout func(a tf32.Meta, options ...map[string]interface{}) tf32.Meta
}

// NewState creates a new State
func NewState() State {
	rnd := rand.New(rand.NewSource(1))
	dropout := tf32.U(func(k tf32.Continuation, node int, a *tf32.V, options ...map[string]interface{}) bool {
		size, width := len(a.X), a.S[0]
		c, drops, factor := tf32.NewV(a.S...), make([]int, width), float32(1)/(1-.1)
		for i := range drops {
			if rnd.Float64() > .1 {
				drops[i] = 1
			}
		}
		c.X = c.X[:cap(c.X)]
		for i := 0; i < size; i += width {
			for j, ax := range a.X[i : i+width] {
				if drops[j] == 1 {
					c.X[i+j] = ax * factor
				}
			}
		}
		if k(&c) {
			return true
		}
		for i := 0; i < size; i += width {
			for j := range a.D[i : i+width] {
				if drops[j] == 1 {
					a.D[i+j] += c.D[i+j]
				}
			}
		}
		return false
	})

	return State{
		Points:  make(plotter.XYs, 0, 8),
		Rnd:     rnd,
		Dropout: dropout,
	}
}

func (s *State) sample(vectors []float64) []Entropy {
	rnd, dropout := s.Rnd, s.Dropout
	_ = rnd
	other := tf32.NewSet()
	other.Add("words", Width, Length/2)
	for _, w := range other.Weights {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		size := cap(w.X)
		_, _ = factor, size
		for _, value := range vectors {
			w.X = append(w.X, float32(value))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}
	set := tf32.NewSet()
	set.Add("t", Width, 2*Width)
	t := set.ByName["t"]
	for _, w := range set.Weights {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		size := cap(w.X)
		for i := 0; i < size; i++ {
			w.X = append(w.X, float32(rnd.NormFloat64()*factor))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}

	spherical := tf32.U(SphericalSoftmaxReal)
	encoded := tf32.Mul(set.Get("t"), other.Get("words"))
	l1 := dropout(spherical(tf32.Mul(encoded, encoded)))
	l2 := spherical(tf32.Mul(tf32.T(encoded), l1))
	cost := tf32.Avg(tf32.Entropy(l2))

	i := 1
	pow := func(x float32) float32 {
		y := math.Pow(float64(x), float64(i))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return float32(y)
	}
	// The stochastic gradient descent loop
	for i < EpochsT {
		// Calculate the gradients
		total := tf32.Gradient(cost).X[0]

		// Update the point weights with the partial derivatives using adam
		b1, b2 := pow(B1), pow(B2)

		for k, d := range t.D {
			g := d
			m := B1*t.States[StateM][k] + (1-B1)*g
			v := B2*t.States[StateV][k] + (1-B2)*g*g
			t.States[StateM][k] = m
			t.States[StateV][k] = v
			mhat := m / (1 - b1)
			vhat := v / (1 - b2)
			t.X[k] -= EtaT * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
		}

		// Housekeeping
		set.Zero()

		if math.IsNaN(float64(total)) {
			fmt.Println(total)
			break
		}

		s.Points = append(s.Points, plotter.XY{X: float64(len(s.Points)), Y: float64(total)})
		i++
	}

	iencoded := tf32.Mul(set.Get("t"), other.Get("words"))
	il1 := spherical(tf32.Mul(iencoded, iencoded))
	il2 := spherical(tf32.Mul(tf32.T(iencoded), il1))
	e := tf32.Entropy(il2)

	entropies := make([]Entropy, 0, 8)
	e(func(a *tf32.V) bool {
		for key, value := range a.X {
			entropies = append(entropies, Entropy{
				Index:   key,
				Entropy: value,
			})
		}
		return true
	})
	sort.Slice(entropies, func(i, j int) bool {
		return entropies[i].Entropy < entropies[j].Entropy
	})
	return entropies
}

func main() {
	rand.Seed(1)
	flag.Parse()

	if *FlagRank {
		Rank()
		return
	} else if *FlagNeural {
		Neural()
		return
	}

	vectors := []float64{}
	_, err := os.Stat("vectors.gob")
	wordsEnglish := []string{
		"dog",
		"cat",
		"bird",
		"horse",

		"chicken",
		"lamb",
		"pig",
		"cow",

		"spoon",
		"fork",
		"cup",
		"plate",

		"car",
		"bus",
		"scooter",
		"bike",

		"house",
		"door",
		"window",
		"floor",

		"shovel",
		"hoe",
		"plow",
		"axe",

		"pen",
		"pencil",
		"brush",
		"crayon",

		"chair",
		"bed",
		"table",
		"dresser",
	}
	wordsGerman := []string{
		"hund",
		"katze",
		"vogel",
		"pferd",

		"huhn",
		"lamm",
		"schwein",
		"kuh",

		"löffel",
		"gabel",
		"tasse",
		"platte",

		"auto",
		"bus",
		"roller",
		"fahrrad",

		"haus",
		"tür",
		"fenster",
		"boden",

		"schaufel",
		"hacke",
		"pflug",
		"axt",

		"stift",
		"bleistift",
		"bürste",
		"wachsmalstift",

		"stuhl",
		"bett",
		"tisch",
		"kommode",
	}
	dictionary := make(map[string]string)
	for i, english := range wordsEnglish[:Words] {
		german := wordsGerman[i]
		dictionary[english] = german
		dictionary[german] = english
	}
	words := make([]string, 0, len(wordsEnglish)+len(wordsGerman))
	words = append(words, wordsEnglish[:Words]...)
	words = append(words, wordsGerman[:Words]...)
	if err != nil {
		english := NewVectors("cc.en.300.vec.gz")
		german := NewVectors("cc.de.300.vec.gz")

		for _, word := range wordsEnglish[:Words] {
			vector := english.Dictionary[word]
			if len(vector.Vector) == 0 {
				panic(word)
			}
			vectors = append(vectors, vector.Vector...)
		}
		for _, word := range wordsGerman[:Words] {
			vector := german.Dictionary[word]
			if len(vector.Vector) == 0 {
				panic(word)
			}
			vectors = append(vectors, vector.Vector...)
		}
		output, err := os.Create("vectors.gob")
		if err != nil {
			panic(err)
		}
		encoder := gob.NewEncoder(output)
		err = encoder.Encode(vectors)
		if err != nil {
			panic(err)
		}
	} else {
		input, err := os.Open("vectors.gob")
		if err != nil {
			panic(err)
		}
		decoder := gob.NewDecoder(input)
		err = decoder.Decode(&vectors)
		if err != nil {
			panic(err)
		}
	}

	for i := 0; i < Length/2; i++ {
		sum := 0.0
		for j := 0; j < Width; j++ {
			a := vectors[i*Width+j]
			sum += a * a
		}
		length := math.Sqrt(sum)
		for j := 0; j < Width; j++ {
			vectors[i*Width+j] /= length
		}
	}

	if *FlagGradient {
		GradientDescent(dictionary, words, vectors)
		return
	}

	statistics := make([][]int, len(words))
	for i := range statistics {
		statistics[i] = make([]int, len(words))
	}

	accuracy := func(x []Entropy) float64 {
		correctness := 0
		for i := 0; i < Length/2; i++ {
			start := words[x[i].Index]
			target := dictionary[start]
			for j := i + 1; j < Length/2; j++ {
				if words[x[j].Index] == target {
					correctness += j - i - 1
					break
				}
			}
		}
		return 2 * float64(correctness) / Length
	}

	state := NewState()
	for i := 0; i < 1; i++ {
		e := state.sample(vectors)
		for _, value := range e {
			fmt.Println(value.Entropy, words[value.Index], dictionary[words[value.Index]])
		}
		fmt.Println(accuracy(e))
		for j, value := range e {
			if j > 2 {
				statistics[value.Index][e[j-2].Index]++
			}
			if j > 1 {
				statistics[value.Index][e[j-1].Index]++
			}
			if j < len(words)-1 {
				statistics[value.Index][e[j+1].Index]++
			}
			if j < len(words)-2 {
				statistics[value.Index][e[j+2].Index]++
			}

		}
	}
	fmt.Println(words[0])
	for i, value := range statistics[0] {
		fmt.Println(value, words[i], dictionary[words[i]])
	}

	// Plot the cost
	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(state.Points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
	if err != nil {
		panic(err)
	}
}
