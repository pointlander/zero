// Copyright 2023 The Zero Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"

	"github.com/muesli/clusters"
	"github.com/muesli/kmeans"
	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

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

	var d clusters.Observations
	/*l2(func(a *tf32.V) bool {
		for i := 0; i < len(a.X)/2; i += Width {
			c := clusters.Coordinates{}
			for j := 0; j < Width; j++ {
				c = append(c, float64(a.X[i+j+Length/2]))
			}
			d = append(d, c)
		}
		return true
	})*/
	for i := len(w.X) / 2; i < len(w.X); i += Width {
		c := clusters.Coordinates{}
		for j := 0; j < Width; j++ {
			c = append(c, float64(w.X[i+j]))
		}
		d = append(d, c)
	}
	km := kmeans.New()
	clusters, err := km.Partition(d, Words)
	if err != nil {
		panic(err)
	}

	for _, c := range clusters {
		for _, o := range c.Observations {
			q := o.Coordinates().Coordinates()
			for i, v := range d {
				same := true
				for j, x := range v.Coordinates().Coordinates() {
					if x != q[j] {
						same = false
						break
					}
				}
				if same {
					fmt.Fprintf(debug, "%d %s ", i, words[i])
					break
				}
			}
		}
		fmt.Fprintf(debug, "\n")
	}
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
