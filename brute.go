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

	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func compress(rnd *rand.Rand, name string, vectors []float64) []float64 {
	debug, err := os.Create(fmt.Sprintf("%s_output.txt", name))
	if err != nil {
		panic(err)
	}
	defer debug.Close()

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

	set := tf32.NewSet()
	set.Add("words", Width, Length/2)
	w := set.ByName["words"]
	for _, w := range set.Weights {
		for _, value := range vectors {
			w.X = append(w.X, float32(value))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}
	set.Add("inputs", Width, Length/2)
	inputs := set.ByName["inputs"]
	inputs.X = append(inputs.X, w.X...)
	inputs.States = make([][]float32, StateTotal)
	for i := range inputs.States {
		inputs.States[i] = make([]float32, len(inputs.X))
	}

	spherical := tf32.U(SphericalSoftmaxReal)
	l1 := dropout(spherical(tf32.Mul(set.Get("words"), set.Get("inputs"))))
	l2 := spherical(tf32.Mul(tf32.T(set.Get("words")), l1))
	cost := tf32.Avg(tf32.Entropy(l2))

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

		for k, d := range w.D {
			g := d
			m := B1*w.States[StateM][k] + (1-B1)*g
			v := B2*w.States[StateV][k] + (1-B2)*g*g
			w.States[StateM][k] = m
			w.States[StateV][k] = v
			mhat := m / (1 - b1)
			vhat := v / (1 - b2)
			w.X[k] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
		}
		/*for k, d := range inputs.D {
			g := d
			m := B1*inputs.States[StateM][k] + (1-B1)*g
			v := B2*inputs.States[StateV][k] + (1-B2)*g*g
			inputs.States[StateM][k] = m
			inputs.States[StateV][k] = v
			mhat := m / (1 - b1)
			vhat := v / (1 - b2)
			w.X[k] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
		}
		copy(inputs.X, w.X)*/

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

	err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%s_cost.png", name))
	if err != nil {
		panic(err)
	}

	output := make([]float64, len(w.X))
	for key, value := range w.X {
		output[key] = float64(value)
	}
	return output
}

// Brute is the brute model
func Brute(dictionary map[string]string, wordsEnglish, wordsGerman, words []string, vectors []float64) {
	rnd := rand.New(rand.NewSource(1))
	vectors = compress(rnd, "vectors", vectors)

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

	length := len(wordsEnglish)
	englishVectors := vectors[:len(vectors)/2]
	germanVectors := vectors[len(vectors)/2:]

	type Match struct {
		Index int
		Value float64
	}

	match := func(w int, vectors []float64) []Match {
		matches := make([]Match, 0, 8)
		for i := 0; i < length; i++ {
			if i == w {
				continue
			}
			sum := 0.0
			for j := 0; j < Width; j++ {
				sum += vectors[w*Width+j] * vectors[i*Width+j]
			}
			matches = append(matches, Match{
				Index: i,
				Value: sum,
			})
		}
		values, sum := make([]float64, len(matches)), 0.0
		for j, ax := range matches {
			values[j] = ax.Value * ax.Value
			sum += values[j]
		}
		for j, cx := range values {
			matches[j].Value = cx / sum
		}
		sort.Slice(matches, func(i, j int) bool {
			return matches[i].Value > matches[j].Value
		})
		return matches
	}
	matchWord := func(w int) {
		fmt.Println(wordsEnglish[w], wordsGerman[w])

		matchesEnglish := match(w, englishVectors)
		matchesGerman := match(w, germanVectors)
		for i, match := range matchesEnglish {
			matchGerman := matchesGerman[i]
			fmt.Println(wordsEnglish[match.Index], wordsGerman[matchGerman.Index], matchGerman.Value, match.Value)
		}
		fmt.Println()
	}

	matchWord(0)
	matchWord(12)
	matchWord(27)

	type Cost struct {
		Index int
		Value float64
	}
	brute := func(w int) {
		costs := make([]Cost, 0, 8)
		eng := match(w, englishVectors)
		fmt.Println(wordsEnglish[w], wordsGerman[w])
		for i := 0; i < length; i++ {
			deu := match(i, germanVectors)
			cost := 0.0
			for j, value := range deu {
				diff := value.Value - eng[j].Value
				cost += diff * diff
			}
			costs = append(costs, Cost{
				Index: i,
				Value: cost,
			})
		}
		sort.Slice(costs, func(i, j int) bool {
			return costs[i].Value > costs[j].Value
		})
		for _, value := range costs {
			fmt.Println(value.Index, wordsEnglish[value.Index], wordsGerman[value.Index], value.Value)
		}
		fmt.Println()
	}
	brute(0)
	brute(12)
	brute(27)
}
