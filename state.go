// Copyright 2023 The Zero Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/muesli/clusters"
	"github.com/muesli/kmeans"
	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

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

func (s *State) autoencode(dictionary map[string]string, words []string, vectors []float64) {
	const Width = Width + 2
	rnd, dropout := s.Rnd, s.Dropout
	_ = rnd
	other := tf32.NewSet()
	other.Add("words", Width, Length/2)
	w := other.ByName["words"]
	for _, w := range other.Weights {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		size := cap(w.X)
		_, _ = factor, size
		for i := 0; i < Length/2; i++ {
			for j := 0; j < Width-2; j++ {
				w.X = append(w.X, float32(vectors[i*(Width-2)+j]))
			}
			if i < Length/4 {
				w.X = append(w.X, 0, 1)
			} else {
				w.X = append(w.X, 1, 0)
			}
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}
	set := tf32.NewSet()
	set.Add("t", Width, Width)
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
	l2 := tf32.Mul(tf32.T(encoded), l1)
	cost := tf32.Avg(tf32.Quadratic(other.Get("words"), l2))

	i := 1
	pow := func(x float32) float32 {
		y := math.Pow(float64(x), float64(i))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return float32(y)
	}
	// The stochastic gradient descent loop
	for i < EpochsA {
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
			t.X[k] -= EtaA * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
		}

		// Housekeeping
		set.Zero()

		if math.IsNaN(float64(total)) {
			fmt.Println(total)
			break
		}
		fmt.Println(i, total)
		s.Points = append(s.Points, plotter.XY{X: float64(len(s.Points)), Y: float64(total)})
		i++
	}

	for i := 0; i < Length/4; i++ {
		w.X[i*Width+Width-2] = 1
		w.X[i*Width+Width-1] = 0
	}

	iencoded := tf32.Mul(set.Get("t"), other.Get("words"))
	il1 := spherical(tf32.Mul(encoded, encoded))

	var d clusters.Observations
	iencoded(func(a *tf32.V) bool {
		for i := 0; i < len(a.X); i += Width {
			c := clusters.Coordinates{}
			for j := 0; j < Width; j++ {
				c = append(c, float64(a.X[i+j]))
			}
			d = append(d, c)
		}
		return true
	})
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
					fmt.Printf("%d %s ", i, words[i])
					break
				}
			}
		}
		fmt.Printf("\n")
	}

	il1(func(a *tf32.V) bool {
		for i := 0; i < Length/2; i++ {
			max, index := float32(0.0), 0
			for j := 0; j < Length/2; j++ {
				if i == j {
					continue
				}
				a := a.X[i*(Length/2)+j]
				if a > max {
					max, index = a, j
				}
			}
			word := words[i]
			expected := dictionary[word]
			actual := words[index]
			if expected == actual {
				fmt.Println("correct", i, word, expected, actual)
			} else {
				fmt.Println(i, word, expected, actual)
			}
		}
		return true
	})
	return
}

func (s *State) sample(words []string, vectors []float64) []Entropy {
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

	var d clusters.Observations
	iencoded(func(a *tf32.V) bool {
		for i := 0; i < len(a.X); i += 2 * Width {
			c := clusters.Coordinates{}
			for j := 0; j < 2*Width; j++ {
				c = append(c, float64(a.X[i+j]))
			}
			d = append(d, c)
		}
		return true
	})
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
					fmt.Printf("%d %s ", i, words[i])
					break
				}
			}
		}
		fmt.Printf("\n")
	}

	return entropies
}

// Transform is transform mode
func Transform(dictionary map[string]string, words []string, vectors []float64) {
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
		e := state.sample(words, vectors)
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
