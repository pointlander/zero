// Copyright 2023 The Zero Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"sort"
	"time"

	"github.com/mjibson/go-dsp/fft"
	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// Genome is a genome
type Genome struct {
	Words   []string
	Vectors []float64
	Fit     float64
}

// Swap swaps two genomes
func (g Genome) Swap(i, j int) {
	g.Words[i], g.Words[j] = g.Words[j], g.Words[i]
	buffer, a, b := make([]float64, Width), g.Vectors[i*Width:i*Width+Width], g.Vectors[j*Width:j*Width+Width]
	copy(buffer, a)
	copy(a, b)
	copy(b, buffer)
}

// Move moves a genomes
func (g Genome) Move(i, j int) {
	words, vectors := make([]string, 0, len(g.Words)), make([]float64, 0, len(g.Vectors))
	a, b := g.Words[i], g.Vectors[i*Width:i*Width+Width]
	for x, value := range g.Words {
		if x == i {
			continue
		} else if x == j {
			words = append(words, a)
		} else {
			words = append(words, value)
		}
	}
	index := 0
	for x := 0; x < len(g.Vectors); x += Width {
		if index == i {
			index++
			continue
		} else if index == j {
			vectors = append(vectors, b...)
			index++
		} else {
			vectors = append(vectors, g.Vectors[index*Width:index*Width+Width]...)
			index++
		}
	}
}

func simplify(rnd *rand.Rand, name string, vectors []float64) []float64 {
	debug, err := os.Create(fmt.Sprintf("%s_output.txt", name))
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
	set.Add("words", Width, Length/4)
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
	set.Add("inputs", Width, Length/4)
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
		for k, d := range inputs.D {
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

// Fitness calculates the fitness of a genome
func (g Genome) Fitness(truth []float64) float64 {
	sum, index, length := 0.0, 0, len(g.Words)
	for i := 0; i < length; i++ {
		a := g.Vectors[i*Width : i*Width+Width]
		type Link struct {
			Index int
			Value float64
		}
		links := make([]Link, length)
		for j := 0; j < length; j++ {
			b := g.Vectors[j*Width : j*Width+Width]
			total := 0.0
			for key, value := range a {
				total += value * b[key]
			}
			links[j].Index = j
			links[j].Value = total
		}
		sort.Slice(links, func(i, j int) bool {
			return math.Abs(links[i].Value) > math.Abs(links[j].Value)
		})
		valid := make(map[int]float64)
		for _, value := range links[:2] {
			valid[value.Index] = value.Value
		}
		for j := 0; j < length; j++ {
			if value, ok := valid[j]; ok {
				sum += (value - truth[index]) * (value - truth[index])
			}
			index++
		}
	}
	return sum
}

// IFitness calculates the fitness of a genome using fft
func (g Genome) IFitness(a [][]complex128) float64 {
	sum, index, length := 0.0, 0, len(g.Words)
	y := make([][]float64, length)
	for i := 0; i < length; i++ {
		a := g.Vectors[i*Width : i*Width+Width]
		y[i] = make([]float64, length)
		for j := 0; j < length; j++ {
			b := g.Vectors[j*Width : j*Width+Width]
			total := 0.0
			for key, value := range a {
				total += value * b[key]
			}
			y[i][j] = total
			index++
		}
	}
	b := fft.FFT2Real(y)
	for i, aa := range a {
		for j, aaa := range aa {
			diffPhase := cmplx.Phase(aaa) - cmplx.Phase(b[i][j])
			//diffAbs := cmplx.Abs(aaa) - cmplx.Abs(b[i][j])
			sum += /*diffAbs*diffAbs +*/ diffPhase * diffPhase
		}
	}
	return sum
}

// Copy copies a genome
func (g Genome) Copy() Genome {
	vectors, words := make([]float64, len(g.Vectors)), make([]string, len(g.Words))
	copy(vectors, g.Vectors)
	copy(words, g.Words)
	return Genome{
		Words:   words,
		Vectors: vectors,
	}
}

// Genetic is the genetic algorithm model
func Genetic(dictionary map[string]string, wordsEnglish, wordsGerman, words []string, vectors []float64) {
	rnd := rand.New(rand.NewSource(1))
	simple := false
	flat, a, length := make([]float64, 0, len(wordsEnglish)), make([][]float64, len(wordsEnglish)), len(wordsEnglish)
	englishVectors := vectors[:len(vectors)/2]
	if simple {
		englishVectors = simplify(rnd, "english", englishVectors)
	}
	for i := 0; i < length; i++ {
		a[i] = make([]float64, length)
		x := englishVectors[i*Width : i*Width+Width]
		for j := 0; j < length; j++ {
			y := englishVectors[j*Width : j*Width+Width]
			total := 0.0
			for key, value := range x {
				total += value * y[key]
			}
			a[i][j] = total
			flat = append(flat, total)
		}
	}
	aa := fft.FFT2Real(a)
	_ = aa
	germanVectors, genomes := vectors[len(vectors)/2:], make([]Genome, 0, 8)
	if simple {
		germanVectors = simplify(rnd, "german", germanVectors)
	}
	for i := 0; i < 256; i++ {
		v := make([]float64, len(vectors)/2)
		copy(v, germanVectors)
		w := make([]string, len(wordsGerman))
		copy(w, wordsGerman)
		genome := Genome{
			Vectors: v,
			Words:   w,
		}
		rnd.Shuffle(len(wordsGerman), func(i, j int) {
			genome.Swap(i, j)
		})
		genomes = append(genomes, genome)
	}
	v := make([]float64, len(vectors)/2)
	copy(v, germanVectors)
	w := make([]string, len(wordsGerman))
	copy(w, wordsGerman)
	target := Genome{
		Vectors: v,
		Words:   w,
	}
	average, squared := 0.0, 0.0
	for i := range genomes {
		fit := genomes[i].Fitness(flat)
		genomes[i].Fit = fit
		average += fit
		squared += fit * fit
	}
	t := target.Fitness(flat)
	fmt.Println("target=", t)
	average /= float64(len(genomes))
	stddev := math.Sqrt(squared/float64(len(genomes)) - average*average)
	fmt.Println("average=", average)
	fmt.Println("stddev=", stddev)
	for i := range genomes {
		genomes[i].Fit = math.Abs(genomes[i].Fit)
	}
	size := len(genomes)
	for g := 0; g < 1024; g++ {
		for i := 0; i < size; i++ {
			cp := genomes[i].Copy()
			cp.Swap(rnd.Intn(len(cp.Words)), rnd.Intn(len(cp.Words)))
			cp.Fit = math.Abs(cp.Fitness(flat))
			genomes = append(genomes, cp)
		}
		sort.Slice(genomes, func(i, j int) bool {
			return genomes[i].Fit < genomes[j].Fit
		})
		genomes = genomes[:size]
		fmt.Println(g, genomes[0].Fit)
	}
	for i, word := range wordsEnglish {
		fmt.Printf("%7s %13s\n", word, wordsGerman[i])
	}
	fmt.Println()
	for _, word := range genomes[0].Words {
		fmt.Println(word)
	}
	fmt.Println()
}
