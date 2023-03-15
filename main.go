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
	"strconv"
	"strings"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tc128"
	"github.com/pointlander/gradient/tf32"
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
	// EtaA is the learning rate for autoencoder based model
	EtaA = .1
	// EpochsT is the number of epochs for autoencoder based model
	EpochsA = 256
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
	// FlagTransform transform based entropy minimization
	FlagTransform = flag.Bool("transform", false, "transform based entropy minimization")
	// FlagAutoencode autoencoder mode
	FlagAutoencode = flag.Bool("autoencode", false, "autoencoder mode")
)

// Entropy is the output self entropy of the model
type Entropy struct {
	Index   int
	Entropy float32
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
	} else if *FlagTransform {
		Transform(dictionary, words, vectors)
		return
	} else if *FlagAutoencode {
		state := NewState()
		state.autoencode(dictionary, words, vectors)

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

		err = p.Save(8*vg.Inch, 8*vg.Inch, "autoencoder_cost.png")
		if err != nil {
			panic(err)
		}
		return
	}

	rnd := rand.New(rand.NewSource(1))

	set := tc128.NewSet()
	set.Add("A", Width, Width)
	for _, w := range set.Weights {
		factor, size := math.Sqrt(2.0/float64(w.S[0])), cap(w.X)
		for i := 0; i < size; i++ {
			w.X = append(w.X, complex(factor*rnd.NormFloat64(), factor*rnd.NormFloat64()))
		}
		w.States = make([][]complex128, 1)
		for i := range w.States {
			w.States[i] = make([]complex128, len(w.X))
		}
	}

	other := tc128.NewSet()
	other.Add("X", Width, Length/2)
	x := other.ByName["X"]
	for i := 0; i < Length/2; i++ {
		if i < Length/4 {
			x.X = append(x.X, cmplx.Rect(vectors[i], math.Pi/4))
		} else {
			x.X = append(x.X, complex(vectors[i], 0))
		}
	}

	l1 := tc128.Mul(set.Get("A"), other.Get("X"))
	cost := tc128.Avg(tc128.Quadratic(other.Get("X"), l1))

	iterations := 128
	points := make(plotter.XYs, 0, iterations)
	phase := make(plotter.XYs, 0, iterations)
	alpha, eta := complex(.3, 0), complex(.3, 0)
	for i := 0; i < iterations; i++ {
		set.Zero()
		other.Zero()

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
		for _, w := range set.Weights {
			for l, d := range w.D {
				w.States[0][l] = alpha*w.States[0][l] - eta*d*scaling
				w.X[l] += w.States[0][l]
			}
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

	err = p.Save(8*vg.Inch, 8*vg.Inch, "complex_cost.png")
	if err != nil {
		panic(err)
	}
}
