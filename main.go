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
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

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
	Length = 4 * 993 //128
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
	// FlagPhase complex phase mode
	FlagPhase = flag.Bool("phase", false, "complex phase mode")
	// FlagGenetic genetic algorithm mode
	FlagGenetic = flag.Bool("genetic", false, "genetic mode")
	// FlagBrute brute force mode
	FlagBrute = flag.Bool("brute", false, "brute force mode")
)

// Entropy is the output self entropy of the model
type Entropy struct {
	Index   int
	Entropy float32
}

func se(gradient bool, rnd *rand.Rand, name string, Q, K, V []float64) ([]float64, []float64) {
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
	_ = dropout

	set := tf32.NewSet()
	set.Add("q", Width, Length/4)
	q := set.ByName["q"]
	for _, value := range Q {
		q.X = append(q.X, float32(value))
	}
	q.States = make([][]float32, StateTotal)
	for i := range q.States {
		q.States[i] = make([]float32, len(q.X))
	}

	set.Add("k", Width, Length/4)
	k := set.ByName["k"]
	for _, value := range K {
		k.X = append(k.X, float32(value))
	}
	k.States = make([][]float32, StateTotal)
	for i := range k.States {
		k.States[i] = make([]float32, len(k.X))
	}

	set.Add("v", Width, Length/4)
	v := set.ByName["v"]
	for _, value := range V {
		v.X = append(v.X, float32(value))
	}
	v.States = make([][]float32, StateTotal)
	for i := range v.States {
		v.States[i] = make([]float32, len(v.X))
	}

	spherical := tf32.U(SphericalSoftmaxReal)
	l1 := dropout(spherical(tf32.Mul(set.Get("q"), set.Get("k"))))
	l2 := spherical(tf32.Mul(tf32.T(set.Get("v")), l1))
	cost := tf32.Avg(tf32.Entropy(l2))

	if gradient {
		i := 1
		pow := func(x float32) float32 {
			y := math.Pow(float64(x), float64(i))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return float32(y)
		}
		points := make(plotter.XYs, 0, 8)
		for i < 8*64+1 {
			start := time.Now()
			total := tf32.Gradient(cost).X[0]

			b1, b2 := pow(B1), pow(B2)

			for j, d := range v.D {
				g := d
				m := B1*v.States[StateM][j] + (1-B1)*g
				vv := B2*v.States[StateV][j] + (1-B2)*g*g
				v.States[StateM][j] = m
				v.States[StateV][j] = vv
				mhat := m / (1 - b1)
				vhat := vv / (1 - b2)
				v.X[j] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
			}

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
	}

	l1 = spherical(tf32.Mul(set.Get("q"), set.Get("k")))
	l2 = spherical(tf32.Mul(tf32.T(set.Get("v")), l1))
	cost = tf32.Entropy(l2)

	e := make([]float64, 0, 8)
	cost(func(a *tf32.V) bool {
		for _, value := range a.X {
			e = append(e, float64(value))
		}
		return true
	})

	vec := make([]float64, 0, 8)
	l2(func(a *tf32.V) bool {
		for _, value := range a.X {
			vec = append(vec, float64(value))
		}
		return true
	})
	return e, vec
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
	/*wordsEnglish := []string{
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
	}*/
	wordsEnglish := []string{}
	wordsGerman := []string{}
	dictionary := make(map[string]string)
	/*for i, english := range wordsEnglish[:Words] {
		german := wordsGerman[i]
		dictionary[english] = german
		dictionary[german] = english
	}*/
	for _, pair := range Pairs {
		dictionary[pair.English] = pair.German
		dictionary[pair.German] = pair.English
	}
	words := make([]string, 0, len(wordsEnglish)+len(wordsGerman))
	//words = append(words, wordsEnglish[:Words]...)
	for _, pair := range Pairs {
		wordsEnglish = append(wordsEnglish, pair.English)
		words = append(words, pair.English)
	}
	//words = append(words, wordsGerman[:Words]...)
	for _, pair := range Pairs {
		wordsGerman = append(wordsGerman, pair.German)
		words = append(words, pair.German)
	}
	if err != nil {
		english := NewVectors("cc.en.300.vec.gz")
		german := NewVectors("cc.de.300.vec.gz")

		for _, pair := range Pairs {
			vector := english.Dictionary[pair.English]
			if len(vector.Vector) == 0 {
				panic(pair.English)
			}
			vectors = append(vectors, vector.Vector...)
		}
		for _, pair := range Pairs {
			vector := german.Dictionary[pair.German]
			if len(vector.Vector) == 0 {
				panic(pair.German)
			}
			vectors = append(vectors, vector.Vector...)
		}

		/*for _, word := range wordsEnglish[:Words] {
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
		}*/
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
	} else if *FlagPhase {
		Phase(dictionary, words, vectors)
		return
	} else if *FlagGenetic {
		Genetic(dictionary, wordsEnglish, wordsGerman, words, vectors)
		return
	} else if *FlagBrute {
		Brute(dictionary, wordsEnglish, wordsGerman, words, vectors)
		return
	}

	rnd := rand.New(rand.NewSource(1))

	length := len(wordsEnglish)
	fmt.Println("length=", length)
	englishVectors := vectors[:len(vectors)/2]
	germanVectors := vectors[len(vectors)/2:]

	entropyEnglish, x := se(false, rnd, "english_english", englishVectors, englishVectors, englishVectors)
	entropyGerman, y := se(false, rnd, "german_english", englishVectors, englishVectors, germanVectors)

	for i, value := range entropyEnglish {
		fmt.Println(entropyGerman[i], value)
	}

	type Rank struct {
		Index int
		Value float64
	}
	test := func(x, y []float64, t int) (correctness int) {
		ranks := make([]Rank, 0, 8)
		for i := 0; i < length; i++ {
			sum := 0.0
			for j := 0; j < Width; j++ {
				diff := y[i*Width+j] - x[t*Width+j]
				sum += math.Abs(diff)
			}
			ranks = append(ranks, Rank{
				Index: i,
				Value: sum,
			})
		}
		sort.Slice(ranks, func(i, j int) bool {
			return ranks[i].Value < ranks[j].Value
		})
		for i, value := range ranks {
			if value.Index == t {
				correctness = i
				break
			}
		}
		return correctness
	}
	correctness := 0
	for i := 0; i < length; i++ {
		correctness += test(x, y, i)
	}
	fmt.Println("correctness english", float64(correctness)/float64(length))

	correctness = 0
	for i := 0; i < length; i++ {
		correctness += test(y, x, i)
	}
	fmt.Println("correctness german", float64(correctness)/float64(length))
}
