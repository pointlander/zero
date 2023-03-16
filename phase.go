// Copyright 2023 The Zero Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"sort"

	"github.com/pointlander/gradient/tc128"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// Phase is complex phase model
func Phase(dictionary map[string]string, words []string, vectors []float64) {
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
	for i := 0; i < len(vectors); i++ {
		if i < len(vectors)/2 {
			x.X = append(x.X, cmplx.Rect(vectors[i], math.Pi/8))
		} else {
			x.X = append(x.X, cmplx.Rect(vectors[i], -math.Pi/8))
		}
	}

	l1 := tc128.Mul(set.Get("A"), other.Get("X"))
	cost := tc128.Avg(tc128.Quadratic(other.Get("X"), l1))

	iterations := 1024
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

	x.X = x.X[:0]
	for i := 0; i < len(vectors); i++ {
		if i < len(vectors)/2 {
			x.X = append(x.X, cmplx.Rect(vectors[i], 0))
		} else {
			x.X = append(x.X, cmplx.Rect(vectors[i], 0))
		}
	}

	type Pair struct {
		S float64
		I int
	}
	pairs := make([][]Pair, Length/4)
	for i := range pairs {
		pairs[i] = make([]Pair, 0, 8)
	}
	l1(func(a *tc128.V) bool {
		for k := 0; k < Length/4; k++ {
			for i := Length / 4; i < Length/2; i++ {
				var aa, bb, ab complex128
				for j := 0; j < Width; j++ {
					a, b := a.X[k*Width+j], a.X[i*Width+j]
					aa += a * a
					bb += b * b
					ab += a * b
				}
				s := ab / (cmplx.Sqrt(aa) * cmplx.Sqrt(bb))
				pairs[k] = append(pairs[k], Pair{
					S: cmplx.Abs(s),
					I: i,
				})
			}
		}
		return true
	})
	a := float64(0.0)
	for k := range pairs {
		sort.Slice(pairs[k], func(i, j int) bool {
			return pairs[k][i].S > pairs[k][j].S
		})
		for i, pair := range pairs[k] {
			if words[k] == dictionary[words[pair.I]] {
				a += float64(i)
				break
			}
		}
	}
	fmt.Println("accuracy=", a/float64(Length/4))
}
