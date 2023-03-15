// Copyright 2023 The Zero Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/cmplx"
	"math/rand"
	"sort"

	"github.com/pointlander/gradient/tc128"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

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
