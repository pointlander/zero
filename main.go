// Copyright 2023 The Zero Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/cmplx"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/mat"

	"github.com/pointlander/pagerank"
)

const (
	// Size is the size of the square matrix
	Size = 10
)

func main() {
	rand.Seed(1)

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
