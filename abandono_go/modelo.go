package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

type LogisticRegression struct {
	Weights []float64
	Bias    float64
	Rate    float64
	Epochs  int
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func (lr *LogisticRegression) Train(data []Record) {
	nFeatures := len(data[0].Features)
	lr.Weights = make([]float64, nFeatures)
	// Initialize weights randomly
	for i := range lr.Weights {
		lr.Weights[i] = rand.Float64()*0.1 - 0.05
	}
	lr.Bias = 0

	n := float64(len(data))
	for e := 0; e < lr.Epochs; e++ {
		for _, r := range data {
			z := lr.Bias
			for i, f := range r.Features {
				z += lr.Weights[i] * f
			}
			pred := sigmoid(z)
			error := pred - r.Label

			lr.Bias -= lr.Rate * error / n
			for i, f := range r.Features {
				lr.Weights[i] -= lr.Rate * error * f / n
			}
		}
	}
}

func (lr *LogisticRegression) PredictProb(features []float64) float64 {
	z := lr.Bias
	for i, f := range features {
		z += lr.Weights[i] * f
	}
	return sigmoid(z)
}

func (lr *LogisticRegression) Predict(features []float64) float64 {
	if lr.PredictProb(features) >= 0.5 {
		return 1.0
	}
	return 0.0
}

type FeatureWeight struct {
	Name   string
	Weight float64
	AbsW   float64
}

func (lr *LogisticRegression) PrintFeatureImportance() {
	var fws []FeatureWeight
	for i, w := range lr.Weights {
		fws = append(fws, FeatureWeight{Name: FeatureNames[i], Weight: w, AbsW: math.Abs(w)})
	}

	sort.Slice(fws, func(i, j int) bool {
		return fws[i].AbsW > fws[j].AbsW
	})

	fmt.Println("\n=== TOP CAUSAS DE ABANDONO ENCONTRADAS POR EL MODELO ===")
	fmt.Println("Las variables con peso positivo (+) empujan al alumno a ABANDONAR.")
	fmt.Println("Las variables con peso negativo (-) ayudan a RETENER al alumno.")
	fmt.Println("---------------------------------------------------------------")
	for i := 0; i < 10 && i < len(fws); i++ {
		impacto := "Aumenta riesgo abandono"
		if fws[i].Weight < 0 {
			impacto = "Ayuda a retener"
		}
		fmt.Printf("%d. %-25s | Peso: %6.3f (%s)\n", i+1, fws[i].Name, fws[i].Weight, impacto)
	}
	fmt.Println("===============================================================\n")
}
