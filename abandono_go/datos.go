package main

import (
	"encoding/csv"
	"math"
	"os"
	"strconv"
)

type Record struct {
	Features []float64
	Label    float64
}

var FeatureNames []string
var ciudades, carreras, becas []string

func readCSV(filename string) ([]Record, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	lines, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	ciudadMap := make(map[string]int)
	carreraMap := make(map[string]int)
	becaMap := make(map[string]int)

	for _, line := range lines[1:] {
		ciudadMap[line[5]] = 1
		carreraMap[line[7]] = 1
		becaMap[line[18]] = 1
	}

	ciudades = getKeys(ciudadMap)
	carreras = getKeys(carreraMap)
	becas = getKeys(becaMap)

	FeatureNames = []string{
		"edad", "es_masculino", "es_casado", "hijos", "vive_fuera",
	}
	for _, c := range ciudades {
		FeatureNames = append(FeatureNames, "ciudad_"+c)
	}
	for _, c := range carreras {
		FeatureNames = append(FeatureNames, "carrera_"+c)
	}
	FeatureNames = append(FeatureNames,
		"semestre_actual", "promedio_general", "materias_reprobadas", "asistencia_promedio",
		"es_bach_general", "cambio_carrera", "ingreso_mensual",
		"trabaja", "horas_trabajo_sem", "tiene_beca",
	)
	for _, c := range becas {
		FeatureNames = append(FeatureNames, "beca_"+c)
	}
	FeatureNames = append(FeatureNames,
		"nivel_estres", "satisfaccion_carrera", "motivacion", "problemas_salud",
		"apoyo_familiar", "distancia_uni_km", "tiempo_traslado_min",
		"calidad_docentes", "infraestructura", "carga_academica",
	)

	var records []Record
	for _, line := range lines[1:] {
		var feats []float64
		feats = append(feats, parse(line[1]))
		if line[2] == "M" { feats = append(feats, 1) } else { feats = append(feats, 0) }
		if line[3] == "Casado" { feats = append(feats, 1) } else { feats = append(feats, 0) }
		feats = append(feats, parse(line[4]), parse(line[6]))

		for _, c := range ciudades {
			if line[5] == c { feats = append(feats, 1) } else { feats = append(feats, 0) }
		}
		for _, c := range carreras {
			if line[7] == c { feats = append(feats, 1) } else { feats = append(feats, 0) }
		}

		feats = append(feats, parse(line[8]), parse(line[9]), parse(line[10]), parse(line[11]))
		if line[12] == "General" { feats = append(feats, 1) } else { feats = append(feats, 0) }
		feats = append(feats, parse(line[13]), parse(line[14]), parse(line[15]), parse(line[16]), parse(line[17]))

		for _, c := range becas {
			if line[18] == c { feats = append(feats, 1) } else { feats = append(feats, 0) }
		}

		for i := 19; i <= 28; i++ {
			feats = append(feats, parse(line[i]))
		}

		label := parse(line[29])
		records = append(records, Record{Features: feats, Label: label})
	}
	return records, nil
}

func parse(s string) float64 {
	f, _ := strconv.ParseFloat(s, 64)
	return f
}

func getKeys(m map[string]int) []string {
	var keys []string
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

type Scaler struct {
	Means []float64
	Stds  []float64
}

func Standardize(records []Record) Scaler {
	nFeatures := len(records[0].Features)
	means := make([]float64, nFeatures)
	stds := make([]float64, nFeatures)
	n := float64(len(records))

	for _, r := range records {
		for i, f := range r.Features {
			means[i] += f
		}
	}
	for i := range means {
		means[i] /= n
	}

	for _, r := range records {
		for i, f := range r.Features {
			stds[i] += (f - means[i]) * (f - means[i])
		}
	}
	for i := range stds {
		stds[i] = math.Sqrt(stds[i] / n)
		if stds[i] == 0 {
			stds[i] = 1
		}
	}

	for i := range records {
		for j := range records[i].Features {
			records[i].Features[j] = (records[i].Features[j] - means[j]) / stds[j]
		}
	}

	return Scaler{Means: means, Stds: stds}
}

func (s Scaler) ScaleRecord(r *Record) {
	for j := range r.Features {
		r.Features[j] = (r.Features[j] - s.Means[j]) / s.Stds[j]
	}
}
