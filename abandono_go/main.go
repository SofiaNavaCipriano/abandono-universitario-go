package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("Cargando datos del CSV...")
	records, err := readCSV("estudiantes_abandono.csv")
	if err != nil {
		fmt.Println("Error leyendo CSV:", err)
		return
	}

	fmt.Println("Total de estudiantes cargados:", len(records))

	fmt.Println("Normalizando características (StandardScaler)...")
	scaler := Standardize(records)

	fmt.Println("Entrenando modelo de Regresión Logística...")
	lr := LogisticRegression{
		Rate:   0.1,
		Epochs: 2000,
	}
	lr.Train(records)

	lr.PrintFeatureImportance()

	aciertos := 0
	for _, r := range records {
		pred := lr.Predict(r.Features)
		if pred == r.Label {
			aciertos++
		}
	}
	fmt.Printf("Precisión global del modelo: %.2f%%\n\n", float64(aciertos)/float64(len(records))*100)

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Println("\n========================================")
		fmt.Println("  EVALUAR CASO FICTICIO")
		fmt.Println("========================================")
		fmt.Print("Deseas evaluar un estudiante? (s/n): ")
		os.Stdout.Sync()
		if !scanner.Scan() {
			break
		}
		resp := strings.TrimSpace(strings.ToLower(scanner.Text()))
		if resp != "s" && resp != "si" && resp != "sí" {
			break
		}

		r := readStudent(scanner)
		scaler.ScaleRecord(&r)
		prob := lr.PredictProb(r.Features)
		pred := lr.Predict(r.Features)

		fmt.Println("\n--- RESULTADO ---")
		fmt.Printf("Probabilidad de abandono: %.2f%%\n", prob*100)
		if pred == 1 {
			fmt.Println("PREDICCION: El estudiante ABANDONARA sus estudios.")
		} else {
			fmt.Println("PREDICCION: El estudiante NO abandona sus estudios.")
		}
	}
}

func prompt(scanner *bufio.Scanner, msg string) string {
	fmt.Print(msg)
	os.Stdout.Sync()
	if !scanner.Scan() {
		return ""
	}
	return strings.TrimSpace(scanner.Text())
}

func promptFloat(scanner *bufio.Scanner, msg string) float64 {
	for {
		line := prompt(scanner, msg)
		val, err := strconv.ParseFloat(line, 64)
		if err == nil {
			return val
		}
		fmt.Println("Valor invalido. Ingresa un numero.")
	}
}

func promptOption(scanner *bufio.Scanner, msg string, options []string) int {
	for {
		fmt.Println(msg)
		for i, opt := range options {
			fmt.Printf("  [%d] %s\n", i+1, opt)
		}
		fmt.Print("Opcion: ")
		os.Stdout.Sync()
		if !scanner.Scan() {
			return 0
		}
		line := strings.TrimSpace(scanner.Text())
		val, err := strconv.ParseInt(line, 10, 64)
		if err == nil && val >= 1 && val <= int64(len(options)) {
			return int(val) - 1
		}
		fmt.Println("Opcion invalida.")
	}
}

func readStudent(scanner *bufio.Scanner) Record {
	fmt.Println("\n--- Ingresa los datos del estudiante ---")

	edad := promptFloat(scanner, "Edad: ")
	gen := prompt(scanner, "Genero (M/F): ")
	esMasc := 0.0
	if strings.ToUpper(gen) == "M" {
		esMasc = 1.0
	}

	civil := prompt(scanner, "Estado civil (Casado/Soltero): ")
	esCas := 0.0
	if strings.ToUpper(civil) == "CASADO" {
		esCas = 1.0
	}

	hijos := promptFloat(scanner, "Numero de hijos: ")

	ciudadIdx := promptOption(scanner, "Ciudad de origen:", ciudades)
	viveFuera := promptFloat(scanner, "Vive fuera de su ciudad? (0=No, 1=Si): ")

	carreraIdx := promptOption(scanner, "Carrera:", carreras)

	semestre := promptFloat(scanner, "Semestre actual: ")
	promedio := promptFloat(scanner, "Promedio general (0-10, ej: 7.5): ")
	repro := promptFloat(scanner, "Materias reprobadas: ")
	asis := promptFloat(scanner, "Asistencia promedio % (0-100): ")

	bac := prompt(scanner, "Tipo de bachillerato (General/Tecnico): ")
	esBachGral := 0.0
	if strings.ToUpper(bac) == "GENERAL" {
		esBachGral = 1.0
	}

	cambio := promptFloat(scanner, "Ha cambiado de carrera? (0=No, 1=Si): ")
	ingreso := promptFloat(scanner, "Ingreso familiar mensual: ")
	trabaja := promptFloat(scanner, "Trabaja? (0=No, 1=Si): ")
	horasTrab := promptFloat(scanner, "Horas de trabajo por semana: ")
	tieneBeca := promptFloat(scanner, "Tiene beca? (0=No, 1=Si): ")

	becaIdx := promptOption(scanner, "Tipo de beca:", becas)

	nivelEstres := promptFloat(scanner, "Nivel de estres (1-10): ")
	satisfaccion := promptFloat(scanner, "Satisfaccion carrera (1-10): ")
	motivacion := promptFloat(scanner, "Motivacion (1-10): ")
	salud := promptFloat(scanner, "Problemas salud (0=No, 1=Si): ")
	apoyo := promptFloat(scanner, "Apoyo familiar (1-10): ")
	distancia := promptFloat(scanner, "Distancia universidad (km): ")
	traslado := promptFloat(scanner, "Tiempo traslado (min): ")
	calidad := promptFloat(scanner, "Calidad docentes (1-10): ")
	infra := promptFloat(scanner, "Infraestructura (1-10): ")
	carga := promptFloat(scanner, "Carga academica (1-10): ")

	feats := []float64{}
	feats = append(feats, edad, esMasc, esCas, hijos, viveFuera)

	for i := 0; i < len(ciudades); i++ {
		if i == ciudadIdx {
			feats = append(feats, 1)
		} else {
			feats = append(feats, 0)
		}
	}

	for i := 0; i < len(carreras); i++ {
		if i == carreraIdx {
			feats = append(feats, 1)
		} else {
			feats = append(feats, 0)
		}
	}

	feats = append(feats, semestre, promedio, repro, asis,
		esBachGral, cambio, ingreso, trabaja, horasTrab, tieneBeca)

	for i := 0; i < len(becas); i++ {
		if i == becaIdx {
			feats = append(feats, 1)
		} else {
			feats = append(feats, 0)
		}
	}

	feats = append(feats, nivelEstres, satisfaccion, motivacion, salud,
		apoyo, distancia, traslado, calidad, infra, carga)

	return Record{Features: feats}
}
