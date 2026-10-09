// Copyright 2026 The mtm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command mtm trains a probabilistic Markov Turing machine on a text
// file and runs it to generate more text.
//
//	mtm -file pg100.txt -order 2 -n 200
//	mtm -start "To be" -sentences 4 -temp 0.8 -seed 1
//	mtm -holdout 0.1 -order 3
package main

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"strings"
	"time"
)

func main() {
	file := flag.String("file", "pg100.txt", "training text")
	order := flag.Int("order", 2, "Markov order, 1..6 symbols of state")
	n := flag.Int("n", 200, "maximum symbols to write")
	sentences := flag.Int("sentences", 0, "stop after this many sentence ends (0 disables)")
	temp := flag.Float64("temp", 1, "sampling temperature; 0 picks the mode at each context")
	seed := flag.Int64("seed", 0, "random seed; 0 uses the current time")
	start := flag.String("start", "", "prompt already on the tape")
	holdout := flag.Float64("holdout", 0.1, "fraction of the tape scored before it is learned; 0 scores nothing")
	flag.Parse()

	if err := run(*file, *order, *n, *sentences, *temp, *seed, *start, *holdout); err != nil {
		fmt.Fprintf(os.Stderr, "mtm: %s\n", err)
		os.Exit(1)
	}
}

func run(file string, order, n, sentences int, temp float64, seed int64, start string, holdout float64) error {
	if order < 1 || order > maxOrder {
		return fmt.Errorf("order %d is outside 1..%d", order, maxOrder)
	}
	if n < 1 {
		return fmt.Errorf("-n must be positive")
	}
	if sentences < 0 {
		return fmt.Errorf("-sentences must be >= 0")
	}
	if temp < 0 {
		return fmt.Errorf("-temp must be >= 0")
	}
	if holdout < 0 || holdout >= 1 {
		return fmt.Errorf("-holdout %g is outside [0, 1)", holdout)
	}

	data, err := os.ReadFile(file)
	if err != nil {
		return err
	}
	m, err := New(order)
	if err != nil {
		return err
	}
	ev, scored, err := m.TrainAndScore(string(data), holdout, temp)
	if err != nil {
		return err
	}
	if scored {
		if ev.KnownTokens > 0 {
			fmt.Fprintf(os.Stderr, "held out %d tokens (%.1f%%): %.3f bits/token, perplexity %.2f, known %.3f bits/token (%d), oov %d, temp %g\n",
				ev.Tokens, holdout*100, ev.Bits, ev.Perplexity, ev.KnownBits, ev.KnownTokens, ev.OOV, temp)
		} else {
			fmt.Fprintf(os.Stderr, "held out %d tokens (%.1f%%): %.3f bits/token, perplexity %.2f, oov %d, temp %g\n",
				ev.Tokens, holdout*100, ev.Bits, ev.Perplexity, ev.OOV, temp)
		}
	}

	s := seed
	if s == 0 {
		s = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewPCG(uint64(s), uint64(s)^0x9e3779b97f4a7c15))

	var prompt []uint32
	implicitNL := false
	if start == "" {
		prompt = prime(m)
		implicitNL = len(prompt) > 0
	} else {
		prompt, err = m.Encode(start)
		if err != nil {
			return err
		}
	}

	text, err := m.Generate(rng, prompt, n, sentences, temp)
	if err != nil {
		return err
	}
	if implicitNL {
		text = strings.TrimLeft(text, "\n")
	}

	st := m.Stats()
	fmt.Fprintf(os.Stderr, "learned %d tokens, %d symbols, %d contexts, order %d, temp %g, seed %d\n",
		st.Tokens, st.Symbols, st.Contexts, st.Order, temp, s)
	fmt.Println(text)
	return nil
}
