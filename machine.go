// Copyright 2026 The mtm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	// maxOrder is the longest Markov context the machine will learn.
	maxOrder = 6
	// linearLimit is how many distinct successors a state keeps in a
	// slice before it grows an index. Most contexts have one or two.
	linearLimit = 8
	// moveRight is the only head motion this machine learns. Each step
	// writes onto blank tape and advances one cell.
	moveRight = 1
)

// context is the finite-control state: the last n symbols written to
// the left of the head. w[0] is the oldest symbol. Unused slots stay
// zero so the value can be a map key.
type context struct {
	n uint8
	w [maxOrder]uint32
}

func (c context) push(sym uint32, order int) context {
	if int(c.n) < order {
		c.w[c.n] = sym
		c.n++
		return c
	}
	copy(c.w[:order-1], c.w[1:order])
	c.w[order-1] = sym
	c.n = uint8(order)
	return c
}

func (c context) dropOldest() context {
	if c.n == 0 {
		return c
	}
	copy(c.w[:c.n-1], c.w[1:c.n])
	c.n--
	c.w[c.n] = 0
	return c
}

// dist is the learned distribution over symbols to write.
type dist struct {
	sym   []uint32
	cnt   []uint32
	total uint32
	index map[uint32]uint32
}

func (d *dist) add(id uint32) {
	if d.index != nil {
		if i, ok := d.index[id]; ok {
			d.cnt[i]++
		} else {
			d.index[id] = uint32(len(d.sym))
			d.sym = append(d.sym, id)
			d.cnt = append(d.cnt, 1)
		}
		d.total++
		return
	}
	for i, s := range d.sym {
		if s == id {
			d.cnt[i]++
			d.total++
			return
		}
	}
	d.sym = append(d.sym, id)
	d.cnt = append(d.cnt, 1)
	d.total++
	if len(d.sym) > linearLimit {
		d.index = make(map[uint32]uint32, len(d.sym)*2)
		for i, s := range d.sym {
			d.index[s] = uint32(i)
		}
	}
}

// sample draws a symbol. Temperature 0 (and anything below 1e-3) is
// greedy: the most frequent successor wins, and earlier observations
// win ties. Temperature 1 samples in proportion to the counts. Higher
// temperatures flatten the distribution.
func (d *dist) sample(rng *rand.Rand, temp float64) uint32 {
	if temp < 1e-3 || len(d.sym) == 1 {
		best := 0
		for i := 1; i < len(d.cnt); i++ {
			if d.cnt[i] > d.cnt[best] {
				best = i
			}
		}
		return d.sym[best]
	}
	if math.Abs(temp-1) < 1e-9 {
		pick := rng.Uint32N(d.total)
		var run uint32
		for i, c := range d.cnt {
			run += c
			if pick < run {
				return d.sym[i]
			}
		}
		return d.sym[len(d.sym)-1]
	}
	inv := 1 / temp
	sum := 0.0
	weights := make([]float64, len(d.cnt))
	for i, c := range d.cnt {
		w := math.Pow(float64(c), inv)
		weights[i] = w
		sum += w
	}
	pick := rng.Float64() * sum
	run := 0.0
	for i, w := range weights {
		run += w
		if pick < run {
			return d.sym[i]
		}
	}
	return d.sym[len(d.sym)-1]
}

// Tape is the machine's work tape. The head writes at head and the
// cells to the left are the symbols already produced.
type Tape struct {
	cells []uint32
	head  int
}

func (t *Tape) state(order int) context {
	from := t.head - order
	if from < 0 {
		from = 0
	}
	window := t.cells[from:t.head]
	var c context
	c.n = uint8(len(window))
	copy(c.w[:], window)
	return c
}

// Machine is a probabilistic Turing machine that writes text.
//
// The finite-control state is a Markov context: the last Order symbols
// to the left of the head. A transition samples a symbol from the
// distribution learned for that context, writes it onto the blank cell
// under the head, and moves one cell to the right. The next state is
// that context shifted by the written symbol. A context that was never
// observed falls back to the longest suffix that was.
type Machine struct {
	order  int
	vocab  []string
	ids    map[string]uint32
	trans  map[context]*dist
	tokens int
}

// New returns a machine that conditions each symbol on at most order
// preceding symbols.
func New(order int) (*Machine, error) {
	if order < 1 || order > maxOrder {
		return nil, fmt.Errorf("order %d is outside 1..%d", order, maxOrder)
	}
	return &Machine{
		order: order,
		ids:   make(map[string]uint32),
		trans: make(map[context]*dist),
	}, nil
}

func (m *Machine) intern(s string) uint32 {
	if id, ok := m.ids[s]; ok {
		return id
	}
	id := uint32(len(m.vocab))
	m.vocab = append(m.vocab, s)
	m.ids[s] = id
	return id
}

// Learn walks a training tape from left to right and counts, for every
// context length up through Order, which symbol was written next.
func (m *Machine) Learn(ids []uint32) {
	tape := Tape{cells: ids}
	var ctx context
	for tape.head < len(tape.cells) {
		sym := tape.cells[tape.head]
		m.observe(ctx, sym)
		ctx = ctx.push(sym, m.order)
		tape.head += moveRight
	}
	m.tokens += len(ids)
	for _, d := range m.trans {
		d.index = nil
	}
}

func (m *Machine) observe(ctx context, sym uint32) {
	for {
		m.add(ctx, sym)
		if ctx.n == 0 {
			return
		}
		ctx = ctx.dropOldest()
	}
}

func (m *Machine) add(ctx context, sym uint32) {
	d := m.trans[ctx]
	if d == nil {
		d = &dist{}
		m.trans[ctx] = d
	}
	d.add(sym)
}

// distribution returns the longest observed suffix of ctx.
func (m *Machine) distribution(ctx context) *dist {
	for {
		if d := m.trans[ctx]; d != nil && d.total > 0 {
			return d
		}
		if ctx.n == 0 {
			return nil
		}
		ctx = ctx.dropOldest()
	}
}

func (m *Machine) sample(ctx context, rng *rand.Rand, temp float64) (uint32, bool) {
	d := m.distribution(ctx)
	if d == nil {
		return 0, false
	}
	return d.sample(rng, temp), true
}

// Step samples one transition and applies it. The head moves right.
func (m *Machine) Step(rng *rand.Rand, tape *Tape, temp float64) (uint32, bool) {
	sym, ok := m.sample(tape.state(m.order), rng, temp)
	if !ok {
		return 0, false
	}
	if tape.head == len(tape.cells) {
		tape.cells = append(tape.cells, sym)
	} else {
		tape.cells[tape.head] = sym
	}
	tape.head += moveRight
	return sym, true
}

// Train tokenizes text and learns it.
func (m *Machine) Train(text string) error {
	parts := Tokenize(text)
	if len(parts) == 0 {
		return fmt.Errorf("training text has no tokens")
	}
	ids := make([]uint32, len(parts))
	for i, part := range parts {
		ids[i] = m.intern(part)
	}
	m.Learn(ids)
	return nil
}

// Encode tokenizes a prompt into symbols the machine already knows.
func (m *Machine) Encode(text string) ([]uint32, error) {
	parts := Tokenize(text)
	ids := make([]uint32, 0, len(parts))
	for _, part := range parts {
		id, ok := m.ids[part]
		if !ok {
			return nil, fmt.Errorf("unknown token %q", part)
		}
		ids = append(ids, id)
	}
	return ids, nil
}

// Generate runs the machine for at most tokens steps, starting with
// seed already written to the left of the head. If sentences is
// positive, the run also stops after that many sentence-ending marks.
func (m *Machine) Generate(rng *rand.Rand, seed []uint32, tokens, sentences int, temp float64) (string, error) {
	if m.tokens == 0 {
		return "", fmt.Errorf("machine has not learned")
	}
	if tokens < 1 {
		return "", fmt.Errorf("token count must be positive")
	}
	if temp < 0 {
		return "", fmt.Errorf("temperature must be >= 0")
	}
	cells := make([]uint32, len(seed), len(seed)+tokens)
	copy(cells, seed)
	tape := Tape{cells: cells, head: len(seed)}
	stops := 0
	for written := 0; written < tokens; written++ {
		sym, ok := m.Step(rng, &tape, temp)
		if !ok {
			break
		}
		if sentences > 0 && isSentenceEnd(m.vocab[sym]) {
			stops++
			if stops >= sentences {
				break
			}
		}
	}
	return m.Text(tape.cells), nil
}

// Text renders tape symbols with spaces a person would expect.
func (m *Machine) Text(ids []uint32) string {
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = m.vocab[id]
	}
	return joinTokens(parts)
}

// Stats describes what the machine learned.
type Stats struct {
	Tokens   int
	Symbols  int
	Contexts int
	Order    int
}

// Stats returns counts from learning.
func (m *Machine) Stats() Stats {
	return Stats{
		Tokens:   m.tokens,
		Symbols:  len(m.vocab),
		Contexts: len(m.trans),
		Order:    m.order,
	}
}
