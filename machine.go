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
	// unkID is a held-out symbol the machine has not learned. It is not
	// a vocab index.
	unkID = ^uint32(0)
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
	zTemp float64
	zLog  float64
	zOK   bool
}

func (d *dist) add(id uint32) {
	d.zOK = false
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

func (d *dist) countOf(id uint32) (uint32, bool) {
	if d == nil || d.total == 0 {
		return 0, false
	}
	if d.index != nil {
		i, ok := d.index[id]
		if !ok {
			return 0, false
		}
		return d.cnt[i], true
	}
	for i, s := range d.sym {
		if s == id {
			return d.cnt[i], true
		}
	}
	return 0, false
}

// logPartition is log(sum count^(1/temp)), computed in a shifted log
// space so a low temperature does not overflow.
func (d *dist) logPartition(temp float64) float64 {
	if d.zOK && d.zTemp == temp {
		return d.zLog
	}
	inv := 1 / temp
	maxLog := math.Inf(-1)
	logs := make([]float64, len(d.cnt))
	for i, c := range d.cnt {
		logs[i] = inv * math.Log(float64(c))
		if logs[i] > maxLog {
			maxLog = logs[i]
		}
	}
	sum := 0.0
	for _, lg := range logs {
		sum += math.Exp(lg - maxLog)
	}
	d.zTemp = temp
	d.zLog = maxLog + math.Log(sum)
	d.zOK = true
	return d.zLog
}

// mle is the tempered empirical probability of id at this context.
// Unseen symbols get 0. Temperature 1 is count/total. Temperature 0
// puts equal mass on the modes.
func (d *dist) mle(id uint32, temp float64) float64 {
	c, ok := d.countOf(id)
	if !ok || c == 0 || d.total == 0 {
		return 0
	}
	if temp < 1e-3 {
		best := uint32(0)
		for _, n := range d.cnt {
			if n > best {
				best = n
			}
		}
		if c < best {
			return 0
		}
		ties := 0
		for _, n := range d.cnt {
			if n == best {
				ties++
			}
		}
		return 1 / float64(ties)
	}
	if math.Abs(temp-1) < 1e-9 {
		return float64(c) / float64(d.total)
	}
	return math.Exp(math.Log(float64(c))/temp - d.logPartition(temp))
}

// sample draws a symbol from the tempered empirical distribution.
// Temperature 0 picks the mode, and the earliest observation wins a tie.
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
	maxLog := math.Inf(-1)
	logs := make([]float64, len(d.cnt))
	for i, c := range d.cnt {
		logs[i] = inv * math.Log(float64(c))
		if logs[i] > maxLog {
			maxLog = logs[i]
		}
	}
	sum := 0.0
	weights := make([]float64, len(logs))
	for i, lg := range logs {
		w := math.Exp(lg - maxLog)
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
// to the left of the head. A transition samples a symbol, writes it
// onto the blank cell under the head, and moves one cell to the right.
// The symbol is drawn from a Witten-Bell interpolation of every suffix
// of the context, down to a uniform distribution over the known symbols
// and one unknown bin. A context seen once therefore keeps some of its
// mass on that continuation and gives the rest to the shorter contexts.
// Temperature reshapes the empirical distribution at each level and
// does not change how much mass escapes to the shorter context.
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

// baseProb is the 0-gram: one bin per known symbol, plus a bin for a
// symbol the machine has never seen. Every symbol, known or not, gets
// the same share, so a held-out word still has a finite probability.
func (m *Machine) baseProb() float64 {
	return 1 / float64(len(m.vocab)+1)
}

// prob is the interpolated probability of sym in ctx.
func (m *Machine) prob(ctx context, sym uint32, temp float64) float64 {
	d := m.trans[ctx]
	if d == nil || d.total == 0 {
		if ctx.n == 0 {
			return m.baseProb()
		}
		return m.prob(ctx.dropOldest(), sym, temp)
	}
	stay := float64(d.total) / (float64(d.total) + float64(len(d.sym)))
	var lower float64
	if ctx.n == 0 {
		lower = m.baseProb()
	} else {
		lower = m.prob(ctx.dropOldest(), sym, temp)
	}
	return stay*d.mle(sym, temp) + (1-stay)*lower
}

func (m *Machine) sample(ctx context, rng *rand.Rand, temp float64) (uint32, bool) {
	if len(m.vocab) == 0 {
		return 0, false
	}
	return m.sampleFrom(ctx, rng, temp), true
}

// sampleFrom draws from the same interpolation as prob. The base draw
// stays inside the vocabulary: there is no unknown symbol to write.
func (m *Machine) sampleFrom(ctx context, rng *rand.Rand, temp float64) uint32 {
	d := m.trans[ctx]
	if d == nil || d.total == 0 {
		if ctx.n == 0 {
			return uint32(rng.IntN(len(m.vocab)))
		}
		return m.sampleFrom(ctx.dropOldest(), rng, temp)
	}
	stay := float64(d.total) / (float64(d.total) + float64(len(d.sym)))
	if rng.Float64() < stay {
		return d.sample(rng, temp)
	}
	if ctx.n == 0 {
		return uint32(rng.IntN(len(m.vocab)))
	}
	return m.sampleFrom(ctx.dropOldest(), rng, temp)
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

func (m *Machine) learnParts(parts []string) []uint32 {
	ids := make([]uint32, len(parts))
	for i, part := range parts {
		ids[i] = m.intern(part)
	}
	m.Learn(ids)
	return ids
}

// Train tokenizes text and learns it.
func (m *Machine) Train(text string) error {
	parts := Tokenize(text)
	if len(parts) == 0 {
		return fmt.Errorf("training text has no tokens")
	}
	m.learnParts(parts)
	return nil
}

// Eval is the average next-symbol score of a held-out slice.
type Eval struct {
	Tokens     int
	Bits       float64
	Perplexity float64
	OOV        int
}

// TrainAndScore learns text. When holdout is in (0, 1) and the tape has
// at least two tokens, the last fraction is scored under the model of
// the prefix and only then learned, so the score is held-out and
// generation still sees the whole tape. scored is false when nothing
// was held out.
func (m *Machine) TrainAndScore(text string, holdout, temp float64) (Eval, bool, error) {
	if holdout < 0 || holdout >= 1 {
		return Eval{}, false, fmt.Errorf("holdout %g is outside [0, 1)", holdout)
	}
	if temp < 0 {
		return Eval{}, false, fmt.Errorf("temperature must be >= 0")
	}
	parts := Tokenize(text)
	if len(parts) == 0 {
		return Eval{}, false, fmt.Errorf("training text has no tokens")
	}
	if holdout == 0 || len(parts) < 2 {
		m.learnParts(parts)
		return Eval{}, false, nil
	}
	hold := int(math.Round(float64(len(parts)) * holdout))
	if hold < 1 {
		hold = 1
	}
	if hold >= len(parts) {
		hold = len(parts) - 1
	}
	cut := len(parts) - hold
	prefix := m.learnParts(parts[:cut])
	left := prefix
	if len(left) > m.order {
		left = append([]uint32(nil), left[len(left)-m.order:]...)
	}
	ev, err := m.Score(left, parts[cut:], temp)
	if err != nil {
		return Eval{}, false, err
	}
	m.learnParts(parts[cut:])
	return ev, true, nil
}

// Score reports the mean negative log2 probability of each symbol in
// symbols, conditioned on left and then on the symbols already scored.
// A string the machine has not interned is an unknown symbol.
func (m *Machine) Score(left []uint32, symbols []string, temp float64) (Eval, error) {
	if m.tokens == 0 || len(m.vocab) == 0 {
		return Eval{}, fmt.Errorf("machine has not learned")
	}
	if len(symbols) == 0 {
		return Eval{}, fmt.Errorf("no held-out tokens")
	}
	if temp < 0 {
		return Eval{}, fmt.Errorf("temperature must be >= 0")
	}
	var ctx context
	for _, id := range left {
		ctx = ctx.push(id, m.order)
	}
	sum := 0.0
	oov := 0
	for _, sym := range symbols {
		id, ok := m.ids[sym]
		if !ok {
			id = unkID
			oov++
		}
		p := m.prob(ctx, id, temp)
		if p <= 0 || math.IsNaN(p) || math.IsInf(p, 0) {
			return Eval{}, fmt.Errorf("non-positive probability for %q", sym)
		}
		sum += -math.Log2(p)
		ctx = ctx.push(id, m.order)
	}
	bits := sum / float64(len(symbols))
	return Eval{
		Tokens:     len(symbols),
		Bits:       bits,
		Perplexity: math.Exp2(bits),
		OOV:        oov,
	}, nil
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
