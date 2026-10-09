// Copyright 2026 The mtm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
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
	// eow ends a word in the character model. Private use keeps it out
	// of any training text.
	eow = "\uE000"
	// charOrder is the Markov order of the model that spells a word
	// the word model has never seen. That model discounts raw counts
	// at every length, so a frequent spelling stays frequent.
	charOrder = 5
	// maxSpell is the longest spelling the character model will write.
	maxSpell = 64
	// spellTries is how many spellings to draw before giving up on a
	// word that is not already in the vocabulary.
	spellTries = 8
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
	zD    float64
	zLog  float64
	zOK   bool
}

// add records one observation. first is true when id was not already a successor.
func (d *dist) add(id uint32) bool {
	d.zOK = false
	if d.index != nil {
		if i, ok := d.index[id]; ok {
			d.cnt[i]++
			d.total++
			return false
		}
		d.index[id] = uint32(len(d.sym))
		d.sym = append(d.sym, id)
		d.cnt = append(d.cnt, 1)
		d.total++
		return true
	}
	for i, s := range d.sym {
		if s == id {
			d.cnt[i]++
			d.total++
			return false
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
	return true
}

// discMass is the absolute-discounted count, max(count-D, 0).
func discMass(count uint32, D float64) float64 {
	if count == 0 {
		return 0
	}
	w := float64(count) - D
	if w < 0 {
		return 0
	}
	return w
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

// discDenom is the sum of discounted counts when every count is at least 1
// and D is in [0, 1], which is total - D * (number of successors).
func (d *dist) discDenom(D float64) float64 {
	return float64(d.total) - D*float64(len(d.sym))
}

// logDiscPartition is log(sum max(count-D, 0)^(1/temp)).
func (d *dist) logDiscPartition(D, temp float64) float64 {
	if d.zOK && d.zTemp == temp && d.zD == D {
		return d.zLog
	}
	inv := 1 / temp
	maxLog := math.Inf(-1)
	logs := make([]float64, len(d.cnt))
	for i, c := range d.cnt {
		w := discMass(c, D)
		if w <= 0 {
			logs[i] = math.Inf(-1)
			continue
		}
		logs[i] = inv * math.Log(w)
		if logs[i] > maxLog {
			maxLog = logs[i]
		}
	}
	sum := 0.0
	for _, lg := range logs {
		if math.IsInf(lg, -1) {
			continue
		}
		sum += math.Exp(lg - maxLog)
	}
	d.zTemp = temp
	d.zD = D
	if sum <= 0 || math.IsInf(maxLog, -1) {
		d.zOK = false
		d.zLog = math.Inf(-1)
		return d.zLog
	}
	d.zLog = maxLog + math.Log(sum)
	d.zOK = true
	return d.zLog
}

// discMLE is the tempered distribution over discounted counts.
// Unseen symbols and counts that discount to zero get 0. Temperature 1
// is max(count-D, 0) over the discounted total. Temperature 0 puts
// equal mass on the modes.
func (d *dist) discMLE(id uint32, D, temp float64) float64 {
	c, ok := d.countOf(id)
	w := 0.0
	if ok {
		w = discMass(c, D)
	}
	if w <= 0 {
		return 0
	}
	if temp < 1e-3 {
		best := uint32(0)
		for _, n := range d.cnt {
			if n > best && discMass(n, D) > 0 {
				best = n
			}
		}
		if c != best {
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
		den := d.discDenom(D)
		if den <= 0 {
			return 0
		}
		return w / den
	}
	lg := d.logDiscPartition(D, temp)
	if math.IsInf(lg, 0) || math.IsNaN(lg) {
		return 0
	}
	return math.Exp(math.Log(w)/temp - lg)
}

// sampleDisc draws from the tempered discounted distribution.
// Temperature 0 picks the mode, and the earliest observation wins a tie.
func (d *dist) sampleDisc(rng *rand.Rand, D, temp float64) uint32 {
	if len(d.sym) == 0 {
		return 0
	}
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
		den := d.discDenom(D)
		if den <= 0 {
			return d.sym[0]
		}
		pick := rng.Float64() * den
		run := 0.0
		for i, c := range d.cnt {
			run += discMass(c, D)
			if pick < run {
				return d.sym[i]
			}
		}
		return d.sym[len(d.sym)-1]
	}
	inv := 1 / temp
	maxLog := math.Inf(-1)
	weights := make([]float64, len(d.cnt))
	for i, c := range d.cnt {
		w := discMass(c, D)
		if w <= 0 {
			continue
		}
		lg := inv * math.Log(w)
		if lg > maxLog {
			maxLog = lg
		}
		weights[i] = lg
	}
	if math.IsInf(maxLog, -1) {
		return d.sym[0]
	}
	sum := 0.0
	for i, lg := range weights {
		if discMass(d.cnt[i], D) <= 0 {
			continue
		}
		weights[i] = math.Exp(lg - maxLog)
		sum += weights[i]
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
// The symbol is drawn from interpolated Kneser-Ney. The longest context
// discounts its raw counts by a fixed D and gives the removed mass to
// the shorter suffix. Every shorter suffix is estimated from continuation
// counts: how many distinct longer contexts produced that symbol, not
// how often the symbol itself occurred. The shortest distribution backs
// off to a uniform bin over the known symbols and one unknown symbol.
// A character model can replace that bin with the probability of
// spelling the unknown word, including an end-of-word mark, when the
// spelling is more probable than the bin. Known symbols stay on this
// distribution. Temperature reshapes the discounted counts at each
// level and does not change how much mass escapes. It does not reshape
// spellings.
type Machine struct {
	order      int
	vocab      []string
	ids        map[string]uint32
	trans      map[context]*dist
	cont       map[context]*dist
	disc       [maxOrder + 1]float64
	discOK     bool
	tokens     int
	rawLower   bool
	chars      *Machine
	spellCache map[string]float64
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
		cont:  make(map[context]*dist),
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
		first := m.addRaw(ctx, sym)
		// A first sighting of this context is one distinct left extension
		// of the suffix, which is what the lower-order model counts.
		// A raw lower order keeps the counts instead, so frequency survives.
		if first && ctx.n > 0 && !m.rawLower {
			m.addCont(ctx.dropOldest(), sym)
		}
		if ctx.n == 0 {
			return
		}
		ctx = ctx.dropOldest()
	}
}

func (m *Machine) addRaw(ctx context, sym uint32) bool {
	m.discOK = false
	d := m.trans[ctx]
	if d == nil {
		d = &dist{}
		m.trans[ctx] = d
	}
	return d.add(sym)
}

func (m *Machine) addCont(ctx context, sym uint32) {
	m.discOK = false
	d := m.cont[ctx]
	if d == nil {
		d = &dist{}
		m.cont[ctx] = d
	}
	d.add(sym)
}

// ensureDiscount estimates one absolute discount per context length.
// D = n1 / (n1 + 2*n2), with n1 and n2 the number of successors seen
// once and twice. The longest context uses raw counts. Shorter contexts
// use continuation counts, unless this model keeps raw counts at every
// length. A length with no singletons keeps a discount of 0.5 so an
// unseen symbol still escapes.
func (m *Machine) ensureDiscount() {
	if m.discOK {
		return
	}
	var n1, n2 [maxOrder + 1]int
	for ctx, d := range m.trans {
		if !m.rawLower && int(ctx.n) != m.order {
			continue
		}
		for _, c := range d.cnt {
			switch c {
			case 1:
				n1[ctx.n]++
			case 2:
				n2[ctx.n]++
			}
		}
	}
	if !m.rawLower {
		for ctx, d := range m.cont {
			for _, c := range d.cnt {
				switch c {
				case 1:
					n1[ctx.n]++
				case 2:
					n2[ctx.n]++
				}
			}
		}
	}
	for n := 0; n <= m.order; n++ {
		// No singletons means the usual estimate is 0 and an unseen
		// symbol would get probability 0. Keep a discount so the
		// backoff path stays open.
		if n1[n] == 0 {
			m.disc[n] = 0.5
			continue
		}
		m.disc[n] = float64(n1[n]) / float64(n1[n]+2*n2[n])
	}
	m.discOK = true
}

// baseProb is the 0-gram: one bin per known symbol, plus a bin for a
// symbol the machine has never seen. Every symbol, known or not, gets
// the same share, so a held-out word still has a finite probability.
func (m *Machine) baseProb() float64 {
	return 1 / float64(len(m.vocab)+1)
}

// levelDist is the raw distribution at the longest context and the
// continuation distribution at every shorter one. Raw lower orders
// use the raw distribution at every length.
func (m *Machine) levelDist(ctx context) *dist {
	if m.rawLower || int(ctx.n) == m.order {
		return m.trans[ctx]
	}
	return m.cont[ctx]
}

func (m *Machine) escape(ctx context, d *dist) float64 {
	lambda := m.disc[ctx.n] * float64(len(d.sym)) / float64(d.total)
	if lambda > 1 {
		return 1
	}
	return lambda
}

// prob is the interpolated Kneser-Ney probability of sym in ctx.
func (m *Machine) prob(ctx context, sym uint32, temp float64) float64 {
	m.ensureDiscount()
	return m.probFrom(ctx, sym, temp)
}

func (m *Machine) probFrom(ctx context, sym uint32, temp float64) float64 {
	d := m.levelDist(ctx)
	if d == nil || d.total == 0 || len(d.sym) == 0 {
		if ctx.n == 0 {
			return m.baseProb()
		}
		return m.probFrom(ctx.dropOldest(), sym, temp)
	}
	lambda := m.escape(ctx, d)
	var lower float64
	if lambda > 0 {
		if ctx.n == 0 {
			lower = m.baseProb()
		} else {
			lower = m.probFrom(ctx.dropOldest(), sym, temp)
		}
	}
	if lambda >= 1 {
		return lower
	}
	return (1-lambda)*d.discMLE(sym, m.disc[ctx.n], temp) + lambda*lower
}

func (m *Machine) sample(ctx context, rng *rand.Rand, temp float64) (uint32, bool) {
	if len(m.vocab) == 0 {
		return 0, false
	}
	m.ensureDiscount()
	return m.sampleFrom(ctx, rng, temp), true
}

// sampleFrom draws from the same interpolation as prob. At the empty
// context the word model can spend its unknown-symbol slot on a
// spelling it has not written before.
func (m *Machine) sampleFrom(ctx context, rng *rand.Rand, temp float64) uint32 {
	d := m.levelDist(ctx)
	if d == nil || d.total == 0 || len(d.sym) == 0 {
		if ctx.n == 0 {
			return m.drawBase(rng)
		}
		return m.sampleFrom(ctx.dropOldest(), rng, temp)
	}
	lambda := m.escape(ctx, d)
	if lambda >= 1 || rng.Float64() < lambda {
		if ctx.n == 0 {
			return m.drawBase(rng)
		}
		return m.sampleFrom(ctx.dropOldest(), rng, temp)
	}
	return d.sampleDisc(rng, m.disc[ctx.n], temp)
}

// drawBase samples the 0-gram. Without a character model the draw stays
// inside the vocabulary. With one, a single extra slot spells a new word.
func (m *Machine) drawBase(rng *rand.Rand) uint32 {
	n := len(m.vocab)
	if n == 0 {
		return 0
	}
	if m.chars == nil {
		return uint32(rng.IntN(n))
	}
	slot := rng.IntN(n + 1)
	if slot < n {
		return uint32(slot)
	}
	if id, ok := m.spellNovel(rng); ok {
		return id
	}
	return uint32(rng.IntN(n))
}

// spellNovel draws a word from the character model and interns it when
// the spelling is new. Interning does not count the word.
func (m *Machine) spellNovel(rng *rand.Rand) (uint32, bool) {
	if m.chars == nil {
		return 0, false
	}
	for range spellTries {
		w := m.chars.sampleWord(rng)
		if w == "" {
			continue
		}
		if _, known := m.ids[w]; known {
			continue
		}
		return m.intern(w), true
	}
	return 0, false
}

// sampleWord writes runes until the end-of-word mark. The word model
// calls this at temperature 1, whatever temperature it is sampling at.
func (m *Machine) sampleWord(rng *rand.Rand) string {
	var ctx context
	var b strings.Builder
	for range maxSpell {
		id, ok := m.sample(ctx, rng, 1)
		if !ok {
			return ""
		}
		s := m.vocab[id]
		if s == eow {
			return b.String()
		}
		b.WriteString(s)
		ctx = ctx.push(id, m.order)
	}
	return ""
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

// Train tokenizes text and learns the word model. The character model
// stays off, so word probabilities keep the uniform unknown bin.
func (m *Machine) Train(text string) error {
	parts := Tokenize(text)
	if len(parts) == 0 {
		return fmt.Errorf("training text has no tokens")
	}
	m.learnParts(parts)
	return nil
}

// trainChars learns how each token is spelled. Each token is its own
// tape, so the character context does not cross a word boundary.
// Counts stay raw at every length: a repeated spelling stays frequent.
// Spellings are learned again after a held-out score, on the same
// schedule as the word counts.
func (m *Machine) trainChars(parts []string) error {
	if m.chars == nil {
		ch, err := New(charOrder)
		if err != nil {
			return err
		}
		ch.rawLower = true
		m.chars = ch
	}
	m.spellCache = nil
	buf := make([]uint32, 0, 16)
	end := m.chars.intern(eow)
	for _, word := range parts {
		buf = buf[:0]
		for _, r := range word {
			buf = append(buf, m.chars.intern(string(r)))
		}
		buf = append(buf, end)
		m.chars.Learn(buf)
	}
	return nil
}

// spellProb is the character-model probability of writing this spelling
// and then the end-of-word mark, from an empty character context.
func (m *Machine) spellProb(word string) float64 {
	if m.chars == nil {
		return 0
	}
	if m.spellCache == nil {
		m.spellCache = make(map[string]float64)
	} else if p, ok := m.spellCache[word]; ok {
		return p
	}
	p := m.chars.sequenceProb(word)
	m.spellCache[word] = p
	return p
}

// sequenceProb multiplies the character conditionals of word and the
// end-of-word mark. An unseen rune uses the unknown bin.
func (m *Machine) sequenceProb(word string) float64 {
	var ctx context
	logp := 0.0
	for _, r := range word {
		id, ok := m.ids[string(r)]
		if !ok {
			id = unkID
		}
		p := m.prob(ctx, id, 1)
		if p <= 0 || math.IsNaN(p) || math.IsInf(p, 0) {
			return 0
		}
		logp += math.Log(p)
		ctx = ctx.push(id, m.order)
	}
	id, ok := m.ids[eow]
	if !ok {
		id = unkID
	}
	p := m.prob(ctx, id, 1)
	if p <= 0 || math.IsNaN(p) || math.IsInf(p, 0) {
		return 0
	}
	logp += math.Log(p)
	out := math.Exp(logp)
	if out == 0 || math.IsNaN(out) {
		return math.SmallestNonzeroFloat64
	}
	return out
}

// probOOV is the probability of a spelling the word model has not
// interned. The unknown bin is escaped mass times 1/(|V|+1). Multiplying
// back by |V|+1 leaves that mass, and the character model spends it on
// this spelling. Known words are not renormalized, so their probabilities
// stay put. The result is at most the escaped mass.
func (m *Machine) probOOV(ctx context, spelling string, temp float64) float64 {
	return m.prob(ctx, unkID, temp) * float64(len(m.vocab)+1) * m.spellProb(spelling)
}

// Eval is the average next-symbol score of a held-out slice.
// Bits counts every token. KnownBits counts only symbols the prefix
// had already interned, so unknown names do not hide that average.
type Eval struct {
	Tokens      int
	Bits        float64
	Perplexity  float64
	KnownTokens int
	KnownBits   float64
	OOV         int
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
		if err := m.trainChars(parts); err != nil {
			return Eval{}, false, err
		}
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
	if err := m.trainChars(parts[:cut]); err != nil {
		return Eval{}, false, err
	}
	left := prefix
	if len(left) > m.order {
		left = append([]uint32(nil), left[len(left)-m.order:]...)
	}
	ev, err := m.Score(left, parts[cut:], temp)
	if err != nil {
		return Eval{}, false, err
	}
	m.learnParts(parts[cut:])
	if err := m.trainChars(parts[cut:]); err != nil {
		return Eval{}, false, err
	}
	return ev, true, nil
}

// Score reports the mean negative log2 probability of each symbol in
// symbols, conditioned on left and then on the symbols already scored.
// A string the machine has not interned is spelled when that spelling
// beats the unknown bin, and otherwise keeps the bin. Spelling a new
// name is usually the more expensive of the two.
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
	knownSum := 0.0
	known := 0
	oov := 0
	for _, sym := range symbols {
		id, ok := m.ids[sym]
		var p float64
		if !ok {
			id = unkID
			oov++
			p = m.prob(ctx, unkID, temp)
			if m.chars != nil {
				if spelled := m.probOOV(ctx, sym, temp); spelled > p {
					p = spelled
				}
			}
		} else {
			p = m.prob(ctx, id, temp)
		}
		if p <= 0 || math.IsNaN(p) || math.IsInf(p, 0) {
			return Eval{}, fmt.Errorf("non-positive probability for %q", sym)
		}
		bits := -math.Log2(p)
		sum += bits
		if ok {
			knownSum += bits
			known++
		}
		ctx = ctx.push(id, m.order)
	}
	avg := sum / float64(len(symbols))
	knownBits := 0.0
	if known > 0 {
		knownBits = knownSum / float64(known)
	}
	return Eval{
		Tokens:      len(symbols),
		Bits:        avg,
		Perplexity:  math.Exp2(avg),
		KnownTokens: known,
		KnownBits:   knownBits,
		OOV:         oov,
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
