// Copyright 2026 The mtm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"
	"testing"
)

func train(t *testing.T, order int, text string) *Machine {
	t.Helper()
	m, err := New(order)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Train(text); err != nil {
		t.Fatal(err)
	}
	return m
}

func rng(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
}

func TestKneserNeyDiscount(t *testing.T) {
	// "a b c a b c a b d" at order 2. The context (a, b) has raw counts
	// c:2 d:1, so D = n1/(n1+2*n2) = 1/7. Every shorter context was
	// continued only once, so those levels discount everything away and
	// the leftover mass is uniform over 4 symbols plus unknown:
	//   P(c | a b) = 67/105
	//   P(d | a b) = 32/105
	m := train(t, 2, "a b c a b c a b d")
	ctx := context{n: 2}
	ctx.w[0] = m.ids["a"]
	ctx.w[1] = m.ids["b"]
	pC := m.prob(ctx, m.ids["c"], 1)
	pD := m.prob(ctx, m.ids["d"], 1)
	if math.Abs(pC-67.0/105.0) > 1e-9 {
		t.Fatalf("P(c|a b)=%v want %v", pC, 67.0/105.0)
	}
	if math.Abs(pD-32.0/105.0) > 1e-9 {
		t.Fatalf("P(d|a b)=%v want %v", pD, 32.0/105.0)
	}
	if pC <= pD {
		t.Fatalf("frequent continuation lost: P(c)=%v P(d)=%v", pC, pD)
	}
}

func TestContinuationNotRawFrequency(t *testing.T) {
	// b occurs 3 times, always after x, so its continuation count is 1.
	// a occurs twice, after y and after z, so its continuation count is 2.
	// The order-1 backoff is the continuation unigram:
	//   P(a) = 17/54
	//   P(b) = 8/54
	m := train(t, 1, "x b x b x b y a z a")
	pA := m.prob(context{}, m.ids["a"], 1)
	pB := m.prob(context{}, m.ids["b"], 1)
	if math.Abs(pA-17.0/54.0) > 1e-9 {
		t.Fatalf("P(a)=%v want %v", pA, 17.0/54.0)
	}
	if math.Abs(pB-8.0/54.0) > 1e-9 {
		t.Fatalf("P(b)=%v want %v", pB, 8.0/54.0)
	}
	if pA <= pB {
		t.Fatalf("raw frequency won: P(a)=%v P(b)=%v", pA, pB)
	}
}

func TestHeadMovesRight(t *testing.T) {
	m := train(t, 1, "red blue red blue")
	tape := Tape{cells: []uint32{m.ids["red"]}, head: 1}
	sym, ok := m.Step(rng(1), &tape, 0)
	if !ok {
		t.Fatal("step failed")
	}
	if tape.head != 2 {
		t.Fatalf("head = %d", tape.head)
	}
	if int(sym) >= len(m.vocab) {
		t.Fatalf("wrote id %d", sym)
	}
	if tape.cells[1] != sym {
		t.Fatal("symbol not on the tape")
	}
	ctx := context{n: 1, w: [maxOrder]uint32{m.ids["red"]}}
	if m.prob(ctx, m.ids["blue"], 1) <= m.prob(ctx, m.ids["red"], 1) {
		t.Fatal("blue should be more likely after red")
	}
}

func TestBothSuccessorsStayLikely(t *testing.T) {
	m := train(t, 2, "alpha beta gamma alpha beta delta")
	ctx := context{n: 2, w: [maxOrder]uint32{m.ids["alpha"], m.ids["beta"]}}
	pGamma := m.prob(ctx, m.ids["gamma"], 1)
	pDelta := m.prob(ctx, m.ids["delta"], 1)
	if math.Abs(pGamma-pDelta) > 1e-12 {
		t.Fatalf("equal counts diverged: gamma %v delta %v", pGamma, pDelta)
	}
	if pGamma <= 0 {
		t.Fatal("missing successor mass")
	}
	seen := map[string]int{}
	r := rng(7)
	for range 400 {
		id, ok := m.sample(ctx, r, 1)
		if !ok {
			t.Fatal("sample failed")
		}
		seen[m.vocab[id]]++
	}
	if seen["gamma"] < 40 || seen["delta"] < 40 {
		t.Fatalf("successors = %v", seen)
	}

	// An unseen context still writes a known symbol.
	unseen := context{n: 2, w: [maxOrder]uint32{m.ids["delta"], m.ids["delta"]}}
	id, ok := m.sample(unseen, r, 0)
	if !ok {
		t.Fatal("backoff failed")
	}
	if int(id) >= len(m.vocab) {
		t.Fatalf("wrote id %d", id)
	}
}

func TestFrequentSuccessorMoreLikely(t *testing.T) {
	m := train(t, 1, "the cat the cat the dog")
	ctx := context{n: 1, w: [maxOrder]uint32{m.ids["the"]}}
	if m.prob(ctx, m.ids["cat"], 1) <= m.prob(ctx, m.ids["dog"], 1) {
		t.Fatal("cat should beat dog after the")
	}
}

func TestTemperatureChangesProbability(t *testing.T) {
	m := train(t, 1, "the cat the cat the dog")
	ctx := context{n: 1, w: [maxOrder]uint32{m.ids["the"]}}
	hot := m.prob(ctx, m.ids["cat"], 0.5)
	mid := m.prob(ctx, m.ids["cat"], 1)
	cold := m.prob(ctx, m.ids["cat"], 2)
	if hot == mid || mid == cold {
		t.Fatalf("temperature did not change P(cat|the): %v %v %v", hot, mid, cold)
	}
	if !(hot > mid && mid > cold) {
		t.Fatalf("lower temperature should favor the mode: %v %v %v", hot, mid, cold)
	}
}

func TestSameSeedSameText(t *testing.T) {
	const corpus = "To be or not to be that is the question whether tis nobler"
	a := train(t, 2, corpus)
	b := train(t, 2, corpus)
	left, err := a.Generate(rng(99), nil, 30, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	right, err := b.Generate(rng(99), nil, 30, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if left != right {
		t.Fatalf("seed diverged\n%q\n%q", left, right)
	}
	if left == "" {
		t.Fatal("empty generation")
	}
}

func TestStopsAtSentence(t *testing.T) {
	m := train(t, 1, "Hello world. Farewell world.")
	got, err := m.Generate(rng(3), nil, 40, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(got, ".") != 1 {
		t.Fatalf("got %q", got)
	}
}

func TestUnknownPrompt(t *testing.T) {
	m := train(t, 1, "to be or not")
	if _, err := m.Encode("banana"); err == nil {
		t.Fatal("expected unknown token")
	}
}

func TestScoreUnknownIsFinite(t *testing.T) {
	m := train(t, 1, "alpha beta alpha beta alpha")
	ev, err := m.Score([]uint32{m.ids["alpha"]}, []string{"beta", "notaword"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if ev.OOV != 1 || ev.Tokens != 2 || ev.KnownTokens != 1 {
		t.Fatalf("eval %+v", ev)
	}
	if math.IsNaN(ev.Bits) || math.IsInf(ev.Bits, 0) || ev.Bits <= 0 {
		t.Fatalf("bits %v", ev.Bits)
	}
	if math.IsNaN(ev.KnownBits) || math.IsInf(ev.KnownBits, 0) || ev.KnownBits <= 0 {
		t.Fatalf("known bits %v", ev.KnownBits)
	}
	if ev.KnownBits == ev.Bits {
		t.Fatalf("known bits should exclude the unknown token: %+v", ev)
	}
	if _, known := m.ids["notaword"]; known {
		t.Fatal("scoring interned an unknown token")
	}
}

func TestTrainAndScore(t *testing.T) {
	m, err := New(1)
	if err != nil {
		t.Fatal(err)
	}
	ev, ok, err := m.TrainAndScore("aa bb cc dd ee ff gg hh ii zz", 0.2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !ok || ev.Tokens != 2 || ev.OOV != 2 {
		t.Fatalf("ok %v eval %+v", ok, ev)
	}
	if m.tokens != 10 {
		t.Fatalf("tokens %d", m.tokens)
	}
	if _, err := m.Encode("zz"); err != nil {
		t.Fatal(err)
	}

	skipped, err := New(1)
	if err != nil {
		t.Fatal(err)
	}
	ev, ok, err = skipped.TrainAndScore("one two three", 0, 1)
	if err != nil || ok || ev.Tokens != 0 || skipped.tokens != 3 {
		t.Fatalf("ok %v eval %+v tokens %d err %v", ok, ev, skipped.tokens, err)
	}
	if _, _, err := skipped.TrainAndScore("one two", 1, 1); err == nil {
		t.Fatal("expected holdout error")
	}
}

func TestHeldOutPatternBeatsNoise(t *testing.T) {
	easyM, err := New(2)
	if err != nil {
		t.Fatal(err)
	}
	easy, ok, err := easyM.TrainAndScore(strings.Repeat("one two three ", 80), 0.1, 1)
	if err != nil || !ok {
		t.Fatal(err, ok)
	}
	if easy.OOV != 0 || easy.Bits > 1 {
		t.Fatalf("pattern %+v", easy)
	}

	hardM, err := New(2)
	if err != nil {
		t.Fatal(err)
	}
	// The last fifth is a vocabulary the prefix never used.
	hardText := strings.Repeat("one two three ", 80) + strings.Repeat("zz yy xx ", 20)
	hard, ok, err := hardM.TrainAndScore(hardText, 0.2, 1)
	if err != nil || !ok {
		t.Fatal(err, ok)
	}
	if hard.OOV != 60 {
		t.Fatalf("oov %d", hard.OOV)
	}
	if hard.Bits <= easy.Bits+2 {
		t.Fatalf("easy %v hard %v", easy.Bits, hard.Bits)
	}
}

func TestKnownProbUnchangedByChars(t *testing.T) {
	m := train(t, 1, "xx yy xx yy")
	if m.chars != nil {
		t.Fatal("Train trained the character model")
	}
	before := m.prob(context{}, m.ids["xx"], 1)
	if err := m.trainChars([]string{"ab", "ab", "cd"}); err != nil {
		t.Fatal(err)
	}
	after := m.prob(context{}, m.ids["xx"], 1)
	if before != after {
		t.Fatalf("known probability moved: %v -> %v", before, after)
	}
	// The end-of-word mark is the last symbol of each spelling, so it
	// never predicts the next word's first letter.
	eowID := m.chars.ids[eow]
	cross := context{n: 1, w: [maxOrder]uint32{eowID}}
	if d := m.chars.trans[cross]; d != nil && d.total > 0 {
		t.Fatal("character context crossed a word boundary")
	}
}

func TestOOVUsesSpelling(t *testing.T) {
	var b strings.Builder
	for i := range 30 {
		b.WriteByte('w')
		b.WriteString(strconv.Itoa(i))
		b.WriteByte(' ')
	}
	text := b.String()
	m := train(t, 1, text)
	words := make([]string, 40)
	for i := range words {
		words[i] = "ab"
	}
	if err := m.trainChars(words); err != nil {
		t.Fatal(err)
	}
	pBin := m.prob(context{}, unkID, 1)
	pSpell := m.spellProb("ab")
	p := m.probOOV(context{}, "ab", 1)
	want := pBin * float64(len(m.vocab)+1) * pSpell
	if math.Abs(p-want) > 1e-12 {
		t.Fatalf("p=%v want %v", p, want)
	}
	if p <= pBin {
		t.Fatalf("spelling should beat the unknown bin: spell %v bin %v oov %v", pSpell, pBin, p)
	}
	plain := train(t, 1, text)
	evPlain, err := plain.Score(nil, []string{"w0", "ab"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	ev, err := m.Score(nil, []string{"w0", "ab"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if ev.KnownBits != evPlain.KnownBits || ev.KnownTokens != 1 || ev.OOV != 1 {
		t.Fatalf("known %+v plain %+v", ev, evPlain)
	}
	if ev.Bits >= evPlain.Bits {
		t.Fatalf("spelling did not help: plain %v spelled %v", evPlain.Bits, ev.Bits)
	}
	if _, known := m.ids["ab"]; known {
		t.Fatal("scoring interned an unknown token")
	}
}

func TestFrequentSpellingMoreLikely(t *testing.T) {
	m := train(t, 1, "xx yy")
	words := make([]string, 0, 41)
	for range 40 {
		words = append(words, "cat")
	}
	words = append(words, "dog")
	if err := m.trainChars(words); err != nil {
		t.Fatal(err)
	}
	if m.spellProb("cat") <= m.spellProb("dog") {
		t.Fatalf("cat %v dog %v", m.spellProb("cat"), m.spellProb("dog"))
	}
}

func TestImplausibleSpellingKeepsBin(t *testing.T) {
	const text = "xx yy xx yy"
	m := train(t, 1, text)
	if err := m.trainChars([]string{"ab", "cd"}); err != nil {
		t.Fatal(err)
	}
	plain := train(t, 1, text)
	got, err := m.Score(nil, []string{"qqqq"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	want, err := plain.Score(nil, []string{"qqqq"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.Bits-want.Bits) > 1e-9 {
		t.Fatalf("implausible spelling changed the score: got %v want %v", got.Bits, want.Bits)
	}
}

func TestSpellPrefersTrainedWord(t *testing.T) {
	m := train(t, 1, "xx yy")
	words := make([]string, 40)
	for i := range words {
		words[i] = "cat"
	}
	if err := m.trainChars(words); err != nil {
		t.Fatal(err)
	}
	if m.spellProb("cat") <= m.spellProb("qqqq") {
		t.Fatalf("cat %v qqqq %v", m.spellProb("cat"), m.spellProb("qqqq"))
	}
}

func TestSpellNovelInterns(t *testing.T) {
	m := train(t, 1, "aa bb aa bb")
	words := make([]string, 50)
	for i := range words {
		words[i] = "cat"
	}
	if err := m.trainChars(words); err != nil {
		t.Fatal(err)
	}
	ctx := context{n: 1, w: [maxOrder]uint32{m.ids["aa"]}}
	total := m.trans[ctx].total
	before := len(m.vocab)
	saw := false
	r := rng(1)
	for range 400 {
		id := m.drawBase(r)
		if int(id) < before {
			continue
		}
		if m.vocab[id] == "" {
			t.Fatal("empty spelling")
		}
		saw = true
		break
	}
	if !saw {
		t.Fatal("base draw never spelled a new word")
	}
	if m.tokens != 4 {
		t.Fatalf("tokens %d", m.tokens)
	}
	if m.trans[ctx].total != total {
		t.Fatal("spelling a word counted it")
	}
}

func TestOrderRejected(t *testing.T) {
	if _, err := New(0); err == nil {
		t.Fatal("expected error")
	}
	if _, err := New(maxOrder + 1); err == nil {
		t.Fatal("expected error")
	}
}

func TestTokenizeShakespeareForms(t *testing.T) {
	got := Tokenize("know’st o’er never-resting ’tis i’ the")
	want := []string{"know’st", "o’er", "never-resting", "’tis", "i’", "the"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("got %q", got)
	}
}

func TestTokenizeStageAndNewlines(t *testing.T) {
	got := Tokenize("[_Aside._] Hello\n\n\n\nKING.")
	want := []string{"[", "Aside", ".", "]", "Hello", "\n", "\n", "KING", "."}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("got %#v", got)
	}
	text := joinTokens(got)
	if text != "[Aside.] Hello\n\nKING." {
		t.Fatalf("rendered %q", text)
	}
}

func TestQuotesAndElision(t *testing.T) {
	got := Tokenize("answer ‘This fair child of mine’.")
	text := joinTokens(got)
	if text != "answer ‘This fair child of mine’." {
		t.Fatalf("rendered %q from %#v", text, got)
	}
	got = Tokenize("i’ the sun ’tis true")
	text = joinTokens(got)
	if text != "i’ the sun ’tis true" {
		t.Fatalf("rendered %q from %#v", text, got)
	}
	if strings.Join(got, "|") != "i’|the|sun|’tis|true" {
		t.Fatalf("tokens %#v", got)
	}
}

func TestRenderSpacing(t *testing.T) {
	got := joinTokens([]string{"Hello", ",", "world", "!", "\n", "“", "had", "!", "”"})
	want := "Hello, world!\n“had!”"
	if got != want {
		t.Fatalf("got %q", got)
	}
}

func TestGutenbergStripped(t *testing.T) {
	const doc = "ZZZGUTENBERGZZZ stays out\n" +
		"*** START OF THE PROJECT GUTENBERG EBOOK THE COMPLETE WORKS ***\n" +
		"QUUXPLAY begins here\n" +
		"*** END OF THE PROJECT GUTENBERG EBOOK THE COMPLETE WORKS ***\n" +
		"www.gutenberg.org trailer\n"
	got := Tokenize(doc)
	joined := strings.Join(got, " ")
	if strings.Contains(joined, "ZZZGUTENBERGZZZ") || strings.Contains(joined, "gutenberg") {
		t.Fatalf("marker leaked into %q", got)
	}
	if !strings.Contains(joined, "QUUXPLAY") {
		t.Fatalf("body missing: %q", got)
	}
}

func TestPG100BodyOnly(t *testing.T) {
	b, err := os.ReadFile("pg100.txt")
	if err != nil {
		t.Skip(err)
	}
	toks := Tokenize(string(b))
	if len(toks) < 100000 {
		t.Fatalf("too few tokens: %d", len(toks))
	}
	for _, tok := range toks {
		if strings.Contains(strings.ToLower(tok), "gutenberg") {
			t.Fatalf("header leaked: %q", tok)
		}
	}
}
