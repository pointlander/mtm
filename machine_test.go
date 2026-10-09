// Copyright 2026 The mtm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand/v2"
	"os"
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

func TestWittenBellOnceSeen(t *testing.T) {
	// "a b a c" at order 2. The context (a, b) was seen once, followed
	// by a. Witten-Bell keeps half the mass on that continuation and
	// interpolates the rest:
	//   P(a | a b) = 95/112
	//   P(b | a b) = 1/16
	m := train(t, 2, "a b a c")
	ctx := context{n: 2}
	ctx.w[0] = m.ids["a"]
	ctx.w[1] = m.ids["b"]
	pA := m.prob(ctx, m.ids["a"], 1)
	pB := m.prob(ctx, m.ids["b"], 1)
	if math.Abs(pA-95.0/112.0) > 1e-9 {
		t.Fatalf("P(a|a b)=%v want %v", pA, 95.0/112.0)
	}
	if math.Abs(pB-1.0/16.0) > 1e-9 {
		t.Fatalf("P(b|a b)=%v want %v", pB, 1.0/16.0)
	}
	if !(pA < 1 && pA > pB) {
		t.Fatalf("once-seen continuation was not mixed: P(a)=%v P(b)=%v", pA, pB)
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
	if ev.OOV != 1 || ev.Tokens != 2 {
		t.Fatalf("eval %+v", ev)
	}
	if math.IsNaN(ev.Bits) || math.IsInf(ev.Bits, 0) || ev.Bits <= 0 {
		t.Fatalf("bits %v", ev.Bits)
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
