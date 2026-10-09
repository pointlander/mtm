// Copyright 2026 The mtm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
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

func TestGreedyFollowsUniqueContext(t *testing.T) {
	m := train(t, 2, "one two three four one two three four")
	seed, err := m.Encode("one two")
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.Generate(rng(1), seed, 6, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := "one two three four one two three four"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
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
	if m.vocab[sym] != "blue" {
		t.Fatalf("wrote %q", m.vocab[sym])
	}
	if tape.cells[1] != sym {
		t.Fatal("symbol not on the tape")
	}
}

func TestBackoffAndBothSuccessors(t *testing.T) {
	m := train(t, 2, "alpha beta gamma alpha beta delta")
	ctx := context{n: 2, w: [maxOrder]uint32{m.ids["alpha"], m.ids["beta"]}}
	seen := map[string]int{}
	r := rng(7)
	for range 400 {
		id, ok := m.sample(ctx, r, 1)
		if !ok {
			t.Fatal("sample failed")
		}
		seen[m.vocab[id]]++
	}
	if seen["gamma"] == 0 || seen["delta"] == 0 {
		t.Fatalf("successors = %v", seen)
	}
	if len(seen) != 2 {
		t.Fatalf("unexpected successors %v", seen)
	}

	// An unseen context falls back to a shorter one that was learned.
	unseen := context{n: 2, w: [maxOrder]uint32{m.ids["delta"], m.ids["delta"]}}
	id, ok := m.sample(unseen, r, 0)
	if !ok {
		t.Fatal("backoff failed")
	}
	if _, known := m.ids[m.vocab[id]]; !known {
		t.Fatalf("wrote unknown symbol %q", m.vocab[id])
	}
}

func TestGreedyPrefersFrequentSuccessor(t *testing.T) {
	m := train(t, 1, "the cat the cat the dog")
	ctx := context{n: 1, w: [maxOrder]uint32{m.ids["the"]}}
	id, ok := m.sample(ctx, rng(1), 0)
	if !ok {
		t.Fatal("sample failed")
	}
	if m.vocab[id] != "cat" {
		t.Fatalf("got %q", m.vocab[id])
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
