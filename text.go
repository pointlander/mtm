// Copyright 2026 The mtm Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"strings"
	"unicode"
)

// stripGutenberg drops the Project Gutenberg header and footer when the
// standard markers are present. Text without them is returned unchanged.
func stripGutenberg(s string) string {
	const start = "*** START OF THE PROJECT GUTENBERG EBOOK"
	const end = "*** END OF THE PROJECT GUTENBERG EBOOK"
	if i := strings.Index(s, start); i >= 0 {
		rest := s[i+len(start):]
		if nl := strings.IndexByte(rest, '\n'); nl >= 0 {
			s = rest[nl+1:]
		} else {
			s = rest
		}
	}
	if i := strings.Index(s, end); i >= 0 {
		s = s[:i]
	}
	return s
}

// Tokenize splits text into the machine's tape symbols: words,
// punctuation, and newlines. Apostrophes and hyphens stay inside a
// word when they join letters ("know’st", "o’er", "’tis"), and a
// trailing apostrophe stays on the word when it is an elision ("i’ the").
// Opening and closing quotes are their own symbols. Gutenberg italic
// underscores are markup and are removed. Runs of newlines are kept,
// capped at two, so verse breaks and scene breaks remain distinct
// without copying large vertical gaps.
func Tokenize(text string) []string {
	text = stripGutenberg(text)
	text = strings.ReplaceAll(text, "_", "")
	runes := []rune(text)
	tokens := make([]string, 0, len(runes)/5)
	newlines := 0
	flushNL := func() {
		if newlines == 0 {
			return
		}
		if newlines > 2 {
			newlines = 2
		}
		for range newlines {
			tokens = append(tokens, "\n")
		}
		newlines = 0
	}
	for i := 0; i < len(runes); {
		r := runes[i]
		switch {
		case r == '\n':
			newlines++
			i++
		case r == '\r':
			i++
		case unicode.IsSpace(r):
			flushNL()
			i++
		case wordStart(runes, i):
			flushNL()
			j := scanWord(runes, i)
			tokens = append(tokens, string(runes[i:j]))
			i = j
		default:
			flushNL()
			tokens = append(tokens, string(r))
			i++
		}
	}
	flushNL()
	return tokens
}

func wordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r)
}

func joiner(r rune) bool {
	switch r {
	case '\'', '’', '-', '‐', '‑':
		return true
	default:
		return false
	}
}

func apostrophe(r rune) bool {
	return r == '\'' || r == '’'
}

func wordStart(runes []rune, i int) bool {
	r := runes[i]
	if wordRune(r) {
		return true
	}
	return joiner(r) && i+1 < len(runes) && wordRune(runes[i+1])
}

func scanWord(runes []rune, i int) int {
	j := i
	for j < len(runes) {
		r := runes[j]
		if wordRune(r) {
			j++
			continue
		}
		if joiner(r) && j+1 < len(runes) && wordRune(runes[j+1]) {
			j++
			continue
		}
		// "i’ the" keeps the apostrophe; "mine’." leaves it as a quote.
		if apostrophe(r) && j > i && nextWordAfter(runes, j+1) {
			j++
		}
		break
	}
	return j
}

func nextWordAfter(runes []rune, j int) bool {
	for j < len(runes) {
		r := runes[j]
		if r == '\n' || r == '\r' {
			return false
		}
		if unicode.IsSpace(r) {
			j++
			continue
		}
		return wordRune(r)
	}
	return false
}

func isSentenceEnd(s string) bool {
	return s == "." || s == "!" || s == "?"
}

func noSpaceBefore(s string) bool {
	switch s {
	case ".", ",", ";", ":", "!", "?", ")", "]", "}", "”", "’", "'", "%", "—", "–", "-":
		return true
	default:
		return false
	}
}

func noSpaceAfter(s string) bool {
	switch s {
	case "(", "[", "{", "“", "‘", "—", "–", "-":
		return true
	default:
		return false
	}
}

func spaceBetween(prev, cur string) bool {
	if prev == "\n" || cur == "\n" {
		return false
	}
	if noSpaceBefore(cur) || noSpaceAfter(prev) {
		return false
	}
	return true
}

func joinTokens(tokens []string) string {
	var b strings.Builder
	n := 0
	for _, s := range tokens {
		n += len(s) + 1
	}
	b.Grow(n)
	for i, s := range tokens {
		if i > 0 && spaceBetween(tokens[i-1], s) {
			b.WriteByte(' ')
		}
		b.WriteString(s)
	}
	return b.String()
}

// prime is a one-symbol left context that begins a line, when the
// training text contained newlines. It is not part of the prompt.
func prime(m *Machine) []uint32 {
	id, ok := m.ids["\n"]
	if !ok {
		return nil
	}
	return []uint32{id}
}
