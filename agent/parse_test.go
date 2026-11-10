package agent

import (
	"context"
	"testing"
)

func TestParseFactoryCanonicalizesValidSpecifications(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		canonical string
	}{
		{
			name:      "idle",
			input:     "  idle-v1  ",
			canonical: "idle-v1",
		},
		{
			name:      "random unordered parameters and exponent",
			input:     "random-v1:max_fraction=8e-1,launch_probability=0.30,min_fraction=2e-1",
			canonical: "random-v1:launch_probability=0.3,max_fraction=0.8,min_fraction=0.2",
		},
		{
			name:      "heuristic unordered parameters",
			input:     "heuristic-v1:efficiency=0.5,economy=1,defense=0.10,aggression=1.2",
			canonical: "heuristic-v1:aggression=1.2,defense=0.1,economy=1,efficiency=0.5",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			factory, err := ParseFactory(test.input)
			if err != nil {
				t.Fatal(err)
			}
			if got := factory.Spec().String(); got != test.canonical {
				t.Fatalf("Spec = %q, want %q", got, test.canonical)
			}
			created, err := factory.New(context.Background(), AgentInit{AgentSeed: 17})
			if err != nil {
				t.Fatal(err)
			}
			if got := created.ID(); got != factory.Spec().ID() {
				t.Fatalf("agent ID = %q, factory ID = %q", got, factory.Spec().ID())
			}

			roundTripped, err := ParseFactory(factory.Spec().String())
			if err != nil {
				t.Fatalf("parse canonical specification: %v", err)
			}
			if got := roundTripped.Spec().String(); got != test.canonical {
				t.Fatalf("round-tripped Spec = %q, want %q", got, test.canonical)
			}
		})
	}
}

func TestParseFactoryRejectsInvalidSpecifications(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"empty", ""},
		{"whitespace", "   "},
		{"unknown agent", "unknown-v1"},
		{"unknown agent with parameters", "unknown-v1:value=1"},
		{"idle empty parameter list", "idle-v1:"},
		{"idle parameter", "idle-v1:value=1"},
		{"random missing parameter list", "random-v1"},
		{"random empty parameter list", "random-v1:"},
		{"random missing parameter", "random-v1:launch_probability=0.3,min_fraction=0.2"},
		{"random unknown parameter", "random-v1:launch_probability=0.3,min_fraction=0.2,max_fraction=0.8,extra=1"},
		{"random duplicate parameter", "random-v1:launch_probability=0.3,min_fraction=0.2,max_fraction=0.8,min_fraction=0.4"},
		{"assignment missing equals", "random-v1:launch_probability=0.3,min_fraction,max_fraction=0.8"},
		{"assignment missing name", "random-v1:launch_probability=0.3,=0.2,max_fraction=0.8"},
		{"assignment missing value", "random-v1:launch_probability=,min_fraction=0.2,max_fraction=0.8"},
		{"assignment extra equals", "random-v1:launch_probability==0.3,min_fraction=0.2,max_fraction=0.8"},
		{"trailing comma", "random-v1:launch_probability=0.3,min_fraction=0.2,max_fraction=0.8,"},
		{"invalid float", "random-v1:launch_probability=often,min_fraction=0.2,max_fraction=0.8"},
		{"inner whitespace", "random-v1:launch_probability=0.3, min_fraction=0.2,max_fraction=0.8"},
		{"nan", "random-v1:launch_probability=NaN,min_fraction=0.2,max_fraction=0.8"},
		{"infinity", "random-v1:launch_probability=+Inf,min_fraction=0.2,max_fraction=0.8"},
		{"probability out of range", "random-v1:launch_probability=1.1,min_fraction=0.2,max_fraction=0.8"},
		{"reversed fractions", "random-v1:launch_probability=0.3,min_fraction=0.9,max_fraction=0.8"},
		{"heuristic missing parameter", "heuristic-v1:aggression=1,economy=1,efficiency=1"},
		{"heuristic unknown parameter", "heuristic-v1:aggression=1,economy=1,efficiency=1,defense=0,extra=1"},
		{"heuristic negative weight", "heuristic-v1:aggression=-1,economy=1,efficiency=1,defense=0"},
		{"heuristic defense out of range", "heuristic-v1:aggression=1,economy=1,efficiency=1,defense=1.1"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if factory, err := ParseFactory(test.input); err == nil {
				t.Fatalf("ParseFactory(%q) = %#v, want error", test.input, factory)
			}
		})
	}
}
