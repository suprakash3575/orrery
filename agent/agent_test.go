package agent

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func TestSpecCanonicalIdentity(t *testing.T) {
	parameters := map[string]string{
		"economy":    "1",
		"aggression": "1.5",
		"defense":    "0.2",
	}
	spec := NewSpec("heuristic-v1", parameters)

	if got, want := spec.ID(), "heuristic-v1"; got != want {
		t.Fatalf("ID = %q, want %q", got, want)
	}
	if got, want := spec.String(), "heuristic-v1:aggression=1.5,defense=0.2,economy=1"; got != want {
		t.Fatalf("String = %q, want %q", got, want)
	}

	// Spec must own its canonical values. Neither the input map nor a slice
	// returned by Parameters may be able to rewrite its identity afterward.
	parameters["aggression"] = "99"
	gotParameters := spec.Parameters()
	gotParameters[0].Value = "99"
	wantParameters := []Parameter{
		{Name: "aggression", Value: "1.5"},
		{Name: "defense", Value: "0.2"},
		{Name: "economy", Value: "1"},
	}
	if got := spec.Parameters(); !reflect.DeepEqual(got, wantParameters) {
		t.Fatalf("Parameters = %#v, want %#v", got, wantParameters)
	}
	if got, want := spec.String(), "heuristic-v1:aggression=1.5,defense=0.2,economy=1"; got != want {
		t.Fatalf("String after attempted mutation = %q, want %q", got, want)
	}
}

func TestSpecWithoutParametersUsesID(t *testing.T) {
	spec := NewSpec(IdleV1, nil)
	if got := spec.String(); got != IdleV1 {
		t.Fatalf("String = %q, want %q", got, IdleV1)
	}
	if got := spec.Parameters(); len(got) != 0 {
		t.Fatalf("Parameters = %#v, want empty", got)
	}
}

func TestIdleFactoryAndAgent(t *testing.T) {
	factory := IdleFactory{}
	if got := factory.Spec().String(); got != IdleV1 {
		t.Fatalf("factory spec = %q, want %q", got, IdleV1)
	}

	created, err := factory.New(context.Background(), AgentInit{PlayerID: 2})
	if err != nil {
		t.Fatal(err)
	}
	if got := created.ID(); got != IdleV1 {
		t.Fatalf("agent ID = %q, want %q", got, IdleV1)
	}
	decision, err := created.Decide(context.Background(), TurnRequest{PlayerID: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(decision.Actions) != 0 {
		t.Fatalf("idle actions = %#v, want none", decision.Actions)
	}
}

func TestIdleHonorsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := (IdleFactory{}).New(ctx, AgentInit{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("factory error = %v, want context.Canceled", err)
	}
	if _, err := (IdleAgent{}).Decide(ctx, TurnRequest{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("agent error = %v, want context.Canceled", err)
	}
}
