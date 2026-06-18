# AI Algorithms from Scratch

Curated collection of artificial-intelligence coursework implementations focused on optimization, competitive learning, reinforcement/game logic, and deep-learning experiments.

The repository keeps representative scripts rather than the full semester folder. Large datasets, trained weights, cache files, and local environment artifacts are intentionally excluded.

## Contents

```text
optimization/
  pso/basic_pso.py
  pso/neighborhood_pso_visual.py
  aco/ant_colony_route.py
  aco/metro_data.py
  aco/distances.py
  gwo/gwo_benchmarks.py

competitive_learning/
  hybrid_lvq.py
  instar_letter_recognition.py

deep_learning/
  transfer_learning_dog_cat.py

games/
  dots_and_boxes_pygame.py
```

## Topics covered

- Particle Swarm Optimization on multimodal benchmark functions.
- Ant Colony Optimization for route search over a metro-style graph.
- Grey Wolf Optimization over benchmark functions.
- Hybrid LVQ and Instar-style competitive learning.
- Transfer-learning workflow for a dog/cat classifier.
- Pygame implementation of Dots and Boxes.

## Setup

Install the common dependencies:

```bash
pip install -r requirements.txt
```

Some scripts require optional dependencies such as PyTorch, torchvision, pygame, or scikit-image depending on the example being run.

## Notes

- The scripts are research/coursework prototypes and are kept close to their original implementation style.
- Dataset-dependent examples expect local data and do not include private datasets or large trained model weights.
- The Hybrid LVQ code is also presented as a standalone repository in `hybrid-lvq-prototype`.
