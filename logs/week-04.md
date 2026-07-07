# Week 4

**Dates:** 06-29 to 07-06

## Goals

- Actually begin preliminary experimentation with very basic pipeline to test feasibility of research question
- Talk to Dr.Lin and Dr.Pestourie (Georgia Tech Professor) about formulated research question.
- Get into more specifics about what are the parameters of the experiment.
- Main Goal: Bridge the gap between the theoretical quantum mechanics theory (Wave-functions, density matrics, operators, etc) and see how quantum computing alters it in a purely theoretical sense. Also, see how to use quantum mechanics equations (and perhaps RL optimization) to Scientifc ML (PDE optimization on top of regular AI).

## Approach and Implementation

I looked over my Canvas class and some of my previous projects in Scientific ML to see where this latent space model can be applied. On top of taking notes on differentiability and PDEs and AI, I looked at PI-RL and PI-VAE as the encoders for the compact latent space. I still have not yet decided on surrogate model.

Dr.Pestourie said that there has been pushback within the SciML community for combining classical solvers with AI, since purely AI models ad simulators are not as efficient or the best use of time instead of utilizing the best of both worlds. After my research and notes on the topic, I concurred with his opinion and insight.

I also say how this task-distribution problem I talked about last log also lined up with Meta-RL, which is where I am also testing if PI-RL can be used to solve this, instead of just a VAE model with a soft-constrained PDE encoding within the loss function. I am still not sure if I should go down this rought, as it may involve heavy control theory, which i am not that confortable with, but I will look into it.

Also, I talked to Dr.Lin, and she said it was completely fine to switch to this topic.

I talk about this more in the latex document, but as for the specifics for my research topic, for this summer, I will look into problems where the circuit architecture (3 vs 4 qubits) and noise channels (Depolarization Noise vs Phase Dampening). I will keep the measurement operator/channel the same for now. I have not decided what measurement channel to use yet, but most likely it will be the simplest, which is a collection of orthogonal projectors.

## Results

- Wrote into my latex documents more notes pertaining to my new research topic, especially about the specific parameters and limitations of the study.
- Wrote more notes on Ipad on PDES and Differentiability, Quantum CS, Quantum Mechanics, and Physics 2.
- Also wrote on Latex document about connecting the quantum mechanics theory to quantum CS, since my research heavily involves theoretical operators and generalizations in quantum computing and how they differ or use quantum mechnaics principles and equations, which is essential to my theoretical formulation of the encoder and surrogate mode.
- Also, wrote notes on the latent spaces in the latex document as well.

## Notes

Link: https://www.overleaf.com/read/jmqnkwyxsdbz#ff567c
