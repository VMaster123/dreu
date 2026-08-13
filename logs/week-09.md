# Week 9

**Dates:** 08-03 to 08-12

## Goals

- The new goals is to now upgrade the classical model to a SciML model and test the differene
- End goal is to test the advantages of the latent space created that encodes physical properties that can
  classical predict or help in predicting energies in certain metrics such as error correction and speed compared to QisKit
- Convert findings into paper

## Approach and Implementation

- My advisor mentioned how my dataset was very limited because of the range of quibits in the experiment(3-4), so increased it to (3-8).
- Also there was something wrong with my code or model creation where the predictions were not working at all.
- At the end of the day, I was able to increase the model complexity and the data loading (no mean pooling for example), and I was able to get it produce results finally

## Results

- Did the datasplit based on arcitecture, noise model, hamiltonian, and random.
- Trained to classical encoder + surrogate model and got that to some degree the model was able to partial able to create a latent space that learned some physics
- However, I think that is the extent a classical model can go, and now estimates towards a physics-informed encoder and/surrogate is necessary.

## Notes

https://www.overleaf.com/read/jmqnkwyxsdbz#ff567c
