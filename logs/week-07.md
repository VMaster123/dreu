# Week 7

**Dates:** 07-20 to M07-27

## Goals

- Finish the Surrogate Model and finish validation testing for the 3 baseline encoders
- Get a better as to what specific models to implement for the surrogate model.\
- Begin converting writeup to more "research paper" and poster
- Get feedback from Dr.Lin and Pestourie

## Approach and Implementation

- Given how the Python libraries worked and I do not want to mess with them anymore, as they took way too long to figure out, I decided to try my best to add and more libraries unless absolutely necessary.
- Still on code experimentation, but I believe I have a dataset so I can first test if the encoder itself
  has any point, before we get to transferibility between tasks.
- 4 baselines will be raw data to surrogate, classical , hybrid (SciML), and then quantum encoder.
- I have still not yet finalized what the surrogate models will be ideally, or if I should go an RL route (if time permits). My notes on stochastic control theory and RL will be useful in that case.

## Results

- Got a meangingful simulated dataset given circuit parameters and input paramaters and assumption I made into the theoretics that is roughly 30k lines.
- I have also done some preprocessing, test train split, and worked on the hybrid encoder as well. Did some validation testing on the noise implementation in the model and the actual overall dataset, but not the encoder fully.
- Update new diagram of my code.
- Reformatted my notes on the current writeup.

## Notes

- Sorry for the delay. Please note that the code is in its elementary stages and have not been fully set in stone. Full disclosre I used AI for the programming, and will comb over the code extensively by hand alongside using verification and validation methods in the next upcoming weeks. https://www.overleaf.com/read/jmqnkwyxsdbz#ff567c
