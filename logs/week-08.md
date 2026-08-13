# Week 8

**Dates:** 07-26 to 08-04

## Goals

- Improve model and understand the bottleneck the classical model has at the moment
- Convert research log into writeup or poster or presentation.
- Present findings to Dr.Lin

## Approach and Implementation

- So the biggest bottleneck that I had was that for some reason, the model was collapsing and it was just predict 0 energy every time.
- After changes mentioned in the previous log and on the research log Latex page, it was able to get out of that bottleneck.
- However, even after repeated runs, the validation score was not great (roughly 30% accurate).
- So I am going to test whether or not changing the prediction (from energy to Hamitonian), and using a more SciML approach (PDE loss like PINN)
  will solve the issue of the model learning the actual physics behind the data.
- I also confirmed that the data itself is sound and is not the issue either.

## Results

- Fixed the major bottleneck of the model just predicting the mean energy (0 since I normalized that), and we got the model to learn some physics at least.
- Did some MLPOps to save the model and weuight in checkpoints and to save the best runs and stop the epoch's early if no improvement.
- Performed debugging testing to show that dataset is sound, no bad optimization or collapses of the model, it just needs to be improved through a different formulation.

## Notes

https://www.overleaf.com/read/jmqnkwyxsdbz#ff567c
