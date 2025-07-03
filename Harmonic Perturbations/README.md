Visualise perturbations on a sphere in terms of spherical harmonics. Values of the harmonics are used to change the radius of the perturbing sphere. 

Inputs are $l$ and $m$ describing the spherical harmonic $Y_{lm}$ and Save:

Saves the animation by default. Takes a bit of time as it creates a smooth animation. 
Pass `save=False` to not save the gif, and instead to view an interactive animation.

**Animation function:** Uses FUNCANIMATION:
This cannot use blitting, so the animation is slow. But the interactivity is
good. I made the number of segments smaller to use this animation. Start time is FAST.