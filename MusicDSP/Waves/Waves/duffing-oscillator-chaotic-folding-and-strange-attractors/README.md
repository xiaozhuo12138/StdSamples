# duffing oscillator chaotic folding and strange attractors
 Investigation into the non-linear system of the Duffing oscillator that generates chaos through stretching and folding

At the beginning of the 20th century Georg Duffing investigated the physics of periodically driven nonlinear oscillators with damping.
What we now call the Duffing equation is a generalisation of the linear differential equation that describes a damped and forced harmonic motion.
 
The duffing equation has the form,
x¨ + 2αx˙ + βx + γx^3 = Fe*cos(ωt)

where x is the position of the oscillator in the x-direction, each dot refers to the time derivative of x (i.e. one dot is velocity and two dots are acceleration).
The coefficiants refer to: α, non-consevrative damping. β, linear conservative driving force. γ, non-linear driving force. Fe, periodic driving force with a frequency defined by ω.

Different values of these coefficients lead to different behaviour of the oscillator, which in some cases is chaotic.
Here, chaotic refers to chaos theory in which dynamical systems with appearignyl random states of disorder are in fact deterministic in nature, with underlying structure and high
sensitivity to initial conditions.

This investigation was a part of my final assessment for my MPhys, master's level Non-Linear Physics module at the Univeristy of Surrey and alongside the content learnt there
each program was adapted for the specifics of this investigation, and underlying theory gained from the excellent textbooks:
"Dynamical Systems with Applications using Python" by Stephen Lynch (2018) and "Nonlinear Dynamics and Chaos" by Steven H. Strogatz (2016)

Program 1: phasespace.py

By setting Fe = 0 we can consider the system in the absence of the external force and using the following input parameters:

    β = -1
    γ = +1
    ω = +1

    x¨ + 2αx˙ - x + x^3 = 0

To analyze this second order equation we can transform it into two first order equations:


    x˙ = v 
    v˙ = x - x^3 - 2αv


stable points are found when x˙ = v˙ = 0.


    x˙ = v = 0
    v˙ = x - x^3 - 2αv
    0 = x - x^3  
      = x(1 - x^2)

    x = 0 or sqrt(1) 


Therefore there are three stationary points (x,v) = (0,0), (1,0) & (-1,0).

Nullclines are found when either x˙ = 0 or y˙ = 0 


    x˙ = v˙


and thus there is a nullcline in x along the line v = 0


    v˙ = x - x^3 - 2αv


and so the nullcline in y is dependant on the constant alpha.

The regimes with α = 0, α < 0 and α > 0 were considered with the assistance of phase space diagrams in the (x˙,v˙) plane plotted using the program titled "phasespace".
Examples of these phase space diagrams can be found in the corresponding folder.

The program produces both forwards (i.e. where the system goes) and backwards (i.e. where the system came from) trajectories in blue and red respectively for for two sets of initial conditions (-4,4) and (4,-4).
The arrow grid is also plotted in cyan to show the grid of dv/dx values, illustrating the nullclines as the regions where the arrows point in only the x or y direction.

For the system with α = 0 one can see how the trajectories orbit the fixed points with no change to the size of this oscillation over time. However with α = 0.2 one can see how as the damping term removes energy from the system the size of the oscillation reduces and the trajectories spiral in towards, and eventually settling at, the fixed points.
Increasing the value of α, increases the speed at which the trajectories fall into the stable points.
While each plot has both backwards and forwards trajectories simulating the cases for the positive and negative values of α, plotting with a value of α = -0.2 not only shows this relation between backwards and forward trajectories but also shows how the slight change in the direction of the vector field arrows simulates the push of the damping term to either remove energy from the system when α>0 or to add it when α<0

It should be noted though, especially in the context of following investigations that the system remains predictable, with the value of α relating to only the size and speeds of the orbits as well as if the system moves towards or away from the stable points, as this damping term either removes or adds energy to the system.


Program 2: 3Dauto.py

The Duffing equation with the same constants as above has been converted into a 3D autonomous system in the variables x, v ≡ x˙ and θ ≡ ωt, and phase trajectories in the v-x plain along with the solution x(t) have been plotted for various values of α and Fe
The trajectories were plotted from an initial position, (v_0,x_0), of (1,0).

Examples of the plots this produces can be found in the corresponding folder.

Setting Fe = 0 recovers the solutions from Program 1.


Program 3: duffing oscillator poincare sections.py

Poincare maps are produced from the intersection of the periodic orbit of the system with some lower-dimensional subspace, called the poincare section.
In this case the map was taking from the poincare section obtained by the intersection of the attractor with the plane θ = ωt = 0.
Fe was kept constant at a value of 0.41 which corresponds to the Duffing oscillators chaotic scheme and the poincare sections were plotted at diffferent phases, 0 ≤ θ = ωt ≤ 2π (i.e. one cycle of the oscillator). Examples of these plots can be seen in the corresponding folder.


The example lots show the progression of the Poincare section in the (x,v) plane at different phases 0 to 2pi.
The first key observation is that the plots for a phase shift of 0 and of 2pi have the same macroscopic structure as the shape, known as a strange attractor, has completed one full phase cycle and returned to its initial position.
The strange attractor has fractal structure as while one rotation returns it to its original shape, each individual point does not return to its original starting position, only nearby.

The two outer "limbs" of the structure are continuously stretched out, as neighbouring points move away from each other, before they fold in on themselves with the points returning close to their original positions.
This stretching is responsible for the systems sensitivity to its inital conditions while the folding leads to the energy dissipation found in the system due to the damping terms.
As such, this Poincare section provides a qualitative understanding to the nature of the chaotic system.

While I hope the example plots are sufficient to show this stretching and folding, this YouTube video by Stephen Morris provides an excellent animation of what is occuring: https://www.youtube.com/watch?v=y-Xj0c6fRbc 
