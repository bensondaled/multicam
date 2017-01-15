# multicam

I branched this off from other projects since it has been useful to people in various contexts. It uses the CLEye Driver and SDK to acquire 2 PSEye camera feeds via a python interface. It uses multiprocessing to spawn processes for reliable acquisition and saving. This may be unneeded in simple cases, but is useful when the multicam capability is combined with more complex interfaces.

It consists of just one file, cameras.py. A usage example is given in the main of this file.
