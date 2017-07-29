import pyglet
from pyglet.gl import *
import helpers
import numpy as np
import pyopencl as cl
from pyopencl import array
import os

# needs to be global in order to be used from pyglet event calls
coords = []
velocs = []
masses = []

# opencl kernel source
# coords = coordinates of all particles
# velocs = velocities of all particles
# masses = masses of all particles
# n = number of particles
source = \
"""
__kernel void compute_force(__global float4* coords, __global float4* velocs,
__global const float* masses, const int n)
{
    int gid = get_global_id(0);

    // calculate force on particle due to all other particles
    const float G = 0.0001;
    const float4 s1 = (float4)(coords[gid].x, coords[gid].y, coords[gid].z, 0);
    const float m1 = masses[gid];
    float4 F = (float4)(0, 0, 0, 0);
    for (int i = 0; i < n; i++)
    {
        if (i == gid) continue;
        float4 s2 = (float4)(coords[i].x, coords[i].y, coords[i].z, 0);
        float4 m2 = masses[i];
        float4 disp = s2 - s1;
        F += G*m1*m2*fast_normalize(disp)/pow(fast_length(disp)+0.001f, 1);
    }

    // update coords and velocs of the current particle using Euler's method
    const float dt = 0.01;
    float4 a = F/masses[gid];
    velocs[gid] += a*dt;
    coords[gid] += velocs[gid]*dt;
}
"""

def main():
    # global variables
    global coords
    global velocs
    global masses 

    # initialise window
    window = pyglet.window.Window()

    # initialise opencl
    platform = helpers.choose_platform()
    devices = helpers.choose_device(platform)
    context = cl.Context(devices)
    program = cl.Program(context, source).build()
    queue = cl.CommandQueue(context)
    mf = cl.mem_flags

    # initialise particles
    n_bodies = 10000
    x = 1.5 # scaling factor for initial velocity
    for i in range(n_bodies):
        coords.append((np.random.random(4)-0.5)*3)
        velocs.append([coords[i][1]*x, -coords[i][0]*x, 0, 0])
        masses.append(np.random.random(1)[0]*50 + 5)

    # init event handlers
    @window.event
    def on_draw():
        global bodies

        # clear screen
        glClear(GL_COLOR_BUFFER_BIT)

        # set projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(90, window.width/window.height, 0.1, 100)

        # set modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # camera fixed to [2,2,2] looking at [0,0,0] for now
        gluLookAt(2, 2, 2, 0, 0, 0, 0, 0, 1)

        # draw particles
        glPointSize(2)
        glBegin(GL_POINTS)
        for i in range(n_bodies):
            # the brightness of a particle depends on its distance from the origin
            a = 0.5/(np.sqrt(np.sum(np.square(coords[i]))))
            glColor3f(a, a, a)
            glVertex3f(coords[i][0], coords[i][1], coords[i][2])
        glEnd()

    # buffer memory to GPU
    coords_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = np.array(coords).astype(np.float32))
    velocs_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = np.array(velocs).astype(np.float32))
    masses_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.array(masses).astype(np.float32))

    # main loop
    frame_index = 0
    while (True):
        pyglet.clock.tick()

        # compute new particle coordinates on GPU and copy them to coords for displaying
        coords = np.empty_like(coords).astype(np.float32)
        program.compute_force(queue, coords.shape, None, coords_buf, velocs_buf, masses_buf, np.int32(n_bodies))
        cl.enqueue_copy(queue, coords, coords_buf)

        # refresh pyglet window
        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event("on_draw")
            window.flip()

        # create folder for frames if it does not exist
        if (os.path.isdir("./frames") == False):
            print("Creating a folder for output frames... ")
            os.mkdir("frames")
            print("Done")

        # save current frame to disk
        pyglet.image.get_buffer_manager().get_color_buffer().save('./frames/frame_{0}.png'.format(frame_index))
        print('Saved frame to ./frames/frame_{0}.png'.format(frame_index))
        frame_index += 1

if (__name__ == "__main__"):
    main()
