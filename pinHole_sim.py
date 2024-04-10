import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def P2R(radii, angles):
    # polar to rect.
    return radii * np.exp(1j*angles)

def R2P(x):
    # rect. to polar
    return np.abs(x), np.angle(x)

wvl = 1e-9              # wavelenght
A = 1                   # amplitude
gN = 256                # global num samples
use_gN = False     

# screen section
screen_xy = [2, 0]    # _xy: location
screen_l = 1           # _l: length
screen_N = 1024 if not use_gN else gN # _N: num samples on _l
screen_p = [[screen_xy[0], screen_l*(i/screen_N-1/2)] for i in range(screen_N)] # _p: sample points

# object section
object_l = 0.125
object_N = 1024 if not use_gN else gN
object_p = [[0, (object_l)*(i/object_N-1/2)-object_l] if i < object_N//4 else [0, (2*object_l)*(i/object_N)] for i in range(object_N)]
# point source
object_p = [[0, 0]]

# depth slices
depth_N = 1024 if not use_gN else gN

# slit section
slit_xy = [screen_xy[0]*0.8, 0]
slit_scale = 4
slit_w = slit_scale*wvl #_w: width
# under sampling the slit here causes aliasing
# sampling interval should be < wvl/2
slit_N = 1024 if not use_gN else gN
# slit_N = 32
slit_p = [[slit_xy[0], slit_xy[1]+slit_w*(i/slit_N-1/2)] for i in range(slit_N)]
# double slit
# slit_p = [[slit_xy[0], slit_xy[1]+slit_w*(i/slit_N-1/2) - 1*wvl] if i < slit_N//2 else [slit_xy[0], slit_xy[1]+slit_w*(i/slit_N-1/2) + 1*wvl] for i in range(slit_N)]
slit_z = np.zeros(slit_N).astype(complex)

slit_phases = np.zeros(slit_N)
slit_amp = np.zeros(slit_N)
slit_gain = (1/slit_N)    # arbitrary

# compute slit
for slit_idx, sp in enumerate(slit_p):
    z = 0
    for o in object_p:
        d = math.dist(sp, o)
        phase = 2*np.pi*d/wvl
        z += P2R(slit_gain/(d**2), phase)
    slit_z[slit_idx] = z

# compute screen
s_amp = np.zeros((screen_N, depth_N))
s_z = np.zeros((screen_N, depth_N)).astype(complex)
screen_render_times = []
print('rendering screens:')
for d_idx in tqdm(range(depth_N)):
    screen_p = [[(d_idx+1)*screen_xy[0]/depth_N, screen_l*(i/screen_N-1/2)] for i in range(screen_N)]
    screen_render_start = time.process_time()
    for s_idx, s in enumerate(screen_p):
        postSlit = True if s[0] > slit_xy[0] else False
        if postSlit:
            object_p = slit_p
        for o_idx, o in enumerate(object_p):
            d = math.dist(s, o)
            phase = 2*np.pi*d/wvl
            if postSlit:
                amp_prev, phase_prev = R2P(slit_z[o_idx])
            else:
                amp_prev = 1
                phase_prev = 0
            z = P2R(amp_prev/(d**2), phase_prev+phase)
            s_z[s_idx, d_idx] += z

    s_amp[:, d_idx], _ = R2P(s_z[:, d_idx])
    screen_render_time = time.process_time() - screen_render_start
    screen_render_times.append(screen_render_time)

# render time
print(f'average render time (per screen): {np.mean(screen_render_times):0.3f}s')
print(f'total render time: {np.sum(screen_render_times):0.3f}s')

# plot field
fig, ax = plt.subplots()
suptitle_str = '\'i\' source simulation'
fig.suptitle(suptitle_str)
ax.imshow(20*np.log10(s_amp), 
            cmap='hot', interpolation='none')

title_str = f'λ={wvl:.2e}m, slit width={slit_scale}λ, slit samples={slit_N}'
ax.set_title(title_str)
ax.set_xlabel("distance (m)")
ax.set_ylabel("source co-ordinates (m)")
x_tick_inc = depth_N//8 if not (depth_N//8 == 0) else depth_N
y_tic_inc = screen_N//8 if not (screen_N//8 == 0) else screen_N
ax.set_xticks(range(depth_N)[::x_tick_inc], [f'{screen_xy[0]*i/depth_N}' for i in range(depth_N)][::x_tick_inc])
ax.set_yticks(range(screen_N)[::y_tic_inc], [f'{-screen_l*(i/screen_N-1/2)}' for i in range(screen_N)][::y_tic_inc])
ax2 = ax.twinx()
ax2.set_ylabel('screen')
ax2.yaxis.set_tick_params('both', length=0, labelsize=0)
fig.tight_layout()

plt.show()
plt.close()

fig, ax = plt.subplots()
fig.suptitle(title_str)
ax.plot(20*np.log10(s_amp[:,-1]))
ax.set_title(title_str)
ax.set_xlabel("source co-ordinates (m)")
ax.set_ylabel("intensity (log scale)")
ax.set_xticks(range(screen_N)[::y_tic_inc], [f'{screen_l*(i/screen_N-1/2)}' for i in range(screen_N)][::y_tic_inc])

plt.show()
plt.close()

pass
