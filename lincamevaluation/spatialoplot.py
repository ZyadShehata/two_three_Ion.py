import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

sn.set()

def spatialoFunc(x, x0, sat, dwf, loss=1):
    return (((sat**2)*np.cos((x-x0)/2)**2*dwf*loss)/((sat+np.cos(x0)*dwf)*(sat+np.cos(x)*dwf)))


xes = np.linspace(-7*np.pi, 7*np.pi, num=600)

plt.figure(figsize=(15,5))
plt.title(r'theoretical $g^{(2)}\left(\vec{x}_0, \vec{x}_1, \tau=0 \right)$')
plt.xlabel('position (phase)')
plt.ylabel('$g^{(2)}$-signal')
plt.xlim((-7*np.pi, 7*np.pi))


colors = [
    sn.xkcd_rgb["dusty purple"],
    sn.xkcd_rgb["amber"],
    sn.xkcd_rgb["pale red"],
    sn.xkcd_rgb["medium green"],
    sn.xkcd_rgb["denim blue"],
]

pies = np.linspace(0, 1.5*np.pi, num=4)
pies =  np.append(pies, 0.21 * np.pi)

i = 0
for item in pies:
    label = r'$' + str(np.round(item/np.pi, 2)) + r'\cdot \pi$'
    plt.plot(xes, spatialoFunc(xes, item, 1.8, 0.6, 1.), label=label, color=colors[i])
    i += 1

plt.legend(loc="best")
plt.savefig("theo.pdf")