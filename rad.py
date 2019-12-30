import numpy as np
import matplotlib.pyplot as plt
# from sys import argv
import scipy.io as sio
from skimage.io import imread
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
# image = shepp_logan_phantom()

def plot(image = 'Shepp-Logan.mat',cizdirilecek_aci = 0):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
	# image = rescale(image, scale=0.4, mode='reflect', multichannel=False)
	image = list(sio.loadmat(image).values())[3]
	ax1.set_title("Original")
	ax1.imshow(image, cmap=plt.cm.Greys_r)

	# theta = np.linspace(0., 180., max(image.shape), endpoint=False)
	# sinogram = radon(image, theta=theta, circle=True)
	projection = radon(image, np.array([cizdirilecek_aci]))#, circle=True)
	# ax2.set_title("Radon transform\n(Sinogram)")
	ax2.set_title('Radon transform at the angle '+str(cizdirilecek_aci)+'$^\circ$')
	# ax2.set_xlabel("Projection angle (deg)")
	# ax2.set_ylabel("Projection position (pixels)")
	ax2.set_xlabel('The beam which going through the image (t coordinates)')
	ax2.set_ylabel('Projection value for the beams')
	ax2.plot(projection)#, cmap=plt.cm.Greys_r, aspect='auto')

	fig.tight_layout()
	plt.show()
	
if __name__ == '__main__':
	plot()