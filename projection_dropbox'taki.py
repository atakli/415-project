# import progressbar
# pbar = progressbar.ProgressBar()
import PySimpleGUI as sg
import pdb
layout = [
	[sg.Text('Choose where you get the projection data from:')],
    [sg.Radio('From text file     ', "RADIO2"), sg.Radio('From mat file     ', "RADIO2"),
	sg.Radio('Do new projection          ', "RADIO2", default=True)], 
	[sg.Text('Enter the number of beams:')],      
	[sg.InputText()],
	[sg.Text('Enter the step size:')],      
	[sg.InputText()],            
	[sg.Text('kare_kosede_50ye50.mat is the default')],
	[sg.Listbox(values=['lena_256ya256.mat', 'Shepp-Logan.mat'],default_values=['kare_kosede_50ye50.mat'], size=(30, 3))],
	[sg.Text('Choose filter type:')],
	[sg.Radio('Ramp     ', "RADIO3", default=True), sg.Radio('Hanning     ', "RADIO3"),
	sg.Radio('Hamming     ', "RADIO3"), sg.Radio('No filter     ', "RADIO3")],
	[sg.Checkbox('Do only projection')],
	[sg.Text('Enter size of the image if you will use ready projection data:')],
	[sg.InputText()],
	[sg.Submit(), sg.Cancel()]]
window  = sg.Window('Projection GUI', auto_size_text=True, default_element_size=(40, 1)).Layout(layout)
import sys
import time
while True:
	event, values = window.Read()
	if event == 'Submit':
		break
	elif event == 'Cancel':
		sys.exit()
# temp = values[3]
# values.pop(3)
# values.append(temp)
bas = time.time()
if event == 'Submit':
	window.Close()
elif event == 'Cancel':
	sys.exit()
import scipy.io as sio
import numpy as np
pi = np.pi
if values[2] == True:
	if values[5] == []:
		mat = sio.loadmat('kare_kosede_50ye50.mat')						# 1. step: load the image
	else: 
		mat = sio.loadmat(values[5][0])
	img = list(mat.values())[3:][0]
	size = img.shape[0]													# 2. step: determine the size of the image
	number_of_sampling_points = number_of_beams = int(values[3])		# 3. step: get step_size and number of beams
	step_size = float(values[4])
	teta = np.arange(0,180,step_size)								
	teta_degree = teta*pi/180
	number_of_projections = teta_adedi = teta.shape[0]
else:
	if values[0] == True:
		pass	# from txt
	elif values[1] == True:
		mat = sio.loadmat('projection_data.mat')	# from mat
		img = list(mat.values())[3:][0]
		number_of_projections = img.shape[0]
		number_of_sampling_points = number_of_beams = img.shape[1]
		step_size = 180/number_of_sampling_points
		teta = np.arange(0,180,step_size)
		teta_degree = teta*pi/180
		size = int(values[11])
# pdb.set_trace()

if values[6] == True:
	filter = 6
	filter_name = 'Ramp Filter'
else:
	if values[7] == True:
		filter = 7
		filter_name = 'Hanning Filter'
	elif values[8] == True:
		filter = 8
		filter_name = 'Hamming Filter'
	elif values[9] == True:
		filter = 0
		filter_name = 'No Filter'
import matplotlib.pyplot as plt

def project():
	y_values = x_values = np.arange(-size/2, size/2+1)				# determine x & y values on the image

	t = np.linspace(-size/pow(2,1/2), size/pow(2,1/2),number_of_beams)

	x_adedi = x_values.shape[0]
	top_uz = size * np.sqrt(2)
	karsi_uz = []
	for i in teta:
		if i <= 90:
			karsi_uz.append(size*np.sqrt(2)*np.cos((45-i)*pi/180))
		elif 90 < i <= 135:
			karsi_uz.append(size*np.sqrt(2)*np.cos((135-i)*pi/180))
		else:
			karsi_uz.append(size*np.sqrt(2)*np.cos((i-135)*pi/180))

	# 5. step: Find all intersection points for all beams for all projection angles using line equation:
	result=[]
	for aci in teta_degree:						# 8.7 saniye
		for t_degeri in t:
			for x_degeri in x_values:
				resulted_y_values = np.tan(aci) * x_degeri + t_degeri / np.cos(aci) #line equation
				yeni=[aci,t_degeri,x_degeri,resulted_y_values]
				result.append(yeni)
	for aci in teta_degree:						# 9.25 saniye
		for t_degeri in t:
			for y_degeri in y_values:
				if aci==0 and y_degeri==t_degeri:					# in case of 0 in the denominator
					for x_degeri in x_values:
						result.append([aci,t_degeri,x_degeri,y_degeri])
				elif aci != 0:
					resulted_x_values = (y_degeri * np.cos(aci) - t_degeri)/np.sin(aci) # line equation
					yeni=[aci,t_degeri,resulted_x_values,y_degeri]
					result.append(yeni)
	# 6. Step: Remove the points which are irrelevant to the object:
	# final_result=[]
	# for res in pbar(result):
		# if res not in final_result:
			# final_result.append(res)		# bu yöntemin varlığından dolayı Cenab-ı Hakk'a hamd ü senalar olsun.
	
	final_result=[list(t) for t in set(tuple(element) for element in result)]		# 13.84 saniye
	son = []
	# Bu işlemle irrelevant noktaları attığımız için otomatikman mesela 0 derece t=sqrt(-2) noktaları gitti
	for element in final_result:			# 6.5 saniye
		if (float(element[2]) <= float(x_values[-1]) and float(element[2]) >= float(x_values[0]) and float(element[3]) <= float(y_values[-1]) and float(element[3]) >= float(y_values[0])):
			son.append(element)
	son=sorted(son)							# 7. Step: Sort the relevant points	 (2.2 saniye)
# Below, I grouped the elements of 'son' variable with respect to their angle and t values while it had one row only before this işlem
	temp_aci_t_degeri = son[0][0:2]
	alt_liste=[son[0]]
	son_son=[]		
	for i in son[1:]:			
		if i[0:2] == temp_aci_t_degeri:
			alt_liste.append(i)
			temp_aci_t_degeri = i[0:2]
		else:
			son_son.append(alt_liste)
			alt_liste = []
			alt_liste.append(i)
			temp_aci_t_degeri = i[0:2]
	son_son.append(alt_liste)
	# 8. Find the midpoint and the length of line segments:
	midX=[]
	midY=[]
	distance_son_son=[]
	for i in son_son:					# 3.32 saniye
		temp=i[0]
		distance=[]
		for j in i[1:]:
			temp_midX=((j[2]+temp[2])/2)
			temp_midY=((j[3]+temp[3])/2)
			dist_temp = pow((j[2]-temp[2])*(j[2]-temp[2])+(j[3]-temp[3])*(j[3]-temp[3]),1/2)
			midX.append(temp_midX)
			midY.append(temp_midY)
			distance.append(dist_temp)
			temp = j
		distance_son_son.append(distance)
	# 9. Detect the address (row and column data) by using the midpoint data.
	rowdata=[]
	columndata=[]
	# midX ve midY'nin içindeki 0.00001 gibi sayıları round et:		# 8.17 saniye
	# midX_yeni = []
	# midY_yeni = []
	# for i in midX:
		# if abs(i - round(i)) < 0.0001:
			# midX_yeni.append(round(i))
		# else:
			# midX_yeni.append(i)
	# for i in midY:
		# if abs(i - round(i)) < 0.0001:
			# midY_yeni.append(round(i))
		# else:
			# midY_yeni.append(i)
	
	# midX = midX_yeni
	# midY = midY_yeni	
	for midYpoints in midY:												#bu iki for 14.26 saniye
		rowdata.append(np.ceil(size/2 - np.floor(midYpoints))-1)
	for midXpoints in midX:
		columndata.append(np.ceil(size/2 + np.ceil(midXpoints))-1)
	# 10. Sum all pixel value and distance products
	say = 0
	projection = []
	for i in distance_son_son:		# 2.24 saniye
		toplam=0
		for j in i:
			toplam=toplam+(j*img[int(rowdata[say])][int(columndata[say])])	
			say=say+1
		projection.append(toplam)
	grup=[]
	sa=0
	for te in teta:
		if ( int(te) == 45 or int(te) == 135):
			grup.append(number_of_beams)
		else:
			k=0
			for i in range(len(t)):
				if abs(t[i]) > karsi_uz[sa]/2:
					k+=1
				else:
					break
			grup.append(number_of_beams-k*2)
		sa+=1
	# açılara göre gruplu projection:
	son_projection=[]
	say_sırala = 0
	for grup_elemanı in grup:
		ara_projection=[]
		for i in range(grup_elemanı):
			ara_projection.append(projection[i+ say_sırala])
		say_sırala = i+ say_sırala + 1
		son_projection.append(ara_projection)
	# açılara göre gruplu distance:
	say_sırala = 0
	son_distance=[]
	for grup_elemanı in grup:
		ara_distance=[]
		for i in range(grup_elemanı):
			ara_distance.append(distance_son_son[i+ say_sırala])
		say_sırala = i+ say_sırala + 1
		son_distance.append(ara_distance)
	# make the projection with 0s which occur when the teta values other than 45 and 90 degrees
	import copy
	son_projection_with_zeros = copy.deepcopy(son_projection)
	son_distance_with_zeros = copy.deepcopy(son_distance)
	grup_say=0		
	for pro in son_projection_with_zeros:				#4.26 saniye
		if (len(pro) < number_of_beams):
			for i in range(int((number_of_beams - grup[grup_say])/2)):
				pro.insert(0,0)
				pro.insert(len(pro),0)
		grup_say+=1
	grup_say=0
	for pro in son_distance_with_zeros:
		if (len(pro) < number_of_beams):
			for i in range(int((number_of_beams - grup[grup_say])/2)):
				pro.insert(0,0)
				pro.insert(len(pro),0)
		grup_say+=1
	flatttened_projection = [y for x in son_projection_with_zeros for y in x]
	
	with open('projection_data.txt','w') as dosya_txt:
		dosya_txt.write(str(number_of_projections)+'\n'+str(number_of_sampling_points)+'\n')
		for k in range(len(son_projection_with_zeros)):
			dosya_txt.write(str(k+1)+'\n')
			for j in son_projection_with_zeros[k]:
				dosya_txt.write(str(j)+'\n')
	mat_array=np.array(son_projection_with_zeros)
	# pdb.set_trace()
	sio.savemat('projection_data.mat', mdict={'projection': mat_array})
	if values[10] == True:
		plot_projection(flatttened_projection)
	return son_projection_with_zeros,son_distance_with_zeros,rowdata,columndata
def plot_projection(projection):
	plt.plot(projection)
	plt.xlabel('The beam which going through the image')
	plt.ylabel('Projection value for the beams')
	plt.show()
	
from numpy.fft import fft2,ifft2
from mpl_toolkits.axes_grid1 import make_axes_locatable
image_to_be_reconstructed,distance,rowdata,columndata = project()

def ramp_filter():
	fft_of_projection = fft2(image_to_be_reconstructed)
	
	if number_of_sampling_points % 2 == 0:
		temp = number_of_sampling_points/2
		first_half_of_filter = np.linspace(0,1/(temp-0.5)*(temp-1),temp)
		high_pass_filter = np.array(list(first_half_of_filter) + list(first_half_of_filter[::-1]))
	else:
		temp = np.floor(number_of_sampling_points/2) + 1
		first_half_of_filter = np.linspace(0,1,temp)
		high_pass_filter = np.array(list(first_half_of_filter) + list(first_half_of_filter[::-1][1:]))
		
	filtered_fft_of_projection = fft_of_projection * high_pass_filter
	
	ifft_of_projection = ifft2(filtered_fft_of_projection)
	# ifft_of_projection'ı array'den listeye çevir:
	liste_ifft_of_projection = []
	for k in ifft_of_projection:
		liste_ifft_of_projection.append([i for i in k])
	return liste_ifft_of_projection
def hanning_filter():
	fft_of_projection = fft2(image_to_be_reconstructed)
	
	high_pass_filter = np.hanning(number_of_sampling_points)
		
	filtered_fft_of_projection = fft_of_projection * high_pass_filter
	
	ifft_of_projection = ifft2(filtered_fft_of_projection)
	# ifft_of_projection'ı array'den listeye çevir:
	liste_ifft_of_projection = []
	for k in ifft_of_projection:
		liste_ifft_of_projection.append([i for i in k])
	return liste_ifft_of_projection
def hamming_filter():
	fft_of_projection = fft2(image_to_be_reconstructed)
	
	high_pass_filter = np.hamming(number_of_sampling_points)
		
	filtered_fft_of_projection = fft_of_projection * high_pass_filter
	
	ifft_of_projection = ifft2(filtered_fft_of_projection)
	# ifft_of_projection'ı array'den listeye çevir:
	liste_ifft_of_projection = []
	for k in ifft_of_projection:
		liste_ifft_of_projection.append([i for i in k])
	return liste_ifft_of_projection

def back_projection(getir=None):

	if getir == None:
		getir = image_to_be_reconstructed
	# Multiply the filtered projection data with the distance:
	netice = []
	for i in getir:
		o=[]
		for k in i:
			o.append(k*np.array(distance[getir.index(i)][i.index(k)]))
		netice.append(o)
	kl=np.array([1.6024768-0.52718694j, 1.6024768-0.52718694j])
	tur = type(kl)
	son_netice=[]
	for i in netice:
		ara_netice=[]
		for k in i:
			if type(k) == tur:
				daha_ara_netice=[]
				for j in k:
					daha_ara_netice.append(j)
				ara_netice.append(daha_ara_netice)
			else:
				ara_netice.append(k)
		son_netice.append(ara_netice)
	
	img_back = np.zeros((size,size))
	say = 0
	for i in son_netice:	# en fazla bu döngü süre alıyo, ama bu da 2 saniye (100 beam 5 derece'de)
		for j in i:
			if not j == 0:
				for k in j:
					img_back[int(rowdata[say])][int(columndata[say])] += k.real
					say += 1
	# pdb.set_trace()
	max_img=np.amax(img_back)	
	img_normalized=img_back/max_img
	
	error_img = img - img_normalized
	max_img_er=np.amax(error_img)	
	img_normalized_er=error_img/max_img_er
	
	# plt.figure(2)
	fig,(original,back,error) = plt.subplots(1,3)
	plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9 , top=0.9 , wspace=0.4, hspace=0.2)
	original.imshow(img,cmap='gray')
	im_err = error.imshow(img_normalized_er,cmap='gray')
	im_back = back.imshow(img_normalized,cmap='gray')
	divider_b = make_axes_locatable(back)
	divider_e = make_axes_locatable(error)
	cax1 = divider_b.append_axes("right", size="5%", pad=0.05)
	cax2 = divider_e.append_axes("right", size="5%", pad=0.05)
	original.set_title('Original image')
	back.set_title('Back projected image')
	error.set_title('Error')
	fig.colorbar(im_back,cax=cax1)
	fig.colorbar(im_err,cax=cax2)
	# plt.savefig()
	plt.suptitle('number_of_sampling_points: '+str(number_of_sampling_points)+'\n'+' step_size: '+str(step_size)+'\n'+filter_name)
	plt.show()
if values[10] == True:
	project()
else:
	if filter == 6:
		back_projection(ramp_filter())
	elif filter == 7:
		back_projection(hanning_filter())
	elif filter == 8:
		back_projection(hamming_filter())
	elif filter == 0:
		back_projection()
	
