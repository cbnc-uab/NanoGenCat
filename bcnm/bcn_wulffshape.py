"""
Name       : WulffShape.py
Author     : Javier Heras-Domingo
Description: Get Ideal Wulff Shape using Python-ASE library. 
	     Based on Pymatgen WulffShape functionality.
"""
import warnings
import math
import numpy as np
import scipy as sp
import ase
from ase.spacegroup import crystal, Spacegroup
from scipy.spatial import ConvexHull

#Avoid deprecated matplotlib warnings.
warnings.filterwarnings("ignore")


def hkl_to_str(hkl):
	"""
	Prepare for display on plots for Miller Index.
	"""
	str_format = '($'
	for x in hkl:
		if x < 0:
			str_format += '\\overline{' + str(-x) + '}'
		else:
			str_format += str(x)
	str_format += '$)'
	return str_format


def get_tri_area(pts):
	"""
	Compute the area of 3 points.
	"""
	a, b, c = pts[0], pts[1], pts[2]
	v1 = np.array(b) - np.array(a)
	v2 = np.array(c) - np.array(a)
	area_tri = abs(sp.linalg.norm(sp.cross(v1, v2)) / 2)
	return area_tri


def get_angle(v1, v2, units="degrees"):
	"""
	Compute the angle between two vectors.
	"""
	d = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
	d = min(d, 1)
	d = max(d, -1)
	angle = math.acos(d)
	if units == "degrees":
		return math.degrees(angle)
	elif units == "radians":
		return angle
	else:
		raise ValueError("Invalid units {}".format(units))


class W_Facet:
	"""
	Wulff Facet container for each computed Wulff plane.
	"""

	def __init__(self, normal, e_surf, normal_pt, dual_pt, index, m_ind_orig, miller):

		self.normal = normal
		self.e_surf = e_surf
		self.normal_pt = normal_pt
		self.dual_pt = dual_pt
		self.index = index
		self.m_ind_orig = m_ind_orig
		self.miller = miller
		self.points = []
		self.outer_lines = []


class IdealWulff:
	"""
	Generate Wulff shape from list of miller index and surface energies,
	with given conventional unit cell.
	"""

	def __init__(self, structure, spacegroup, miller_list, e_surf_list):
		"""
		Structure   : Python-ASE object of the conventional unit cell.
		Spacegroup  : Symmetry Spacegroup from ASE.
		miller_list : List of miller Index list.
		e_surf_list : List of surface energies of the corresponding miller index.
		"""


		if any([se < 0 for se in e_surf_list]):
			print('WARNING: Unphysical (negative) surface energy detected.')


		self.color_ind = list(range(len(miller_list)))
		self.input_miller_fig = [ hkl_to_str(x) for x in miller_list]
		
		#1.Store input data
		self.structure = structure
		self.lattice = self.structure.get_cell()
		self.miller_list = tuple([tuple(x) for x in miller_list])
		self.hkl_list = tuple([(x[0], x[1], x[-1]) for x in miller_list])
		self.e_surf_list = tuple(e_surf_list)
		self.spacegroup = spacegroup
		self.sg = Spacegroup(spacegroup)

		#2.Get surface normal from get_miller_eq
		self.facets = self.get_miller_eq()

		#3.Consider the dual condition
		dual_pts = [x.dual_pt for x in self.facets]
		dual_convex = ConvexHull(dual_pts)
		dual_cv_simp = dual_convex.simplices

		#4.Get Cross-point from the simplices of the dual ConvexHull
		wulff_pt_list = [self.get_cross_pt_dual_simp(dual_simp) for dual_simp in dual_cv_simp]
		wulff_convex = ConvexHull(wulff_pt_list)
		wulff_cv_simp = wulff_convex.simplices
		
		#5.Store simplices and convex
		self.dual_cv_simp = dual_cv_simp
		self.wulff_pt_list = wulff_pt_list
		self.wulff_cv_simp = wulff_cv_simp
		self.wulff_convex = wulff_convex

		#6.Declare on_wulff and color_area
		self.on_wulff, self.color_area = self.get_simpx_plane()

		#7.Miller Areas
		miller_area = []
		for m, in_mill_fig in enumerate(self.input_miller_fig):
			miller_area.append(in_mill_fig + ' : ' + str(round(self.color_area[m], 10)))
		self.miller_area = miller_area


	def get_miller_eq(self):
		"""
		Get all the facets functions for Wulff shape calculations.
		"""

		all_hkl = []
		color_ind = self.color_ind
		planes = []
		recp = self.structure.get_reciprocal_cell()
		sg = Spacegroup(self.spacegroup)
		recp_ops = sg.get_op()

		for i, (hkl, energy) in enumerate(zip(self.hkl_list, self.e_surf_list)):
			hkl_eq = [all_hkl.append((x[0], x[1], x[2], energy, i)) for x in sg.equivalent_reflections(hkl)]
	
		for miller in all_hkl:
			hkl = miller[0:3]
			normal = np.dot(hkl, recp)
			normal /= sp.linalg.norm(normal)
			normal_pt = [x * miller[3] for x in normal]
			dual_pt = [x / miller[3] for x in normal]
			color_plane = color_ind[divmod(miller[-1], len(color_ind))[1]]
			planes.append(W_Facet(normal, miller[3], normal_pt, dual_pt, color_plane, miller[-1], hkl))

		planes.sort(key=lambda x: x.e_surf)
		return planes

	
	def get_cross_pt_dual_simp(self, dual_simp):
		"""
		Get cross points with dual symmetry operations.
		"""
		
		matrix_surfs = [self.facets[dual_simp[i]].normal for i in range(3)]
		matrix_e = [self.facets[dual_simp[i]].e_surf for i in range(3)]
		cross_pt = sp.dot(sp.linalg.inv(matrix_surfs), matrix_e)
		return cross_pt

	def get_simpx_plane(self):
		"""
		Locate the plane on wulff_cv, comparing the center of the triangle
		with the plane functions.
		"""
		
		on_wulff = [False] * len(self.miller_list)
		surface_area = [0.0] * len(self.miller_list)
		for simpx in self.wulff_cv_simp:
			pts = [self.wulff_pt_list[simpx[i]] for i in range(3)]
			center = np.sum(pts, 0) / 3.0
			for plane in self.facets:
				abs_diff = abs(np.dot(plane.normal, center) - plane.e_surf)
				if abs_diff < 1e-8:
					on_wulff[plane.index] = True
					surface_area[plane.index] += get_tri_area(pts)
					
					plane.points.append(pts)
					plane.outer_lines.append([simpx[0], simpx[1]])
					plane.outer_lines.append([simpx[1], simpx[2]])
					plane.outer_lines.append([simpx[0], simpx[2]])
					break
		for plane in self.facets:
			plane.outer_lines.sort()
			plane.outer_lines = [line for line in plane.outer_lines if plane.outer_lines.count(line) != 2]
		
		return on_wulff, surface_area

	def get_colors(self, color_set, alpha, off_color, custom_colors={}):
		"""
		Assign colors according to the surface energies of on_wulff facets.
		"""
		
		import matplotlib as mpl
		import matplotlib.pyplot as plt
		color_list = [off_color] * len(self.hkl_list)
		color_proxy_on_wulff = []
		miller_on_wulff =[]
		e_surf_on_wulff = [(i, e_surf) for i, e_surf in enumerate(self.e_surf_list) if self.on_wulff[i]]

		c_map = plt.get_cmap(color_set)
		e_surf_on_wulff.sort(key=lambda x: x[1], reverse=False)
		e_surf_on_wulff_list = [x[1] for x in e_surf_on_wulff]
		if len(e_surf_on_wulff) > 1:
			cnorm = mpl.colors.Normalize(vmin=min(e_surf_on_wulff_list), vmax=max(e_surf_on_wulff_list))
		else:
			cnorm = mpl.colors.Normalize(vmin=min(e_surf_on_wulff_list) - 0.1, vmax=max(e_surf_on_wulff_list) + 0.1)

		scalar_map = mpl.cm.ScalarMappable(norm=cnorm, cmap=c_map)

		for i, e_surf in e_surf_on_wulff:
			color_list[i] = scalar_map.to_rgba(e_surf, alpha=alpha)
			if tuple(self.miller_list[i]) in custom_colors.keys():
				color_list[i] = custom_colors[tuple(self.miller_list[i])]
			color_proxy_on_wulff.append(
				plt.Rectangle((2,2), 1, 1, fc=color_list[i], alpha=alpha))
		scalar_map.set_array([x[1] for x in e_surf_on_wulff])
		color_proxy = [plt.Rectangle((2,2), 1, 1, fc=x, alpha=alpha) for x in color_list]

		return color_list, color_proxy, color_proxy_on_wulff, miller_on_wulff, e_surf_on_wulff_list

	def show(self, *args, **kwargs):
		"""
		Show the Wulff plot.
		"""
		
		self.get_plot(*args, **kwargs).show()

	def savefig(self, name, dpi=300, transparent=False, *args, **kwargs):
		"""
		Save Wulff shape plot as image file.
		"""
		if transparent == True:
			self.get_plot(*args, **kwargs).savefig(name, dpi=300, transparent=True)

		else:
			self.get_plot(*args, **kwargs).savefig(name, dpi=300, transparent=False)

	def get_line_in_facet(self, facet):
		"""
		Returns the sorted points in a facet used to draw a line
		"""

		lines = list(facet.outer_lines)
		pt = []
		prev = None
		
		while len(lines) > 0:
			if prev is None:
				l = lines.pop(0)
			else:
				for i, l in enumerate(lines):
					if prev in l:
						l = lines.pop(i)
						if l[1] == prev:
							l.reverse()
						break

			pt.append(self.wulff_pt_list[l[0]].tolist())
			pt.append(self.wulff_pt_list[l[1]].tolist())
			prev = l[1]

		return pt

	def get_plot(self, color_set='GnBu', grid=True, axis=True, 
			show_area=False, alpha=1, off_color='red', direction=None, 
			bar_pos=(0.75, 0.15, 0.05, 0.65), bar_on=False, units_in_JPERM2=True, legend=True, aspect_ratio=(4, 4), custom_colors={}):
		
		"""
		Get the Wulff shape plot.
		"""

		import matplotlib as mpl
		import matplotlib.pyplot as plt
		import mpl_toolkits.mplot3d as mpl3
		
		color_list, color_proxy, color_proxy_on_wulff, miller_on_wulff, e_surf_on_wulff = self.get_colors(color_set, alpha, off_color, custom_colors=custom_colors)

		if not direction:
			#direction = max(self.area_fraction_dict.items(), key=lambda x: x[1])[0]
			direction = (0.6, -2, 0)

		fig = plt.figure()
		fig.set_size_inches(aspect_ratio[0], aspect_ratio[1])
		azim, elev = self.get_azimuth_elev([direction[0], direction[1], direction[-1]])

		wulff_pt_list = self.wulff_pt_list

		ax = mpl3.Axes3D(fig, azim=azim, elev=elev)

		for plane in self.facets:
			if len(plane.points) < 1:
				continue
			
			plane_color = color_list[plane.index]
			pt = self.get_line_in_facet(plane)
			
			tri = mpl3.art3d.Poly3DCollection([pt])
			tri.set_color(plane_color)
			tri.set_edgecolor("#000000")
			ax.add_collection3d(tri)

		r_range = max([np.linalg.norm(x) for x in wulff_pt_list])
		ax.set_xlim([-r_range * 1.1, r_range * 1.1])
		ax.set_ylim([-r_range * 1.1, r_range * 1.1])
		ax.set_zlim([-r_range * 1.1, r_range * 1.1])

		if legend:
			color_proxy = color_proxy
			if show_area:
				ax.legend(color_proxy, self.miller_area, loc='upper left', bbox_to_anchor=(0, 1), fancybox=True, shadow=False)
			else:
				ax.legend(color_proxy_on_wulff, self.miller_list, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3, fancybox=True, shadow=False)
		
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')

		if bar_on:
			cmap = plt.get_cmap(color_set)
			cmap.set_over('0.25')
			cmap.set_under('0.75')
			bounds = [ round(e, 2) for e in e_surf_on_wulff]
			bounds.append(1.2 * bounds[-1])
			norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
			
			ax1 = fig.add_axes(bar_pos)
			cbar = mpl.colorbar.ColorbarBase(
				ax1, cmap=cmap, norm=norm, boundaries=[0] + bounds + [10], extend='both', ticks=bounds[:-1], spacing='proportional', orientation='vertical')
			units = "$J/m^2$" if units_in_JPERM2 else "$eV/\AA^2$"
			cbar.set_label('Surface Energies (%s)' %(units), fontsize=100)

		if grid:
			ax.grid('off')
		if axis:
			ax.axis('off')

		return plt

	def get_azimuth_elev(self, miller_index):
		"""
		Wulff plot representation direction and elevation.
		"""
		
		if miller_index == (0, 0, 1) or miller_index == (0, 0, 0, 1):

			return 0, 90
		else:
			cart = np.dot(miller_index, self.lattice)
			azim = get_angle([cart[0], cart[1], 0], (1, 0, 0))
			v = [cart[0], cart[1], 0]
			elev = get_angle(cart, v)
			return azim, elev

	@property
	def volume(self):
		"""
		Volume of the Wulff Shape.
		"""
		return self.wulff_convex.volume

	@property
	def miller_area_dict(self):
		"""
		Returns miller area dictionary.
		"""
		return dict(zip(self.miller_list, self.color_area))

	@property
	def miller_energy_dict(self):
		"""
		Returns miller energy dictionary.
		"""
		return dict(zip(self.miller_list, self.e_surf_list))

	@property
	def surface_area(self):
		"""
		Total surface area of Wulff shape.
		"""
		return sum(self.miller_area_dict.values())

	@property
	def weighted_surface_energy(self):
		"""
		Returns Weighted surface energy.
		"""
		return self.total_surface_energy / self.surface_area

	@property
	def area_fraction_dict(self):
		"""
		Returns area fraction dict.
		"""
		return {hkl: self.miller_area_dict[hkl] / self.surface_area for hkl in self.miller_area_dict.keys()}

	@property
	def anisotropy(self):
		"""
		Returns Wulff shape anisotropy.
		"""
	
		square_diff_energy = 0
		weighted_energy = self.weighted_surface_energy
		area_frac_dict = self.area_fraction_dict
		miller_energy_dict = self.miller_energy_dict

		for hkl in miller_energy_dict.keys():
			square_diff_energy += np.power((miller_energy_dict[hkl] - weighted_energy), 2) * area_frac_dict[hkl]

		return np.sqrt(square_diff_energy) / weighted_energy

	@property
	def shape_factor(self):
		"""
		Returns shape factor. Useful for determining the critical nucleus size.
		"""
		return self.surface_area / (self.volume ** (2 / 3))

	@property
	def effective_radius(self):
		"""
		Returns radius of the Wulff shape when it is approximated as a sphere.
		"""
		return ((3/4)*(self.volume/np.pi)) ** (1 / 3)

	@property
	def total_surface_energy(self):
		"""
		Total surface energy of the Wulff shape.
		"""
		
		tot_surface_energy = 0
		for hkl in self.miller_energy_dict.keys():
			tot_surface_energy += self.miller_energy_dict[hkl] * self.miller_area_dict[hkl]

		return tot_surface_energy


	@property
	def tot_corner_sites(self):
		"""
		Returns the number of vertices in the convex hull.
		"""
		return len(self.wulff_convex.vertices)


	@property
	def tot_edges(self):
		"""
		Returns the number of edges in the convex hull.
		"""
		
		all_edges = []
		for facet in self.facets:
			edges = []
			pt = self.get_line_in_facet(facet)
			
			lines = []
			for i, p in enumerate(pt):
				if i == len(pt) / 2:
					break
				lines.append(tuple(sorted(tuple([tuple(pt[i*2]), tuple(pt[i*2+1])]))))

			for i, p in enumerate(lines):
				if p not in all_edges:
					edges.append(p)

			all_edges.extend(edges)

		return len(all_edges)

	@property
	def surface_area_percentage(self):
		"""
		Returns surface contribution percentage on the Wulff shape.
		"""
		
		percentage_dict = {}
		w_surf_area = self.surface_area
		surf_areas = self.miller_area_dict.values()
		surf_index = list(self.miller_area_dict.keys())
		
		for m, n in enumerate(surf_areas):
			percentage = (n / w_surf_area) * 100
			percentage_dict.update({str(surf_index[m]): round(percentage, 6)})

		return percentage_dict
	
	@property
	def wulff_shape_info(self):
		"""
		Returns all the information available of the computed Wulff shape.
		"""
		print(' ')
		print('====WULFF SHAPE INFO====')
		print('[+] Shape Factor         : %.6f ' % self.shape_factor)
		print('[+] Anisotropy           : %.6f ' % self.anisotropy)
		print('[+] Weighted Surf. Energy: %.6f ' %  self.weighted_surface_energy)
		print('[+] Surface Area         : %.6f ' % self.surface_area)
		print('[+] Volume               : %.6f ' % self.volume)
		print('[+] Effective Radius     : %.6f ' % self.effective_radius)
		print('[+] Total Surf. Energy   : %.6f ' % self.total_surface_energy)
		print('[+] N. Corners           : ', self.tot_corner_sites)
		print('[+] N. Edges             : ', self.tot_edges)
		print('====Surface Area====')
		for a, b in zip(self.miller_area_dict.keys(), self.miller_area_dict.values()):
			print('     [-]'+str(a)+' : '+str(round(b, 6)))
		print('====Area Fraction====')
		for c, d in zip(self.area_fraction_dict.keys(), self.area_fraction_dict.values()):
			print('     [-]'+str(c)+' : '+str(round(d,6)))
		print('====Surf. Area Percentage====')
		for e, f in zip(self.surface_area_percentage.keys(), self.surface_area_percentage.values()):
			print('     [-]'+str(e)+' : '+str(round(f,6)))
		print('====END====')
		print(' ')
		return


