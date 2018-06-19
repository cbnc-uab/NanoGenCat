import numpy as np


class ClusterBase:
    ##print("Call to ClusterBase")
    def get_layer_distance(self, miller, layers=1, tol=1e-9, new=True):
        """Returns the distance between planes defined by the given miller
        index.
        """
        ##print("JoyDivision:self.atomic_basis",self.atomic_basis)
        ##print("JoyDivision:self.lattice_basis",self.lattice_basis)
        ###print("**************",miller,"*****************")
        """
	I MODIFIED THIS THRESHOLD BECAUSE IT DIDN'T WORK WITH THE TiO2 (514) SURFACE
        threshold = 0.20000001
        """
        threshold = 0.12000001
        if new:
            # Create lattice sample
            size = np.zeros(3, int)
            for i, m in enumerate(miller):
                ##i = 0,1,2
                size[i] = np.abs(m) + 4
                ##print("i",i,"np.abs(m)",np.abs(m),"size[i]",size[i])

            m = len(self.atomic_basis)
            p = np.zeros((size.prod() * m, 3))
            for h in range(size[0]):
                for k in range(size[1]):
                    for l in range(size[2]):
                        i = h * (size[1] * size[2]) + k * size[2] + l
                        p[m * i:m * (i + 1)] = np.dot([h, k, l] +
                                                      self.atomic_basis,
                                                      self.lattice_basis)
                        ##print(h,k,l)
                        ##print([h, k, l] + self.atomic_basis)
                        ##print(self.lattice_basis)
            #print("p \n",p)
            #exit()
            ##print("************************************")
            # Project lattice positions on the miller directest_cluster.pytion.
            
            n = self.miller_to_direction(miller)
            ##IN d, THE POINTS THAT HAVE THE SAME VALUE ARE ON THE SAME PLANE
            d = np.sum(n * p, axis=1)
            #print("n \n",n,"\n p \n",p,"\n n*p \n",n*p, "\n d \n",d)
            ##print("\n d1 \n",d,)
            ##ONLY THE POSITIVE VALUES ARE KEPT AND SORTED
            if np.all(d < tol):
                # All negative incl. zero
                d = np.sort(np.abs(d))
                reverse = True
            else:
                # Some or all positive
                d = np.sort(d[d > -tol])
                #
                #print("\n d1 \n",d)
                reverse = False
            ##d1-d0, d2-d1, etc.
            ##IF THE DIFFERENCE IS >0 CONCATENATE, WHICH MEANS THE ELEMENTS
            ##OF THE NEXT d ARE ONE PER LAYER
            #print("concatenate",(d[1:] - d[:-1]))
            d = d[np.concatenate((d[1:] - d[:-1] > threshold, [True]))]
            #d = d[np.concatenate((d[1:] - d[:-1] > tol, [True]))]
            #print("\n d2 \n",d)
            ##THE NEXT D ARE THE INTERLAYER DISTANCES
            d = d[1:] - d[:-1]
            #print("\n d3 \n",d)
            #for i in d:
            #    if i > threshold:
            #        d[np.concatenate]
            #exit()

            # Look for a pattern in the distances between layers. A pattern is
            # accepted if more than 50 % of the distances obeys it.
            pattern = None
            #print("LEN D",len(d))
            for i in range(len(d)):
                for n in range(1, (len(d) - i) // 2 + 1):
                    #print("I",i,"N",n)
                    #print(d[i:i + n])
                    #print(d[i + n:i + 2 * n])
                    #print(d[i:i + n] - d[i + n:i + 2 * n])
                    ##print(range(1, (len(d) - i) // 2 + 1))
                    #print(len(d) - i)
                    ##EVALUATE IF ?
                    #print(i,i + n, i + n,i + 2 * n)
                    ##print(d[i:i + n] - d[i + n:i + 2 * n])
                    ###print(np.abs(d[i:i + n] - d[i + n:i + 2 * n]))
                    ###print(np.all(np.abs(d[i:i + n] - d[i + n:i + 2 * n]) < tol))
                    if np.all(np.abs(d[i:i + n] - d[i + n:i + 2 * n]) < tol):
                        counts = 2
                        for j in range(i + 2 * n, len(d)-n, n):
                            #print(j,j + n)
                            if np.all(np.abs(d[j:j + n] - d[i:i + n]) < tol):
                                counts += 1
                        if counts * n * 1.0 / len(d) > 0.5:
                            pattern = d[i:i + n].copy()
                            ###if len(pattern) > 1:
                                ###pattern = np.array([pattern[0]+pattern[1]])
                            #print("pattern",pattern)
                            break
                if pattern is not None:
                    break
            ##print("HANS:miller",miller, pattern)
            if pattern is None:
                #aise RuntimeError('Could not find layer distance for the ' +
                                   #'(%i,%i,%i) surface.' % miller)
                raise RuntimeError('Could not find layer distance for the ',
                                   miller, 'surface')
            if reverse:
                pattern = pattern[::-1]

            if layers < 0:
                pattern = -1 * pattern[::-1]
                layers *= -1        
            ##print(miller,"layers",layers) 
            ##The thing here below in () rounds up to the next integer
            map = np.arange(layers - layers % 1 + 1, dtype=int) % len(pattern)
            #print(np.arange(layers - layers % 1 + 1),np.arange(layers - layers % 1 + 1) % len(pattern))
            ##print("MILLER",miller,"MAP:",map)
            ##print("TRAVIATA",miller,pattern[map][:-1].sum() + layers % 1 * pattern[map][-1])
            ##THESE Rs ARE THE DISTANCES BETWEEN PLANES OF A SURFACES(1 OF 2)
            ##print(pattern,pattern[map][:-1].sum())
            ##print(layers % 1* pattern[map][-1])
            ##print(layers % 1 * pattern[map][-1])
            ##print("miller",miller,"layers",layers,pattern[map])
            ##print("m",miller,"layers",layers,"Parov",pattern[map][:-1].sum() + layers % 1 * pattern[map][-1])
            return pattern[map][:-1].sum() + layers % 1 * pattern[map][-1]
        n = self.miller_to_direction(miller)
        d1 = d2 = 0.0

        d = np.abs(np.sum(n * self.lattice_basis, axis=1))
        mask = np.greater(d, 1e-10)
        if mask.sum() > 0:
            d1 = np.min(d[mask])

        if len(self.atomic_basis) > 1:
            atomic_basis = np.dot(self.atomic_basis, self.lattice_basis)
            d = np.sum(n * atomic_basis, axis=1)
            s = np.sign(d)
            d = np.abs(d)
            mask = np.greater(d, 1e-10)
            if mask.sum() > 0:
                d2 = np.min(d[mask])
                s2 = s[mask][np.argmin(d[mask])]

        if d2 > 1e-10:
            if s2 < 0 and d1 - d2 > 1e-10:
                d2 = d1 - d2
            elif s2 < 0 and d2 - d1 > 1e-10:
                d2 = 2 * d1 - d2
            elif s2 > 0 and d2 - d1 > 1e-10:
                d2 = d2 - d1

            if np.abs(d1 - d2) < 1e-10:
                ld = np.array([d1])
            elif np.abs(d1 - 2 * d2) < 1e-10:
                ld = np.array([d2])
            else:
                assert d1 > d2, 'Something is wrong with the layer distance.'
                ld = np.array([d2, d1 - d2])
        else:
            ld = np.array([d1])

        if len(ld) > 1:
            if layers < 0:
                ld = np.array([-ld[1], -ld[0]])
                layers *= -1

            map = np.arange(layers - (layers % 1), dtype=int) % len(ld)
            r = ld[map].sum() + (layers % 1) * ld[np.abs(map[-1] - 1)]
        else:
            r = ld[0] * layers
        ##THESE Rs ARE THE DISTANCES BETWEEN PLANES OF A SURFACES (2 OF 2)
        return r
        ##print("***********************")
    def miller_to_direction(self, miller, norm=True):
        """Returns the direction corresponding to a given Miller index."""
        d = np.dot(miller, self.resiproc_basis)
        if norm:
            d = d / np.linalg.norm(d)
        return d
