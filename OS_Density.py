import numpy as np
from scipy import ndimage as ndi
import pandas as pd
from skimage.measure import EllipseModel
import math
from scipy.spatial.distance import pdist, squareform
from statistics import stdev
from numpy import linalg
from tqdm import tqdm    
        
def isotropic_dilation(image, radius, out=None, spacing=None):

    dist = ndi.distance_transform_edt(np.logical_not(image), sampling=spacing)
    return np.less_equal(dist, radius, out=out)

def get_n_nuclei_neighbors(r,matrice) :
    
    return np.sum(matrice <= r,axis =1) - 1

def get_dist_to_neigbours(n,matrice):
    
    dist_nuclei = []
    new_n = int(min(n,len(matrice)))
    for row in matrice : 
        dist_nuclei.append(sorted(row,reverse=False)[new_n+1])
        
    return dist_nuclei

def get_dist_to_neigbours_centroids(r,matrice,coord_nuclei):
    
    dist_centroids_nuclei = []
    list_list_of_pos = []
    for pos in range(len(matrice)) :
        list_of_pos = [i  for i, valeur in enumerate(matrice[pos]) if valeur <= r and i != pos]
        list_of_pos_completed = list_of_pos + [pos]
        
        if len(list_of_pos) == 0 :
            distance = 0
        else : 
            
            list_of_coord = coord_nuclei[list_of_pos]
            
            new_coord = np.mean(list_of_coord, 0)
            
            distance = pdist([coord_nuclei[pos],new_coord])[0]
        
        dist_centroids_nuclei.append(distance)
        list_list_of_pos.append(list_of_pos_completed)

    return np.reshape(dist_centroids_nuclei,(len(dist_centroids_nuclei),)), list_list_of_pos

def get_neighbors_average_distance_to_nuclei(matrice,list_list_of_pos):

    dist_nuclei = []

    for pos, list_of_pos in enumerate(list_list_of_pos) :

        list_of_pos.remove(pos)

        list_of_distance = matrice[pos][list_of_pos]
        mean_distance = np.mean(list_of_distance)

        dist_nuclei.append(mean_distance)

    return np.reshape(dist_nuclei,(len(dist_nuclei),))

def is_aligne(points):
    if len(points) > 2 :
        x1,y1 = points[0]
        x2,y2 = points[1]
        
        for i in range(2,len(points)):
            
            x,y = points[i]
            
            if (y2-y1)*(x-x1) != (y - y1)*(x2 - x1):
                return False
            
        return True 
    
    else :
        return False    
    
def midpoint(p1, p2):

    mid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]


    return mid

def ellipsoide_fit(x, y, z):
    
    if not [np.std(x),np.std(y),np.std(z)] == [0,0,0] :
    
        # D = np.array([x * x + y * y - 2 * z * z,
        #             x * x + z * z - 2 * y * y,
        #             2 * x * y,
        #             2 * x * z,
        #             2 * y * z,
        #             2 * x,
        #             2 * y,
        #             2 * z,
        #             1 - 0 * x])
        # d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
        # if not np.linalg.det(D.dot(D.T)) == 0 :
            
        #     u = np.linalg.solve(D.dot(D.T), D.dot(d2))
        #     a = np.array([u[0] + 1 * u[1] - 1])
        #     b = np.array([u[0] - 2 * u[1] - 1])
        #     c = np.array([u[1] - 2 * u[0] - 1])
        #     v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
        #     A = np.array([[v[0], v[3], v[4], v[6]],
        #                 [v[3], v[1], v[5], v[7]],
        #                 [v[4], v[5], v[2], v[8]],
        #                 [v[6], v[7], v[8], v[9]]])

        #     center = np.linalg.solve(- A[:3, :3], v[6:9])

        #     translation_matrix = np.eye(4)
        #     translation_matrix[3, :3] = center.T

        #     R = translation_matrix.dot(A).dot(translation_matrix.T)

        #     evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
        #     evecs = evecs.T

        #     radii = np.sqrt(1. / np.abs(evals))
        #     #radii *= np.sign(evals)
            
        #     semi = radii
        """ Find the minimum volume ellipsoid which holds all the points
        
        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!
        
        Here, P is a numpy array of N dimensional points like this:
        P = [[x,y,z,...], <-- one point per line
             [x,y,z,...],
             [x,y,z,...]]
        
        Returns:
        (center, radii, rotation)
        
        """

        P = np.array([x,y,z]).T

        (N, d) = np.shape(P)
        d = float(d)
    
        # Q will be our working array
        Q = np.vstack([np.copy(P.T), np.ones(N)]) 
        QT = Q.T
        
        # initializations
        u = (1.0 / N) * np.ones(N)
        # Khachiyan Algorithm

        V = np.dot(Q, np.dot(np.diag(u), QT))
        if np.linalg.det(V) != 0 :
            M = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u

        # center of the ellipse 
            center = np.dot(P.T, u)
        
            # the A matrix for the ellipse
            A = linalg.inv(
                        np.dot(P.T, np.dot(np.diag(u), P)) - 
                        np.array([[a * b for b in center] for a in center])
                        ) / d
                            
            # Get the values we'd like to return
            U, s, rotation = linalg.svd(A)
            radii = 1.0/np.sqrt(s)
            
        elif np.array([x,y,z]).shape[1] == 3:

            # center = np.mean(P, axis=0)
            # P -= center
            # A = np.vstack(P).T
            # C = np.dot(A, A.T)
            # eigen_values, eigen_vectors = np.linalg.eig(C)
            # semi_axes_lengths = np.sqrt(eigen_values)
            # semi_axes_lengths = np.sort(semi_axes_lengths)[::-1]
            # major_axis_length = semi_axes_lengths[0]
            # intermediate_axis_length = semi_axes_lengths[1]
            # minor_axis_length = semi_axes_lengths[2]
            # radii = semi_axes_lengths

            # Calculate the center of the ellipse using affine plane fitting
            centroid, _, _ = np.linalg.svd(P)
            center = centroid[0]

            # Center the points around the estimated center
            centered_points = points - center

            # Calculate the covariance matrix of the centered points
            cov_matrix = np.cov(centered_points, rowvar=False)

            # Calculate the eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # Sort the eigenvalues and eigenvectors in descending order
            indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[indices]
            eigenvectors = eigenvectors[:, indices]

            # The principal axes are the eigenvectors associated with the eigenvalues
            major_axis = eigenvectors[:, 0]
            medium_axis = eigenvectors[:, 1]
            minor_axis = eigenvectors[:, 2]

            radii =  [major_axis,medium_axis,minor_axis]


        elif np.array([x,y,z]).shape[1] == 2:
    

            #dist = np.linalg.norm(P[0] - P[1])
            #center =  np.mean(P, axis=0)

            #radii =  [dist/2,0.0,0.0]

            
            dist = np.nan
            center =  [np.nan,np.nan,np.nan]

            radii =  [np.nan,np.nan,np.nan]
                
        
        return center, radii

        # elif is_aligne([x,y]) or is_aligne([x,z]) or is_aligne([y,z]):  
        #     if  is_aligne([x,y]) :      
        #         return False   
        # 
 
    elif 0 in [np.std(x) ,np.std(y),np.std(z)] : 
        ell = EllipseModel()
        stdevs = []
        points = [] 
        for axe in [x,y,z] :
            if not np.std(axe) == 0 :
                stdevs.append(stdev(axe)) 
                points.append(axe)
            else :
                other_point = axe
        points = np.array(points)
        # stdevs = [stdev(p) for p in [x, y, z]]
        # dims_sorted = np.argsort(stdevs)
        # all_points = [x, y, z]
        # points = np.array([all_points[dim] for dim in dims_sorted[1:]]).T
        #other_point = all_points[dims_sorted[0]]
        is_success = ell.estimate(points)
        if is_success:
            c1, c2, a, b, theta = ell.params
            center = [c1,c2,np.mean(other_point)]
            semi =  [a,b,0.0]
            radii = theta

        else :
            if min(points[:,0]) == max(points[:,0]) :
                points = np.fliplr(points)

            pos_min = np.where(points[:,0] == min(points[:,0]))
            pos_max = np.where(points[:,0] == max(points[:,0]))


            a = math.sqrt(((points[pos_max,0]-points[pos_min,0])**2)+((points[pos_max,1]-points[pos_min,1])**2))
            c1,c2 =  midpoint(points[pos_min][0],points[pos_max][0])
            
            coord_1 = np.reshape(np.array([points[pos_min,0],points[pos_max,0]]),(2,))
            coord_2 = np.reshape(np.array([points[pos_min,1],points[pos_max,1]]),(2,))
            
            slope, intercept = np.polyfit(coord_1,coord_2,1)

            degree = math.atan(slope)
            
            center = [c1,c2,np.mean(other_point)]
            semi =  [a/2,0.0,0.0]
            
        
    else :
        
        #center = [np.mean(x),np.mean(y),np.mean(z)]
        #semi =  [0.0,0.0,0.0]       
        
        center = [np.nan,np.nan,np.nan]
        semi =  [np.nan,np.nan,np.nan]       
    return center, semi

def test_create_list_circle(a,b,r,stepSize = 0.1):

    #Generated vertices
    list_of_coord = []
    t = 0
    while t < 2 * math.pi:
        list_of_coord.append([r * math.cos(t) + a, r*math.sin(t) + b, 0])
        t += stepSize
        
    return list_of_coord

def test_create_list_ellipse(a,b,r1,r2,phi,stepSize = 0.1):
    
    #Generated vertices
    list_of_coord = []
    t = 0
    phi = math.radians(phi)
    while t <= 2 * (math.pi) + stepSize:
        list_of_coord.append([r1*math.cos(t)*math.cos(phi) - r2*math.sin(t)*math.sin(phi) + a,
                              r1*math.cos(t)*math.cos(phi) + r2*math.sin(t)*math.cos(phi) + b,
                              0])
        t += stepSize
        
    return list_of_coord

def test_create_list_line(a,b,r,stepSize = 0.1):
    
    #Generated vertices
    list_of_coord = []
    t = 0
    while t < r:
        list_of_coord.append([t, a*t+b, 0])
        t += stepSize
        
    return list_of_coord

def test_other_ellpsoid(x,y,z):
    
    # change xx from vector of length N to Nx1 matrix so we can use hstack
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    z = z[:,np.newaxis]

    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
    J = np.hstack((x*x,y*y,z*z,x*y,x*z,y*z, x, y, z))
    K = np.ones_like(x) #column of ones

    #np.hstack performs a loop over all samples and creates
    #a row in J for each x,y,z sample:
    # J[ix,0] = x[ix]*x[ix]
    # J[ix,1] = y[ix]*y[ix]
    # etc.

    JT=J.transpose()
    JTJ = np.dot(JT,J)
    InvJTJ=np.linalg.inv(JTJ);
    ABC= np.dot(InvJTJ, np.dot(JT,K))

    # Rearrange, move the 1 to the other side
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
    #    or
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
    #  where J = -1
    eansa=np.append(ABC,-1)

    return (eansa)

def ellipsoide_axis(r,matrice,coord_nuclei):
    
    semi_list = []
    for pos in range(len(matrice)) :
        list_of_pos = [i  for i, valeur in enumerate(matrice[pos]) if valeur <= r]

        if len(list_of_pos) < 2 :
            semi_major = 0
            semi_median = 0
            semi_minor = 0

            semi_list.append([semi_major, semi_median, semi_minor])

        else :

        
            list_of_coord = coord_nuclei[list_of_pos]
            
            #list_of_coord = test_create_list_circle(a = 0,b = 0, r = 5 ,stepSize = 0.1) # z = 0
            #list_of_coord = test_create_list_line(a = 1,b = 0, r = 5 ,stepSize = 0.1) # z = 0
            #list_of_coord = test_create_list_ellipse(a = 0,b = 0,r1 = 1,r2 = 2,phi = 45,stepSize = 0.1) # z = 0
            
            list_of_coord = np.array(list_of_coord)
            x = list_of_coord[:, 0]
            y = list_of_coord[:, 1]
            z = list_of_coord[:, 2]   

            #eansa = test_other_ellpsoid(x,y,z)
            try :
                center, semi  = ellipsoide_fit(x,y,z)
            except :
                semi = [np.nan,np.nan,np.nan]
                print('pos : ', pos)
            
            semi_major, semi_median, semi_minor = sorted(semi, reverse=True)
            semi_list.append([semi_major, semi_median, semi_minor])
        
        
    return np.array(semi_list) 
    
def ratio_minor_med(semi_median, semi_minor):

    if np.nan in [semi_median, semi_minor]:
        sol = np.nan
    
    elif semi_median == 0:
        sol = 0
    else : 
        sol = semi_minor/semi_median
    
    return sol

def ratio_med_major(semi_median, semi_major):
    if np.nan in [semi_median, semi_major]:
        sol = np.nan
    
    elif semi_major == 0:
        sol = 0
    else : 
        sol = semi_median/semi_major

    return sol

def ratio_minor_major(semi_minor,semi_major):
    if np.nan in [semi_minor,semi_major]:
        sol = np.nan
    
    elif semi_major == 0:
        sol = 0
    else : 
        sol = semi_minor/semi_major

    return sol

def ratio_minor_x_medi_major_squar(semi_major, semi_median, semi_minor):
    if np.nan in [semi_major, semi_median, semi_minor]:
        sol = np.nan
    
    elif semi_major == 0:
        sol = 0
    else : 
        sol = (semi_minor*semi_median)/(semi_major**2)

    return sol
        
def ratio_minor_squar_medi_x_major(semi_major, semi_median, semi_minor):
    
    if np.nan in [semi_major, semi_median, semi_minor]:
        sol = np.nan
    elif semi_median == 0:
        sol = 0
    else : 
        sol = (semi_minor**2)/(semi_major*semi_median)
    return sol

def get_all_ratio(semi_list):
    all_ratio = []
    for semi in semi_list :
        semi_major, semi_median, semi_minor = semi
        minor_med = ratio_minor_med(semi_median, semi_minor)
        med_major = ratio_med_major(semi_median, semi_major)
        minor_major = ratio_minor_major(semi_minor,semi_major)
        minor_x_medi_major_squar = ratio_minor_x_medi_major_squar(semi_major, semi_median, semi_minor)
        minor_squar_medi_x_major = ratio_minor_squar_medi_x_major(semi_major, semi_median, semi_minor)
        all_ratio.append([minor_med,med_major,minor_major,minor_x_medi_major_squar,minor_squar_medi_x_major])
    return all_ratio

def get_ripley_k(r, n_neighbors, coord_nuclei, coord_nuclei_pixel, mask_organoid,list_of_pos_completed,nuclei_volume_pixel):
    
    
    list_density_ripley = []
    list_density = []
    list_crystal_distance = []
    list_k = []
    list_crystal_distance_ripley = []
    list_ratio_volume_nuclei_neighbors_organoid = []
    my_organoid = np.zeros(mask_organoid.shape)
    rapport_coord_pixel =  coord_nuclei[0]/coord_nuclei_pixel[0]
    
    small_ellipsoid = np.zeros(2 *np.floor( r * (1/rapport_coord_pixel[::-1])).astype(int))
    small_shape = np.array(small_ellipsoid.shape)
    
    all_radius = tuple(r // 2 if r%2 <= 252230 else r // 2 + 1 for r in small_shape)
    r_depth, r_height, r_width = all_radius
    small_ellipsoid[all_radius] = 1
    small_ellipsoid = isotropic_dilation(small_ellipsoid, r, spacing = rapport_coord_pixel[::-1])*1
    
    mask_organoid_pad = np.pad(mask_organoid, ((r_depth, r_depth), (r_height, r_height), (r_width, r_width)))
    my_organoid_pad = np.pad(my_organoid, ((r_depth, r_depth), (r_height, r_height), (r_width, r_width)))
    
    nb_pixel_new_sphere = (small_ellipsoid>0).sum() # number of pixel of ellipsoid radius r 
    
    volume_small_ellipsoid = (4/3)*math.pi*(r**3)
    volume_small_ellipsoid_pixel = nb_pixel_new_sphere*np.prod(rapport_coord_pixel)
    for i in tqdm(range(len(coord_nuclei))):
        
        nuc_centroid = tuple(np.round(coord_nuclei[i][::-1]*(1/rapport_coord_pixel[::-1])).astype(int))
        z0 = nuc_centroid[0]
        z1 = nuc_centroid[0] + 2 * r_depth
        y0 = nuc_centroid[1]
        y1 = nuc_centroid[1] + 2 * r_height
        x0 = nuc_centroid[2]
        x1 = nuc_centroid[2] + 2 * r_width


        mask_organoid_pad_crop=mask_organoid_pad[z0:z1, y0:y1, x0:x1]

        real_sphere = small_ellipsoid * mask_organoid_pad_crop
        nb_pixel_real_sphere = (real_sphere>0).sum() # number of pixel of the real spheroïde arround the nucleus


        nuclei_volume_pixel = np.array(nuclei_volume_pixel)
        sum_volume_neighbor = np.sum(nuclei_volume_pixel[list_of_pos_completed[i]])
        ratio_volume_nuclei_neighbors_organoid = sum_volume_neighbor/nb_pixel_new_sphere

        rapport_volume = nb_pixel_new_sphere / nb_pixel_real_sphere # rapport volume entre l'ellipsoïde imaginaire de rayon r et l'organoïde de rayon r contenu dedans 
        normalized_n_neighbors = n_neighbors[i]*rapport_volume # Nombre de noyaux qu'il y aurait proportionnellement à la taille de l'objet complet
        
        if normalized_n_neighbors == 0:

            density_normalized_ripley = 0
            crystal_distance = 0
            density_normalized = 0 
            crystal_distance_ripley = 0

        else: 

            volume_real_sphere = nb_pixel_real_sphere*np.prod(rapport_coord_pixel)

            crystal_distance = (volume_real_sphere/n_neighbors[i])**(1/3)
            
            

            density_normalized = n_neighbors[i]/volume_real_sphere
        
        density_normalised_mm3 = density_normalized*10**9
        list_density.append(density_normalised_mm3)

        list_crystal_distance.append(crystal_distance)
        list_k.append(normalized_n_neighbors)
        list_ratio_volume_nuclei_neighbors_organoid.append(ratio_volume_nuclei_neighbors_organoid)



    return list_density,list_crystal_distance, list_k, list_ratio_volume_nuclei_neighbors_organoid

def um_maper(c_name:str):
    return c_name.replace('_um','_μm')

def comput_density_stats(im_df, mask_organoid, r_list=[20,40],n_list={1,5}):


    nuclei_centroid = im_df[['nuclei_centroid_X_um','nuclei_centroid_Y_um','nuclei_centroid_Z_um']].values
    nuclei_stat_density = im_df.copy()
    nuclei_centroid_pixel =im_df[['nuclei_centroid_X', 'nuclei_centroid_Y', 'nuclei_centroid_Z']].values
    nuclei_volume_pixel = im_df[['nuclei_volume']].values

    pdist_centroid = pdist(nuclei_centroid)

    matrice_distance = squareform(pdist_centroid)
    for j,r in enumerate(r_list) : 

        n_nuclei_neighbors = get_n_nuclei_neighbors(r,matrice_distance)
        
        dist_to_neigbours_centroids,list_of_pos_completed = get_dist_to_neigbours_centroids(r,matrice_distance,nuclei_centroid)
        semi_list = ellipsoide_axis(r,matrice_distance,nuclei_centroid)
        all_ratio = get_all_ratio(semi_list)
        list_mean_distance = get_neighbors_average_distance_to_nuclei(matrice_distance,list_of_pos_completed)
        list_density,list_crystal_distance,ripley_k, ratio_volume_nuclei_neighbors_organoid = get_ripley_k(r,n_nuclei_neighbors,nuclei_centroid,nuclei_centroid_pixel,mask_organoid,list_of_pos_completed,nuclei_volume_pixel)
        nuclei_stat_density['nuclei_distance_to_neighbors_centroids_' + str(r)+'_um'] = dist_to_neigbours_centroids
        nuclei_stat_density['nuclei_neighbors_average_distance' + str(r)+'_um'] = list_mean_distance
        nuclei_stat_density[['nuclei_major_axis_in_' + str(r)+'_um','nuclei_medium_axis_in_' + str(r)+'_um','nuclei_minor_axis_in_' + str(r)+'_um']] = semi_list
        nuclei_stat_density[['nuclei_ratio_minor_by_medium_in_' + str(r)+'_um','nuclei_ratio_medium_by_major_in_' + str(r)+'_um','nuclei_ratio_minor_by_major_in_' + str(r)+'_um',
                            'nuclei_prolate_ratio_' + str(r)+'_um','nuclei_oblate_ratio_'+ str(r)+'_um']] = all_ratio
        nuclei_stat_density['nuclei_n_neighbors_' + str(r)+'_um'] = n_nuclei_neighbors
        nuclei_stat_density[['nuclei_nb_neighbors_ripley_' + str(r)+'_um','nuclei_nb_nuclei_by_mm_3_' + str(r)+'_um','nuclei_crystal_distance_' + str(r)+'_um',]] = np.array([ripley_k,list_density,list_crystal_distance]).T
    
    for k, n in enumerate(n_list) : 
        print('Traitement with neighbour = ' + str(n)+ 'th')
        dist_nuclei_neigbours_n = get_dist_to_neigbours(n,matrice_distance)
        nuclei_stat_density['nuclei_dist_to_'+str(n)+'th_neigbour'] = dist_nuclei_neigbours_n


    return nuclei_stat_density    
           

