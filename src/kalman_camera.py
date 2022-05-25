# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

### Expect parameters of this kind




class KalmanFilter(object):
    """
    A Kalman filter for camera parameters.
    (3D for constants position, 4D*3 for position,velocity, acceleration model)
    The 17-D state space:
        3x:
            "pan_degrees": pan * 180. / np.pi,
            "tilt_degrees": tilt * 180. / np.pi,
            "roll_degrees": roll * 180. / np.pi,
            "x_focal_length": self.xfocal_length,
        3x:
            "position_meters": self.position.tolist(),
                    # "y_focal_length": self.yfocal_length,
                    # "principal_point": [self.principal_point[0], self.principal_point[1]],
                    # "radial_distortion": self.radial_distortion.tolist(),
                    # "tangential_distortion": self.tangential_disto.tolist(),
                    # "thin_prism_distortion": self.thin_prism_disto.tolist()

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant acceleration model, which is bounced about by noise.

    The predictions for these are taken as a direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        dt = 1.
        nTriples = 4
        nX = 3
        ndim = nX + 3*nTriples#17

        ### MOTION MODEL

        # Create Kalman filter model matrices.
        self._motion_matrix_A = np.eye(ndim, ndim)
        for i in range(2*nTriples):
            self._motion_matrix_A[i, nTriples + i] = dt

        self._process_noise_QdiagonalValues = \
            np.square(np.array([0, 0, 0, 0, 0, 0, 0, 0, .6, .6, .1, 10.0, 0, 0, 0]))
        #TODO: orders of magnitude off, likely


        ### MEASUREMENT PROCESS

        self._measurement_matrix_C = np.eye(nTriples + nX, ndim)#Observation matrix (I believe)
        self._measurement_matrix_C[nTriples:, :] *= 0
        for i in range(1, nX+1):
            self._measurement_matrix_C[-1 * i, -1 * i] = 1

        self._measurement_noise_RdiagonalValues_basic = \
            np.square(np.array([100.0, 10.0, 10.0, 200.0, 10.0,10.0,10.0 ]))
        #camera position noise shouldn't matter too much, since the process noise is zero
        #This means that even if our expected noise is close to zero, measurements would
        #be seen as unlikely values but still average nicely.
        #All we need to ensure is that the initial uncertainty is not over-precise.
        self._measurement_noise_RdiagonalValues_tight = \
            np.square(np.array([2.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0])) # We _believe_ these measurements
        # TODO: orders of magnitude off, likely


        ### PRIOR BELIEF ON CALIBRATION

        #Now we initialize our probability before we take the first measurement.
        mean_pan = 0 #+/- 90
        mean_tilt = 80.0 #81.6 at an end zone two measurements see 81.6 and 81.7
        mean_roll = 0.0 #We always seem to be +/-2, or 4 degrees
        mean_f = 1400 #two measurements see a spread of 100 from that each
        meanMutable = [mean_pan, mean_tilt, mean_roll, mean_f]

        stdMutable = [ 60.0, 10.0, 5.0, 200.0]

        meanVelAcc = [0]*nTriples*2
        stdVel = [0.6, 0.6, .1, 10.0]   #About 50 degrees in 80 clicks is air-speed ball, or 0.6
                                            #no clue on the others--decent tilt, small roll, moderate zoom
        stdAcc = [.05, .01, .01, .1]     #Who knows?

        meanPos =  [0.0, 45.0, -9.75]
        #[4.779806891368249, 55.83919091121594, -9.739767239117437]
        #[-4.096765672149999, 38.46412085563827, -9.76131665631739]
        stdPos = [10., 20., 10.]

        mean = np.array(meanMutable + meanVelAcc + meanPos)
        std  = np.array(stdMutable  + stdVel + stdAcc + stdPos)


        self.initial_mean_vector = mean
        self.initial_cov_matrix  = np.diag(np.square(std))
        self.mean_vectors = []
        self.cov_matrices  = []


    def measDictToArray(self,dict):
        """ Extract measured parameters from a camera dictionary
        Parameters
        ---------
        dict : dict
            A camera dictionary, similar to json format of opencv
        Returns
        -------
        ndarray
            Returns a vectorized form of the measurement,
            vals, velocities, accels, X1pos,X2pos,X3pos of camera
        """
        rr = [float(dict[key]) for key in ["pan_degrees", "tilt_degrees", "roll_degrees", "x_focal_length"]]
        rr.extend(list(dict["position_meters"]))

        return np.array(rr)

    def meanArrayToDict(self, mean, measDict=None):
        """ Return a camera calibration dictionary
        Parameters
        ---------
        mean : ndarray
            A 7-D measurement-style np vector of pan, tilt, roll, focal, X1,X2,X3
        measDict : dict
            A possibly pre-existing camera calibration dict
        Returns
        -------
        dict
            The camera calibration dict modifiedto reflect the inputx
        """
        if measDict is None:
            measDict = {}
        for i,key in enumerate(["pan_degrees", "tilt_degrees", "roll_degrees", "x_focal_length"]):
            measDict[key] = mean[i]
        measDict["position_meters"] = list(mean[-3:])

        return measDict


    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 17 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 17x17 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state.
        """
        mean = self._motion_matrix_A @ mean

        #Take diagonal values for the matrix -- this assumes independence
        motion_cov = np.diag( self._process_noise_QdiagonalValues )
        #The below is adding only diagonal values -- independence --> zero everywhere else
        covariance = self._motion_matrix_A @ covariance @ self._motion_matrix_A.T + motion_cov

        return mean, covariance


    def update_from_measurement(self, latent_mean, latent_covariance, measurement, pseudo_confidence_level = "basic"):
        """Reduce the full latent state space to the measured.
        Parameters
        ----------
        latent_mean : ndarray
            Full state space.
        latent_covariance : ndarray
            Full state space covariance.
        measurement : dict
            "pan_degrees"
            "tilt_degrees"
            "roll_degrees"
            "x_focal_length"
            3x: "position_meters"
        pseudo_confidence_level :
            How much we believe our measurement at this point.
            We use this to define the kind of observation we think it is,
            a low-noise observation vs a high-noise one.
        Returns
        ------
        (ndarray, ndarray)
            Returns the predicted mean and covariance for an observation
            of the underlying latent state space.
        """

        #y_t+1 = C @ mu_t+1
        predicted_observed_mean = self._measurement_matrix_C @ latent_mean

        measurement = self.measDictToArray(measurement)
        innovation_z = measurement - predicted_observed_mean

        if pseudo_confidence_level == "basic":
            observation_noise = self._measurement_noise_RdiagonalValues_basic
        elif pseudo_confidence_level == "tight":
            observation_noise = self._measurement_noise_RdiagonalValues_tight
        else:
            observation_noise = self._measurement_noise_RdiagonalValues_basic

        # innovation_cov_Lzz = latent_covariance + np.diag(observation_noise)
        innovation_cov_Lzz = self._measurement_matrix_C @ latent_covariance @ self._measurement_matrix_C.T \
                         + np.diag(observation_noise) #equivalent to the above line

        ### Putting it together
        Lxz = latent_covariance @ self._measurement_matrix_C.T
        # Kalman_gain = Lxz @ np.invert(innovation_cov_Lzz)
        chol_factor, lower = scipy.linalg.cho_factor(
            innovation_cov_Lzz, lower=True, check_finite=False) #Note: should be .T, but that's equivalent
        Kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), Lxz.T,
            check_finite=False).T

        updated_latent_mean = latent_mean + Kalman_gain @ innovation_z
        updated_latent_cov_matrix = latent_covariance - Kalman_gain @ innovation_cov_Lzz @ Kalman_gain.T

        return updated_latent_mean, updated_latent_cov_matrix


    def forwardBackwardInference(self, allMeasurements):
        """ Infer the latent self.mean_vectors and self.cov_matrices
        Parameters
        ---------
        allMeasurements : iterable
            Lots of dictionaries of camera calibrations
        Returns
        -------
        (list)
            A list of camera dicts, with identical parameters
            as the first given one except for inferred mean vals
            on pan,tilt,roll,zoom, position. The distortion vals,
            for example, are not changed.
        """
        self.allFilter(allMeasurements)
        self.allSmooth(allMeasurements)

        prototypicalDict = allMeasurements[0]
        allMeanCalibrations = [self.meanArrayToDict(mu,prototypicalDict) for mu in self.mean_vectors]
        return allMeanCalibrations

    def allFilter(self, allMeasurements):
        for i,measurement in allMeasurements:
            if i == 0:
                beliefMean, beliefCov = self.initial_mean_vector, self.initial_cov_matrix
            else:
                beliefMean, beliefCov = self.predict(self.mean_vectors[-1], self.cov_matrices[-1])

            mean, cov = self.update_from_measurement(beliefMean, beliefCov, measurement, "basic")
            self.mean_vectors.append(mean)
            self.cov_matrices.append(cov)

    def allSmooth(self, allMeasurements):
        print("Rauch-Tung-Striebel Algorithm TODO")
        pass


    # def initiate(self, measurement):
    #     """Create track from unassociated measurement.
    #     Parameters
    #     ----------
    #     measurement : ndarray
    #         Bounding box coordinates (x, y, a, h) with center position (x, y),
    #         aspect ratio a, and height h.
    #     Returns
    #     -------
    #     (ndarray, ndarray)
    #         Returns the mean vector (8 dimensional) and covariance matrix (8x8
    #         dimensional) of the new track. Unobserved velocities are initialized
    #         to 0 mean.
    #     """
    #     mean_pos = measurement
    #     mean_vel = np.zeros_like(mean_pos)
    #     mean = np.r_[mean_pos, mean_vel]
    #
    #     std = [
    #         2 * self._std_weight_position * measurement[3],
    #         2 * self._std_weight_position * measurement[3],
    #         1e-2,
    #         2 * self._std_weight_position * measurement[3],
    #         10 * self._std_weight_velocity * measurement[3],
    #         10 * self._std_weight_velocity * measurement[3],
    #         1e-5,
    #         10 * self._std_weight_velocity * measurement[3]]
    #     covariance = np.diag(np.square(std))
    #     return mean, covariance

    # def project(self, mean, covariance):
    #     """Project state distribution to measurement space.
    #     Parameters
    #     ----------
    #     mean : ndarray
    #         The state's mean vector (8 dimensional array).
    #     covariance : ndarray
    #         The state's covariance matrix (8x8 dimensional).
    #     Returns
    #     -------
    #     (ndarray, ndarray)
    #         Returns the projected mean and covariance matrix of the given state
    #         estimate.
    #     """
    #     std = [
    #         self._std_weight_position * mean[3],
    #         self._std_weight_position * mean[3],
    #         1e-1,
    #         self._std_weight_position * mean[3]]
    #     innovation_cov = np.diag(np.square(std))
    #
    #     mean = np.dot(self._update_mat, mean)
    #     covariance = np.linalg.multi_dot((
    #         self._update_mat, covariance, self._update_mat.T))
    #     return mean, covariance + innovation_cov

    # def gating_distance(self, mean, covariance, measurements,
    #                     only_position=False, metric='maha'):
    #     """Compute gating distance between state distribution and measurements.
    #     A suitable distance threshold can be obtained from `chi2inv95`. If
    #     `only_position` is False, the chi-square distribution has 4 degrees of
    #     freedom, otherwise 2.
    #     Parameters
    #     ----------
    #     mean : ndarray
    #         Mean vector over the state distribution (8 dimensional).
    #     covariance : ndarray
    #         Covariance of the state distribution (8x8 dimensional).
    #     measurements : ndarray
    #         An Nx4 dimensional matrix of N measurements, each in
    #         format (x, y, a, h) where (x, y) is the bounding box center
    #         position, a the aspect ratio, and h the height.
    #     only_position : Optional[bool]
    #         If True, distance computation is done with respect to the bounding
    #         box center position only.
    #     Returns
    #     -------
    #     ndarray
    #         Returns an array of length N, where the i-th element contains the
    #         squared Mahalanobis distance between (mean, covariance) and
    #         `measurements[i]`.
    #     """
    #     mean, covariance = self.project(mean, covariance)
    #     if only_position:
    #         mean, covariance = mean[:2], covariance[:2, :2]
    #         measurements = measurements[:, :2]
    #
    #     d = measurements - mean
    #     if metric == 'gaussian':
    #         return np.sum(d * d, axis=1)
    #     elif metric == 'maha':
    #         cholesky_factor = np.linalg.cholesky(covariance)
    #         z = scipy.linalg.solve_triangular(
    #             cholesky_factor, d.T, lower=True, check_finite=False,
    #             overwrite_b=True)
    #         squared_maha = np.sum(z * z, axis=0)
    #         return squared_maha
    #     else:
    #         raise ValueError('invalid distance metric')
