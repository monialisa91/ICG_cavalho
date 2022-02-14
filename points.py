import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pantomkins import pt


class points():

    def __init__(self, data_ecg, data_icg, fs):
        self.data_ecg = data_ecg
        self.data_icg = data_icg
        self.fs = fs

    def R_peak_detection(self):
        data_pt = self.data_ecg
        peaks = find_peaks(data_pt, distance=150)[0]
        values = data_pt[np.array(peaks)]
        maksimum = np.sort(values)[-2:]
        thr = 0.8 * np.mean(maksimum)
        peaks_thr = np.where(values>thr)
        peaks_thr2 = peaks[peaks_thr]

        '''plt.plot(np.arange(len(data_pt)), data_pt)
        plt.scatter(peaks_thr2, data_pt[peaks_thr2])
        plt.axhline(thr)
        plt.show()'''

        return peaks_thr2


    def C_point_detection(self):
        R_points = self.R_peak_detection()
        C_points = []
        avg_time = []
        for i in range(len(R_points)-1):
            point0 = R_points[i]
            pointk = R_points[i+1]
            time = pointk - point0
            avg_time.append(time)
            cc = self.data_icg[point0:pointk]
            C_point = np.argmax(cc) + point0
            C_points.append(C_point)

        cc_time_avg = int(np.mean(np.array(avg_time)))
        cc_last = self.data_icg[R_points[-1]:(R_points[-1]+cc_time_avg)]
        C_point = np.argmax(cc_last) + R_points[-1]
        C_points.append(C_point)

        return np.array(C_points)

    def Bisection(self, i, percent):
        '''
        :param i: number of cycle indexed by the C point in the list of Cpoints
        :param percent: percent of the value of the ICG signal in the C point
        :return:
        '''
        Cpoint = self.C_point_detection()[i]
        Value = percent * self.data_icg[Cpoint]
        eps = 0.01  # admissible error
        b = Cpoint
        a = Cpoint - 64
        c = int((a + b) / 2)
        fc = self.data_icg[c]
        for i in range(5):
            if (abs(fc - Value) < eps):
                break
                return c
            elif (fc > Value):
                b = c
                c = int((a + b) / 2)
                fc = self.data_icg[c]
            else:
                a = c
                c = int((a + b) / 2)
                fc = self.data_icg[c]
        return c

    def lineFit(self, i):
        '''
        :param i: number of cycle indexed by the C point in the list of Cpoints
        '''
        B1 = self.Bisection(i, 0.4)
        B2 = self.Bisection(i, 0.8)
        fB1 = self.data_icg[B1]
        fB2 = self.data_icg[B2]

        a = (fB2 - fB1) / (B2 - B1)
        b = fB1 - a * B1
        cross_point = -b / a  # point of intersection with the horizontal axis
        return cross_point

    def signPattern(self, i):
        Cpoint = self.C_point_detection()[i]
        first_der = np.gradient(self.data_icg)
        second_der = np.gradient(first_der)
        changes = 0 # number of sign changes of the second derivative
        if (second_der[Cpoint] > 0):
            return False
        for i in range(Cpoint, Cpoint-50, -1):
            if(second_der[i] * second_der[i+1] < 0):
                changes += 1
            if(changes == 3):
                return True
        return False

    def find_minimum(self, B0, td):
        B0 = int(B0)
        for i in range(B0, B0-100, -1):
            ip = td[i] # initial point
            mp = td[i+1] # middle point
            fp = td[i+2] # final point
            if(mp<ip and mp<fp):
                return i+1
        return None

    def crossing(self, B0, fd):
        '''
        in the case of an absence of the pattern the B point is estimated as the first zero-crossing of the first order
        derivative left to the B0
        :return:
        '''
        B0 = int(B0)
        eps = 0.01
        for i in range(B0, B0 - 100, -1):
            if (abs(fd[i]) < eps):
                return i
        return None


    def B_point_detection(self):
        Cpoints = self.C_point_detection()
        fd = np.gradient(self.data_icg)  # first derivative
        sd = np.gradient(fd)  # second derivative
        td = np.gradient(sd)  # third derivative
        Bpoints = []
        for i in range(len(Cpoints)):
            B0 = self.lineFit(i)
            sp = self.signPattern(i)
            if (sp == True):
                B = self.find_minimum(B0, td)
                Bpoints.append(B)
            else:
                B = self.crossing(B0, fd)
                Bpoints.append(B)
        return np.array(Bpoints)

    def T_point_detection(self):
        data = self.data_ecg
        R_points = self.R_peak_detection()
        RR_ints = []
        T_points = []
        for i in range(len(R_points) - 1):
            R_start = R_points[i]
            R_end = R_points[i + 1]
            RR_interval = R_end - R_start
            RR_ints.append(RR_interval)
            peaks, _ = find_peaks(data[(R_start + 1): (R_start + 1 + int(1 /2 * RR_interval))])
            peaks = peaks + R_start + 1
            peak_T = np.argmax(data[peaks])
            # T_point = np.argmax(data[R_start + 1: (R_start + 1 + int(1 / 3 * RR_interval))]) + R_start + 1
            T_points.append(peaks[peak_T])

        # the last point
        RR_interval = np.mean(np.array(RR_ints))
        R_start = R_points[-1]
        peaks, _ = find_peaks(data[(R_start + 1): (R_start + 1 + int(1 / 3 * RR_interval))])
        peaks = peaks + R_start + 1
        peak_T = np.argmax(data[peaks])
        T_points.append(peaks[peak_T])
        return np.array(T_points)

    def X_points_detection(self):
        Tpoints = self.T_point_detection()
        Rpoints = self.R_peak_detection()
        fd = np.gradient(self.data_icg)  # first derivative
        sd = np.gradient(fd)  # second derivative
        td = np.gradient(sd)  # third derivative
        Xpoints = []
        for i in range(len(Tpoints)):
            print(i)
            R_point = Rpoints[i]
            T_point = Tpoints[i]
            RT = T_point - R_point
            data_int = self.data_icg[(R_point + RT): (R_point + int(1.75*RT))]
            X0 = int(np.argmin(data_int) + R_point + RT)
            minimum = False
            k = 0
            while(minimum == False):
                fp = td[X0-k]
                mp = td[X0-k-1]
                lp = td[X0-k-2]
                k += 1
                if(mp < fp and mp < lp):
                    minimum = True
            Xpoints.append(X0-k-1)
        return np.array(Xpoints)


