#################################
# Your name: nizan shami
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                    A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        X = np.sort(np.random.uniform(size=m))
        Y = [self.get_lable(point) for point in X]

        return np.array([X,Y])


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        true_error_avg = []
        empirical_error_avg = []
        for m in range(m_first, m_last+ 1, step):
            empirical_error = 0
            true_error = 0    
            for times in range(T):
                data_semple = self.sample_from_D(m)
                interval_set, e_s = intervals.find_best_interval(data_semple[0], data_semple[1], k)
                empirical_error += e_s
                true_error += self.calculate_true_error(interval_set)
            true_error_avg.append(true_error/T)
            empirical_error_avg.append(empirical_error/(T*m))


        plt.plot(range(m_first, m_last+ 1, step), empirical_error_avg, label='Empirical Error')
        plt.plot(range(m_first, m_last+ 1, step), true_error_avg, label='True Error')
        plt.legend()
        plt.xlabel("n")
        plt.ylabel("Empirical Error")
        plt.show()



    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        true_error = []
        empirical_error = []
        data = self.sample_from_D(m)
        for k in range(k_first, k_last + 1, step):
            interval_set ,e_s = intervals.find_best_interval(data[0], data[1], k)
            empirical_error.append(e_s/m)
            true_error.append(self.calculate_true_error(interval_set))
        
        plt.plot(range(k_first, k_last + 1, step), empirical_error, label='Empirical Error')
        plt.plot(range(k_first, k_last + 1, step), true_error, label='True Error')
        plt.legend()
        plt.xlabel("k")
        plt.show()
        return np.argmin(empirical_error)

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        true_error = []
        empirical_error = []
        penalty = []
        data = self.sample_from_D(m)
        for k in range(k_first, k_last + 1, step):
            penalty.append(self.penalty(k, m))
            interval_set ,e_s = intervals.find_best_interval(data[0], data[1], k)
            empirical_error.append(e_s/m)
            true_error.append(self.calculate_true_error(interval_set))
        
        sum_of_penalty_and_emp_error = np.array(penalty) + np.array(empirical_error)
        plt.plot(range(k_first, k_last + 1, step), empirical_error, label='Empirical Error')
        plt.plot(range(k_first, k_last + 1, step), true_error, label='True Error')
        plt.plot(range(k_first, k_last + 1, step), penalty, label='penalty')
        plt.plot(range(k_first, k_last + 1, step), sum_of_penalty_and_emp_error, label='Sum Of Penalty and Empirical Error')
        plt.legend()
        plt.xlabel("k")
        plt.show()
        return np.argmin(empirical_error)

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        sample = self.sample_from_D(m)
        np.random.shuffle(sample)
        train = [(sample[0][i], sample[1][i]) for i in range(1200)]
        train.sort(key=lambda x: x[0])
        validation = [(sample[0][i], sample[1][i]) for i in range(1201, 1500)]
        corss_erm = []
        xt = [x[0] for x in train]
        yt = [y[1] for y in train]
        xv = [x[0] for x in validation]
        yv = [y[1] for y in validation]
        for k in range(1, 11):
            ERM_k_intervals, empirical_error_k = intervals.find_best_interval(xt, yt, k)
            corss_erm.append(self.validation_error(ERM_k_intervals, xv, yv))
        return np.argmin(corss_erm)
    #################################
    def get_lable(self, point):
        if (point <= 0.2) or (point >= 0.4 and point <= 0.6) or (point > 0.8):
            return np.random.choice([0, 1], p=[0.2, 0.8])
        else:
            return np.random.choice([0, 1], p=[0.9, 0.1])  
    
    def intersection(self, intervals_set, I):
        """
        returns the intersection of the intervals with the interval I
        """
        sum = 0
        for interval in intervals_set:
            a = max(I[0], interval[0])
            b = min(I[1], interval[1])
            if a < b:
                sum += (b - a)
        return sum

    def union(self, interval_set):
        union = 0
        for interval in interval_set:
            union += interval[1] - interval[0]
        return union

    def find_complements(self, interval_set):
        interval_set.insert(0, (0, 0))
        interval_set.append((1, 1))
        complements = []
        for i in range(len(interval_set) - 1):
            complements.append((interval_set[i][1], interval_set[i + 1][0]))
        return complements
    

    def calculate_true_error(self, intervals_set):
        """
        X1 = [0,0.2]U[0.4,0.6]U[0.8,1]
        X2 = [0.2,0.4]U[0.6,0.8]
        returns the true error of h_(interval_set)
        """

        sumX1 = 0
        sumX2 = 0
        complements_set = self.find_complements(intervals_set)
        union_complements = self.union(complements_set)
        union_intervals = self.union(intervals_set)

        sumX1 += self.intersection(intervals_set, [0, 0.2])
        sumX1 += self.intersection(intervals_set, [0.4, 0.6])
        sumX1 += self.intersection(intervals_set, [0.8, 1])
        expectation = sumX1 * 0.2 + (union_intervals-sumX1) * 0.9

        sumX2 += self.intersection(complements_set, [0.2, 0.4])
        sumX2 += self.intersection(complements_set, [0.6, 0.8])
        expectation += sumX2 * 0.1 + (union_complements-sumX2) * 0.8

        return expectation
    
    def penalty(self, k, n):
        arg = (2*k + np.log(20))/n
        return 2*np.sqrt(arg)
    
    def validation_error(self, ERM_k_intervals, validation_points , validation_label):
        error = 0.0
        for i in range(len(validation_points)):
            predication_x = 0
            for interval in ERM_k_intervals:
                # x is in an interval, so it's predication would be 1
                if validation_points[i] >= interval[0] and validation_points[i] >= interval[1]:
                    predication_x += 1
                    break  # no need to check other intervals
                if validation_points[i] < interval[0]:
                    break  # if x is before start of some interval. if after all of them - end of the loop
            error += (predication_x != validation_label[i])
        return error/len(validation_points)
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

