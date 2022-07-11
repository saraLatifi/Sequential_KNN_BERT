import time
import os

class Time_usage_training:

    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def init(self):
        self.start_time = 0
        self.end_time = 0

        # cpu time
        self.start_time_cpu = 0
        self.end_time_cpu = 0

    def start(self):
        self.start_time = 0
        self.end_time = 0
        self.start_time = time.time()

        # cpu time
        self.start_time_cpu = 0
        self.end_time_cpu = 0
        self.start_time_cpu = time.process_time()

    def stop(self):
        self.end_time = time.time()
        self.end_time_cpu = time.process_time()  # cpu time

    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''

        training_time = self.end_time - self.start_time
        training_time_cpu = self.end_time_cpu - self.start_time_cpu
        return ("Training time:", "Training time CPU:"), (training_time, training_time_cpu)

    def reset(self):
        pass

    def write_training_times_csv(self, results, key, conf=None, checkpointDir=None):
        '''
        Write the result array to a csv file, if a result folder is defined in the configuration
            --------
            results : tuple
        '''
        export_csv = None
        if checkpointDir is not None:  # for bert4rec
            export_csv = checkpointDir + '/running_times/' + 'bert4rec_training_time.csv'
        elif conf is not None and 'results' in conf and 'folder' in conf['results']:  # for KNNs
            export_csv = conf['results']['folder'] + 'training_times/' + 'test_' + conf['type'] + '_' + conf[
                'key'] + '_' + conf['data']['name'] + '.csv'

        if export_csv is not None:
            # Create all directories in the file_path if non-existent.
            directory = os.path.dirname(export_csv)
            if not os.path.exists(directory):
                os.makedirs(directory)

            file = open(export_csv, 'w+')
            file.write('Metrics;')

            file.write(results[0][0])  # Training time
            file.write(';')
            file.write(results[0][1])  # Training time cpu
            file.write(';')
            file.write('\n')

            file.write(key)  # key (algorithm)
            file.write(';')

            file.write(str(results[1][0]))  # value
            file.write(';')
            file.write(str(results[1][1]))  # value (cpu)
            file.write(';')
            file.write('\n')


class Time_usage_testing:

    def __init__(self):
        pass

    def init(self):
        self.start_time = 0
        self.time_sum = 0

       # cpu time
        self.start_time_cpu = 0
        self.time_sum_cpu = 0

        # count
        self.time_count = 0

    def start_predict(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        self.start_time = time.time()

        # cpu time
        self.start_time_cpu = time.process_time();

    def stop_predict(self, batch_size=1):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        self.time_count += batch_size
        end_time = time.time()
        self.time_sum += end_time - self.start_time
        # cpu
        self.time_sum_cpu += time.process_time() - self.start_time_cpu

    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("Prediction time:", "Prediction time CPU:"), (
        self.time_sum / self.time_count, self.time_sum_cpu / self.time_count)

    def reset(self):
        pass

    def write_prediction_times_csv(self, results, key, conf=None, checkpointDir=None):
        '''
        Write the result array to a csv file, if a result folder is defined in the configuration
            --------
            results : tuple of tuples
        '''
        export_csv = None
        if checkpointDir is not None:  # for bert4rec
            export_csv = checkpointDir + '/running_times/' + 'bert4rec_prediction_time.csv'
        elif conf is not None and 'results' in conf and 'folder' in conf['results']:  # for KNNs
            export_csv = conf['results']['folder'] + 'prediction_times/' + 'test_' + conf['type'] + '_' + conf[
                'key'] + '_' + conf['data']['name'] + '.csv'

        if export_csv is not None:
            # Create all directories in the file_path if non-existent.
            directory = os.path.dirname(export_csv)
            if not os.path.exists(directory):
                os.makedirs(directory)

            file = open(export_csv, 'w+')
            file.write('Metrics;')

            file.write(results[0][0])  # prediction time
            file.write(';')
            file.write(results[0][1])  # prediction time cpu
            file.write(';')
            file.write('\n')

            file.write(key)  # key (algorithm)
            file.write(';')

            file.write(str(results[1][0]))  # value
            file.write(';')
            file.write(str(results[1][1]))  # value (cpu)
            file.write(';')
            file.write('\n')