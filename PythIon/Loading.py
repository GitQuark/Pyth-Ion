import os
import pandas.io.parsers
import numpy as np
from scipy import io as spio


# Loads the log file data and outputs a vector of voltages and the sample rate
from PythIon.abfheader import read_header


def load_log_file(file_name, data_directory):
    def data_to_amps(raw_data, adc_bits, adc_vref, closedloop_gain, current_offset):
        # Computations to turn uint16 data into amps
        bitmask = (2 ** 16) - (1 + ((2 ** (16 - adc_bits)) - 1))
        raw_data = -adc_vref + ((2 * adc_vref * (raw_data & bitmask)) / 2 ** 16)
        raw_data = (raw_data / closedloop_gain + current_offset)
        data = raw_data[0]  # Retrurns the list to a single level: [[data]] -> [data]
        return data

    data_file_name = os.path.join(data_directory, file_name + '.log')
    info_file_name = os.path.join(data_directory, file_name + '.mat')

    mat = spio.loadmat(info_file_name)
    # ADC is analog to digital converter
    # Loading in data about file from matlab data file
    sample_rate = mat['ADCSAMPLERATE'][0][0]
    ti_gain = mat['SETUP_TIAgain']
    pre_adc_gain = mat['SETUP_preADCgain'][0][0]
    current_offset = mat['SETUP_pAoffset'][0][0]
    adc_vref = mat['SETUP_ADCVREF'][0][0]
    adc_bits = mat['SETUP_ADCBITS']
    closedloop_gain = ti_gain * pre_adc_gain
    # Info has been loaded

    chimera_file = np.dtype('uint16')  # Was <u2 "Little-endian 2 byte unsigned integer"
    raw_data = np.fromfile(data_file_name, chimera_file)
    # Part to handle low sample rate
    # if sample_rate < 4000e3:
    #     raw_data = raw_data[::round(sample_rate / output_sample_rate)]
    data = data_to_amps(raw_data, adc_bits, adc_vref, closedloop_gain, current_offset)
    data_params = {
        'sample_rate': sample_rate
    }
    return data, data_params


def load_opt_file(file_name, working_dir=os.getcwd()):
    data_file_name = os.path.join(working_dir, file_name + '.opt')
    data = np.fromfile(data_file_name, dtype=np.dtype('>d'))
    # mat_path = '//'.join([data_directory, file_name])
    # if file_name + '_inf.mat' in os.listdir(data_directory):
    #     extention = '_inf.mat'
    #     mat = spio.loadmat(mat_path + extention)[file_name][0][0]
    #     rate_multiplier = 1
    # elif file_name + '.mat' in os.listdir(data_directory):
    #     extention = '.mat'
    #     mat = spio.loadmat(mat_path + extention)[mat_path]
    #     rate_multiplier = 1000
    #     # potential = np.float64(mat['potential'])
    #     # pre_trigger_time_ms = np.float64(mat['pretrigger_time'])
    #     # post_trigger_time_ms = np.float64(mat['posttrigger_time'])
    #
    #     # trigger_data = mat['triggered_pulse']
    #     # start_voltage = trigger_data[0].initial_value
    #     # final_voltage = trigger_data[0].ramp_target_value
    #     # ramp_duration_ms = trigger_data[0].duration
    #     # eject_voltage = trigger_data[1].initial_value
    #     # eject_duration_ms = np.float64(trigger_data[1].duration)
    # else:
    #     return
    # filt_rate = float(mat['filterfreq']) * rate_multiplier
    # sample_rate = float(mat['sample_rate'])

    # # print("data sampled at lower rate than requested, reverting to original sampling rate")
    # # rate_text = str(round(sample_rate) / 1000)
    # # ui.outputsamplerateentry.setText(rate_text)
    # output_sample_rate = min(output_sample_rate, sample_rate)
    #
    # # sample rate can not be >250kHz for axopatch files, displaying with a rate of 250kHz
    # output_sample_rate = min(output_sample_rate, 250000)
    #
    # # print('Already LP filtered lower than or at entry, data will not be filtered')
    # low_pass_cutoff = min(low_pass_cutoff, filt_rate)
    # # ui.LPentry.setText(str((round(low_pass_cutoff) / 1000)))
    #
    # nyquist_freq = 100 * 10 ** 3 / 2
    # wn = round(low_pass_cutoff / nyquist_freq, 4)
    # # noinspection PyTupleAssignmentBalance
    # b, a = signal.bessel(4, wn, btype='low')
    # data = signal.filtfilt(b, a, data)
    data_params = {}
    return data, data_params


def load_txt_file(file_name, working_dir=os.getcwd()):
    data_file_name = os.path.join(working_dir, file_name + 'txt')
    data = pandas.io.parsers.read_csv(data_file_name, skiprows=1)
    # data = np.reshape(np.array(data),np.size(data))*10**9
    data = np.reshape(np.array(data), np.size(data))
    data_params = {}
    return data, data_params


def load_npy_file(file_name, working_dir=os.getcwd()):
    data_file_name = os.path.join(working_dir, file_name + 'txt')
    data = np.load(data_file_name)
    data_params = {}
    return data, data_params


def load_abf_file(file_name, working_dir=os.getcwd()):
    abf_header_size = 6144
    data_file_name = os.path.join(working_dir, file_name + 'txt')
    f = open(data_file_name, "rb")  # reopen the file
    f.seek(abf_header_size, os.SEEK_SET)
    data = np.fromfile(f, dtype=np.dtype('<i2'))
    header = read_header(data_file_name)
    telegraph_mode = int(header['listADCInfo'][0]['nTelegraphEnable'])  # Not sure if this needs to be an int
    if telegraph_mode:
        gain = header['listADCInfo'][0]['fTelegraphAdditGain']
        data = data.astype(float) * (20. / (65536 * gain)) * 10 ** -9
    else:
        data = data.astype(float) * (20 / 65536) * 10 ** -9
    if len(header['listADCInfo']) == 2:
        # v = data[1::2] * gain / 10
        data = data[::2]

    # telegraph_mode = int(header['listADCInfo'][0]['nTelegraphEnable'])
    sample_rate = 1e6 / header['protocol']['fADCSequenceInterval']
    # if telegraph_mode:
    #     abf_low_pass = header['listADCInfo'][0]['fTelegraphFilter']
    # else:
    #     abf_low_pass = sample_rate

    # output sample_rate can not be higher than sample_rate, resetting to original rate.
    # output_sample_rate = min(output_sample_rate, sample_rate)
    # ui.outputsamplerateentry.setText(str((round(sample_rate) / 1000)))

    # Already LP filtered lower than or at entry, data will not be filtered
    # low_pass_cutoff = min(low_pass_cutoff, abf_low_pass)
    # ui.LPentry.setText(str((round(low_pass_cutoff) / 1000)))

    # else:
    #     wn = round(low_pass_cutoff / (100 * 10 ** 3 / 2), 4)
    #     # noinspection PyTupleAssignmentBalance
    #     b, a = signal.bessel(4, wn, btype='low')
    #     data = signal.filtfilt(b, a, data)

    # tags = header['listTag']
    # for tag in tags:
    #     if tag['sComment'][0:21] == "Holding on 'Cmd 0' =>":
    #         cmdv = tag['sComment'][22:]
    #         # cmdv = [int(s) for s in cmdv.split() if s.isdigit()]
    #         cmdt = tag['lTagTime'] / output_sample_rate
    #         p1.addItem(pg.InfiniteLine(cmdt))
    #         # cmdtext = pg.TextItem(text = str(cmdv)+' mV')
    #         cmd_text = pg.TextItem(text=str(cmdv))
    #         p1.addItem(cmd_text)
    #         cmd_text.setPos(cmdt, np.max(data))
    data_params = {
        'sample_rate': sample_rate
    }
    return data, data_params
