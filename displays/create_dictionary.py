import os
import sys
import numpy as np

external_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(external_path)
import helper_functions

class DictionaryCreator:
    def __init__(self, output_folder):
        script_dir = os.path.abspath(os.path.dirname( __file__ ))
        self.output_path = f"{script_dir}/{output_folder}"

    def generate_4digit_numbers(self, lowerlimit, upperlimit, dict, dict_name):
        values = set()
        lowerlimit = lowerlimit*1000 - 1
        upperlimit = upperlimit*1000 + 1

        for i in range(lowerlimit, upperlimit):
            val = i / 1000 

            # Format value to up to 3 decimal places
            formatted = f"{abs(val):.3f}".rstrip('0').rstrip('.')

            if len(formatted) == 3:
                formatted = f"{formatted}.0"

            # Count digits ignoring dot and minus sign
            digit_str = formatted.replace('.', '')
            if len(digit_str) == 4:
                # Pad leading zeros if needed
                if '.' in formatted:
                    integer_part, decimal_part = formatted.split('.')
                    needed = 4 - len(integer_part + decimal_part)
                    integer_part = integer_part.zfill(len(integer_part) + needed)
                    padded = f"{integer_part}.{decimal_part}"
                else:
                    padded = formatted.zfill(4)

                if val < 0:
                    padded = '-' + padded

                values.add(padded)

        # Sort numerically and write to file
        for value in sorted(values, key=lambda x: float(x)):
            dict.write(value + '\n')

    def select_dict(self, dict, dict_name):
        match dict_name:
            case 'metronome':
                for i in range(30, 251):
                    dict.write(str(i) + "\n")

            case 'multimeter_range_1000':            
                self.generate_4digit_numbers(-1000, 1000, dict, dict_name)            
            
            case 'multimeter_range_600':
                self.generate_4digit_numbers(-600, 600, dict, dict_name)

            case 'multimeter_unilateral_range_1000':
                self.generate_4digit_numbers(0, 1000, dict, dict_name)

            case 'multimeter_unilateral_range_60':
                self.generate_4digit_numbers(0, 60, dict, dict_name)

            case 'multimeter_range_750':
                self.generate_4digit_numbers(-750, 750, dict, dict_name)

            case 'multimeter_range_10':
                self.generate_4digit_numbers(-10, 10, dict, dict_name)

            case 'multimeter_temperature celsius':
                for i in range(-40, 1001):
                    if i < 0:
                        dict.write(f'{i:05}' + "\n")
                    else:
                        dict.write(f'{i:04}' + "\n")
            
            case 'multimeter_temperature farenheit':
                for i in range(-40, 1833):
                    if i < 0:
                        dict.write(f'{i:05}' + "\n")
                    else:
                        dict.write(f'{i:04}' + "\n")
            
            case 'multimeter_duty':
                for i in range(1, 100):
                    dict.write(f'{i:03}' + "\n")
                
            case 'power_supply_voltage':
                for i in range(0, 6001):
                    value = i / 100  # Increments of 0.01
                    dict.write(f"{value:05.2f}\n")  # total width of 5 with 2 decimal values

            case 'power_supply_current':
                for i in range(0, 5001):
                    value = i / 1000  # Increments of 0.001 from 0 to 5.000
                    dict.write(f"{value:05.3f}\n")  # total width of 5 with 3 decimal values

            case 'oximeter_spo2':
                for i in range(70, 101):
                    dict.write(str(i) + "\n")

            case 'oximeter_pi':
                for i in range(3, 201):
                    value = i / 10  # Increments of 0.1
                    dict.write(f"{value:02.1f}\n")  # total width of 5 with 3 decimal values

            case 'thermometer_temperature celsius':
                for i in range(340, 423):
                    value = i / 10  # Increments of 0.1
                    dict.write(f"{value:02.1f}\n")
            
            case 'thermometer_hour':
                for hour in range(24):
                    for minute in range(60):
                        time_str = f"{hour:02}:{minute:02}"
                        dict.write(time_str + "\n")
            
            case 'blood_pressure_device_systolic':
                for i in range(85, 181):
                    dict.write(str(i) + "\n")

            case 'blood_pressure_device_diastolic':
                for i in range(50, 81):
                    dict.write(str(i) + "\n")
            
            case 'blood_pressure_device_heart beat':
                for i in range(40, 211):
                    dict.write(str(i) + "\n")


    def create_dict(self, models_folder, dict_list_json='dicts_list.json', device_name=None):
        if device_name == None:
            dict_list = helper_functions.load_json(os.path.join(self.output_path, dict_list_json))
            for model in os.listdir(models_folder):
                if model.endswith(".blend"):
                    device_name = os.path.splitext(model)[0]
                    for subname in dict_list[device_name]:
                        if subname == device_name:
                            dict = open(f"{self.output_path}/{device_name}.txt", "w")
                            self.select_dict(dict, dict_name=f"{device_name}")
                            print(f'The {device_name} dictionary was created!')
                        else:
                            dict = open(f"{self.output_path}/{device_name}_{subname}.txt", "w")
                            self.select_dict(dict, dict_name=f"{device_name}_{subname}")
                            print(f'The {device_name}_{subname} dictionary was created!')

        else:
            dict = open(f"{self.output_path}/{device_name}.txt", "w")
            self.select_dict(dict, dict_name=f"{device_name}")
            print(f'The {device_name} dictionary was created!')

        

                

if __name__ == "__main__":
    dictionary = DictionaryCreator(output_folder="dicts")
    dictionary.create_dict(models_folder='../models')
    
    print('Dictionary Generation Stage Complete! ✅')
